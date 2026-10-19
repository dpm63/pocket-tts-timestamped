"""Teacher rollouts and steer-vector training.

Target: at BOS, the student (frozen model + one vector) should produce the
backbone output z the teacher had at its first sustained-speech frame k, i.e.
predict the onset frame directly instead of the leading silence. Diagnostics
also feed the teacher's onset latent a_k after BOS and compare the next
position with the teacher's z_{k+1} (the second speech frame), without
training on it.
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import THRESHOLD_DB, frame_db, onset
from .sequences import (
    decode_latents,
    prefix_embeddings,
    prefix_kv,
    rollout,
    sample_latent,
    student_forward,
    student_forward_cached,
    tokenize,
)
from .sites import Site, Steer

TENSOR_FIELDS = ["z0", "zk", "zk1", "eps_k", "eps_k1", "a_k", "a_k1", "eos0", "eosk"]


def make_rollouts(model, voice_latents: dict[int, torch.Tensor], pairs: list[tuple[int, str]], batch_size: int, n_frames: int, seed: int,
                  keep_prefix_kv: bool = False) -> dict:  # fmt: skip
    """Free-running teacher rollouts; keep rows that reach sustained speech.

    keep_prefix_kv: also keep each row's prefix keys/values (in memory, for
    cached training without recomputing the prefix).
    """
    gen = torch.Generator(device=model.flow_lm.bos_emb.device).manual_seed(seed)
    rec, skipped = defaultdict(list), 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        tokens = [tokenize(model, text) for _, text in chunk]
        prefixes = [
            prefix_embeddings(model, voice_latents[v], t)[0]
            for (v, _), t in zip(chunk, tokens, strict=True)
        ]
        r = rollout(model, prefixes, n_frames, gen, keep_prefix_kv=keep_prefix_kv)
        pcm = decode_latents(model, r["latent"]).cpu().numpy()
        for b, ((v, text), tok) in enumerate(zip(chunk, tokens, strict=True)):
            first, k = onset(frame_db(pcm[b]) > THRESHOLD_DB)
            if k < 0:
                skipped += 1
                continue
            rec["voice"].append(v)
            rec["text"].append(text)
            rec["tokens"].append(tok.cpu())
            rec["k"].append(k)
            rec["first_voiced"].append(first)
            if keep_prefix_kv:
                rec["prefix_kv"].append(r["prefix_kv"][b])
            for name, value in [
                ("z0", r["z"][b, 0]), ("zk", r["z"][b, k]), ("zk1", r["z"][b, k + 1]),
                ("eps_k", r["eps"][b, k]), ("eps_k1", r["eps"][b, k + 1]),
                ("a_k", r["latent"][b, k]), ("a_k1", r["latent"][b, k + 1]),
                ("eos0", r["eos_logit"][b, 0]), ("eosk", r["eos_logit"][b, k]),
            ]:  # fmt: skip
                rec[name].append(value.cpu())
        print(
            f"rollouts {start + len(chunk)}/{len(pairs)} kept={len(rec['k'])} skipped={skipped}",
            flush=True,
        )
    out = {name: torch.stack(rec[name]) if rec[name] else torch.empty(0) for name in TENSOR_FIELDS}
    out.update(
        voice=torch.tensor(rec["voice"]),
        k=torch.tensor(rec["k"]),
        first_voiced=torch.tensor(rec["first_voiced"]),
    )
    out.update(tokens=rec["tokens"], text=rec["text"], skipped=skipped)
    if keep_prefix_kv:
        out["prefix_kv"] = rec["prefix_kv"]
    return out


def concat_rollouts(parts: list[dict]) -> dict:
    out = {}
    for key in parts[0]:
        if key == "skipped":
            out[key] = sum(p[key] for p in parts)
        elif isinstance(parts[0][key], list):
            out[key] = [x for p in parts for x in p[key]]
        else:
            out[key] = torch.cat([p[key] for p in parts])
    return out


def subset_rollouts(data: dict, n: int) -> dict:
    """The first n rows (rollout pairs are drawn at random, so this is a random subset)."""
    out = {}
    for key, value in data.items():
        out[key] = value if key == "skipped" else value[:n]
    return out


class Batcher:
    """Builds student inputs for rows of a rollout set.

    store_prefix_kv: keep every row's prefix keys/values (taken from the
    rollouts when they kept them, else computed once) so cached training never
    recomputes the prefix. Without it, get_kv computes them per call.
    """

    def __init__(
        self,
        model,
        data: dict,
        voice_latents: dict[int, torch.Tensor],
        store_prefix_kv: bool = False,
        chunk: int = 32,
    ):
        self.model, self.data = model, data
        self.device = model.flow_lm.bos_emb.device
        self.voice_latents = {v: lat.to(self.device) for v, lat in voice_latents.items()}
        self.kv = data.get("prefix_kv")
        if store_prefix_kv and self.kv is None:
            self.kv = [
                kv
                for start in range(0, len(self), chunk)
                for kv in self._compute_kv(range(start, min(start + chunk, len(self))))
            ]

    def __len__(self) -> int:
        return len(self.data["k"])

    def _prefixes(self, idx) -> list[tuple[torch.Tensor, list[int]]]:
        d = self.data
        return [
            prefix_embeddings(
                self.model, self.voice_latents[int(d["voice"][i])], d["tokens"][i].to(self.device)
            )
            for i in idx
        ]

    def _compute_kv(self, idx) -> list[dict[str, torch.Tensor]]:
        return prefix_kv(self.model, [p for p, _ in self._prefixes(idx)])

    def get(self, idx: list[int], prefixes: bool = True) -> dict:
        d = self.data
        batch = {name: d[name][idx].to(self.device) for name in TENSOR_FIELDS}
        batch["k"] = d["k"][idx]
        if prefixes:
            built = self._prefixes(idx)
            batch.update(prefixes=[b[0] for b in built], kinds=[b[1] for b in built])
        return batch

    def get_kv(self, idx: list[int]) -> list[dict[str, torch.Tensor]]:
        if self.kv is not None:
            return [{name: t.to(self.device) for name, t in self.kv[i].items()} for i in idx]
        return self._compute_kv(idx)


def student_outputs(
    model, steer: Steer | None, batcher: Batcher, idx: list[int], cached: bool
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """(targets, z at BOS, z after the teacher's onset latent) for rows idx.

    cached: run only BOS and the next position over the rows' prefix keys/values
    (sites that do not touch the prefix); otherwise the full stateless pass.
    """
    use_cache = cached and (steer is None or steer.site.prefix_independent)
    b = batcher.get(idx, prefixes=not use_cache)
    if use_cache:
        z_bos, z_next = student_forward_cached(model, steer, batcher.get_kv(idx), b["a_k"])
    else:
        z_bos, z_next = student_forward(model, steer, b["prefixes"], b["kinds"], b["a_k"])
    return b, z_bos, z_next


def site_scale(model, site: Site, batcher: Batcher, n: int = 64) -> list[float]:
    """RMS of the activations each of the site's vectors is added to (sets its parameter units)."""
    fl = model.flow_lm
    if site.latent_space:
        return [float(fl.bos_emb.detach().pow(2).mean().sqrt())]
    if site.inserts:
        return [float(fl.conditioner.embed.weight.detach().pow(2).mean().sqrt())]
    probe = Steer(site, fl).to(batcher.device)
    seen = defaultdict(list)

    def record(
        x: torch.Tensor, i: int = 0
    ) -> torch.Tensor:  # record, add nothing; skip calls without kinds (prefill)
        if probe.kinds is not None:
            seen[i].append(x[probe.kinds == site.kind].detach())
        return x

    probe._add = record
    probe.attach(fl)
    try:
        with torch.no_grad():
            student_outputs(model, probe, batcher, list(range(min(n, len(batcher)))), cached=True)
    finally:
        probe.detach_hooks()
    return [float(torch.cat(seen[i]).pow(2).mean().sqrt()) for i in range(site.n_vectors)]


LOSSES = ("z_mse", "flow")
FLOW_DRAWS = 4  # noise draws per row for the flow loss
LOGNORM_MEAN, LOGNORM_STD = 0.4, 1.0  # LSD's flow-matching time distribution


def flow_loss(
    model, z: torch.Tensor, target: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Per-row flow-matching loss of the frozen flow head at z for `target` latents.

    The diagonal (s == t) term of LSD pretraining: it scores how likely the
    head makes the teacher's onset latent, instead of regressing z onto the
    (mean) teacher state.
    """
    x = target.repeat(FLOW_DRAWS, 1)
    e = torch.randn(x.shape, device=x.device, generator=generator)
    t = torch.sigmoid(
        torch.randn(x.shape[0], 1, device=x.device, generator=generator) * LOGNORM_STD
        + LOGNORM_MEAN
    )
    x_t = t * x + (1 - t) * e
    per_draw = (
        (model.flow_lm.flow_net(z.repeat(FLOW_DRAWS, 1), t, t, x_t) - (x - e)).square().sum(dim=-1)
    )
    return per_draw.view(FLOW_DRAWS, -1).mean(dim=0)


def per_row_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean(dim=-1)


@torch.no_grad()
def evaluate_steer(
    model, steer: Steer | None, batcher: Batcher, batch_size: int = 128, cached: bool = True
) -> dict:
    """Mean diagnostics over a rollout set (all rows, and rows with k > 0)."""
    rows = defaultdict(list)
    gen = torch.Generator(device=batcher.device).manual_seed(
        0
    )  # same flow-loss noise every evaluation
    for start in range(0, len(batcher), batch_size):
        idx = list(range(start, min(start + batch_size, len(batcher))))
        b, z_bos, z_next = student_outputs(model, steer, batcher, idx, cached)
        rows["z_mse"].append(per_row_mse(z_bos, b["zk"]))
        rows["latent_mse"].append(per_row_mse(sample_latent(model, z_bos, b["eps_k"]), b["a_k"]))
        rows["flow_loss"].append(flow_loss(model, z_bos, b["a_k"], gen))
        rows["next_z_mse"].append(per_row_mse(z_next, b["zk1"]))
        rows["next_latent_mse"].append(
            per_row_mse(sample_latent(model, z_next, b["eps_k1"]), b["a_k1"])
        )
        rows["bos_eos_rate"].append(
            (model.flow_lm.out_eos(z_bos)[:, 0] > model.eos_threshold).float()
        )
        rows["k"].append(b["k"].to(z_bos.device).float())
    cat = {name: torch.cat(v) for name, v in rows.items()}
    late = cat.pop("k") > 0
    out = {}
    for name, v in cat.items():
        out[name] = float(v.mean())
        out[f"{name}_k>0"] = float(v[late].mean())
    return out


def train_steer(
    model,
    site: Site,
    train: Batcher,
    val: Batcher,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    log_every: int,
    loss: str = "z_mse",
    log=print,
    snapshot_steps: list[int] = (),
    snapshots: dict | None = None,
    cached: bool = True,
) -> tuple[Steer, list[dict]]:
    """Train one steer vector; `snapshots` (if given) is filled with {step: theta} at `snapshot_steps`.

    cached: for sites that leave the prefix untouched, train over cached prefix
    keys/values (see student_forward_cached; build `train` with store_prefix_kv).
    log_every <= 0 skips validation.
    """
    assert loss in LOSSES, loss
    for p in model.parameters():
        p.requires_grad_(False)
    fl = model.flow_lm
    scale = site_scale(model, site, train)
    init = fl.conditioner.embed.weight.detach().mean(0) if site.inserts else None
    steer = Steer(site, fl, scale=scale, init=init).to(train.device)
    steer.attach(fl)
    opt = torch.optim.Adam([steer.theta], lr=lr)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))),
    )
    rng = random.Random(seed)
    history = []
    try:
        for step in range(steps + 1):
            if snapshots is not None and step in snapshot_steps:
                snapshots[step] = steer.theta.detach().cpu().clone()
            if log_every > 0 and (step % log_every == 0 or step == steps):
                metrics = {
                    "step": step,
                    **evaluate_steer(model, steer, val, cached=cached),
                    "theta_norm": float(steer.theta.norm()),
                }
                history.append(metrics)
                log(
                    f"[{site.name}/{loss}] step {step} val z_mse={metrics['z_mse']:.4f} (k>0 {metrics['z_mse_k>0']:.4f}) flow={metrics['flow_loss']:.3f} "
                    f"latent={metrics['latent_mse']:.4f} next_z={metrics['next_z_mse']:.4f} next_latent={metrics['next_latent_mse']:.4f} "
                    f"bos_eos={metrics['bos_eos_rate']:.3f} |theta|={metrics['theta_norm']:.2f}"
                )
            if step == steps:
                break
            # Sets smaller than a batch train full-batch.
            idx = rng.sample(range(len(train)), min(batch_size, len(train)))
            b, z_bos, _ = student_outputs(model, steer, train, idx, cached)
            value = (
                F.mse_loss(z_bos, b["zk"])
                if loss == "z_mse"
                else flow_loss(model, z_bos, b["a_k"]).mean()
            )
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
            sched.step()
    finally:
        steer.detach_hooks()
    return steer, history


def summarize_rollouts(data: dict) -> dict:
    k = data["k"].numpy()
    return {
        "n": int(len(k)),
        "skipped": int(data["skipped"]),
        "mean_k": float(k.mean()),
        "zero_rate": float((k == 0).mean()),
        "hist": np.bincount(k).tolist(),
    }

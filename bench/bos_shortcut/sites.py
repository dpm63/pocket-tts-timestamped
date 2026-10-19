"""Injection sites for learned vectors in the frozen FlowLM.

Every position of the FlowLM stream has a kind:

    [BOS_BEFORE_VOICE][VOICE x Tv][TEXT x L]([SOFT])[BOS][AUDIO ...]

A site is "add the vector to positions of kind K at residual-stream index l"
(l = 0 is the transformer input, l = n_layers its output before out_norm,
"z" the out_norm output that feeds the flow and EOS heads). A site may list
several residual indices (e.g. bos_all), with one vector per index, trained
jointly. Two sites are not plain additions:
    bos_latent  the vector lives in the 32-d latent space and goes through
                input_linear, i.e. it is a learned offset to bos_emb
    soft_token  the vector is an extra position inserted between TEXT and BOS

The same hooks serve training (the caller builds the sequence and sets
`steer.kinds`) and inference (`steered(model, steer)` labels the positions of
every TTSModel._run_flow_lm call).
"""

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

PAD, BOS_BEFORE_VOICE, VOICE, TEXT, SOFT, BOS, AUDIO = range(7)


@dataclass(frozen=True)
class Site:
    name: str
    kind: int
    layer: int | str | tuple[int, ...]  # residual index (or several), "z", or "insert"
    latent_space: bool = False

    @property
    def inserts(self) -> bool:
        return self.layer == "insert"

    @property
    def points(self) -> tuple:
        """Where the site adds vectors: residual indices and/or "z" (none for an insert)."""
        if self.inserts:
            return ()
        return self.layer if isinstance(self.layer, tuple) else (self.layer,)

    @property
    def n_vectors(self) -> int:
        return max(1, len(self.points))

    @property
    def prefix_independent(self) -> bool:
        """Leaves the voice/text prefix untouched, so its keys/values can be cached in training."""
        return self.kind in (BOS, AUDIO) and not self.inserts

    @property
    def cacheable(self) -> bool:
        """Only touches the voice prefix, so it lives in the cached voice state."""
        return self.kind in (BOS_BEFORE_VOICE, VOICE)


def all_sites(n_layers: int) -> dict[str, Site]:
    sites = [Site("bos_latent", BOS, 0, latent_space=True)]
    sites += [Site(f"bos_L{layer}", BOS, layer) for layer in range(n_layers + 1)]
    sites += [Site("bos_all", BOS, tuple(range(n_layers + 1)))]
    sites += [
        Site("bos_z", BOS, "z"),
        Site("text_L0", TEXT, 0),
        Site("voice_L0", VOICE, 0),
        Site("bos_before_voice", BOS_BEFORE_VOICE, 0),
        Site("soft_token", SOFT, "insert"),
    ]
    return {s.name: s for s in sites}


class Steer(nn.Module):
    """Learned vector(s) added at `site`, one per site point (or inserted, for soft_token).

    theta is [n_vectors, dim], each row in units of its point's `scale` (the RMS
    of the activations it is added to), so one learning rate fits every site.
    """

    def __init__(
        self,
        site: Site,
        flow_lm: nn.Module,
        scale: float | list[float] = 1.0,
        init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.site = site
        dim = flow_lm.ldim if site.latent_space else flow_lm.dim
        scale_t = (
            torch.as_tensor(scale, dtype=torch.float32).reshape(-1).expand(site.n_vectors).clone()
        )
        theta = (
            torch.zeros(site.n_vectors, dim)
            if init is None
            else init.reshape(site.n_vectors, dim) / scale_t[:, None]
        )
        self.theta = nn.Parameter(theta)
        self.register_buffer("scale", scale_t)
        # input_linear has no bias, so input_linear(bos + d) = input_linear(bos) + input_linear(d).
        # Kept out of the submodules so the frozen weights are not Steer parameters.
        self.__dict__["_input_linear"] = flow_lm.input_linear if site.latent_space else None
        self.kinds: torch.Tensor | None = None  # [B, S] position kinds of the current call
        self._handles = []

    def vector(self, i: int = 0) -> torch.Tensor:
        v = self.theta[i] * self.scale[i]
        if self._input_linear is not None:
            v = self._input_linear(v)
        return v

    def _add(self, x: torch.Tensor, i: int = 0) -> torch.Tensor:
        if self.kinds is None:
            return x
        assert self.kinds.shape == x.shape[:2], (self.kinds.shape, x.shape)
        mask = (self.kinds == self.site.kind).to(x.dtype)[..., None]
        return x + mask * self.vector(i).to(x.dtype)

    @property
    def attached(self) -> bool:
        return bool(self._handles) or self.site.inserts

    def attach(self, flow_lm: nn.Module) -> None:
        layers = flow_lm.transformer.layers
        for i, point in enumerate(self.site.points):
            # i is bound per hook through the default argument.
            if point == "z":
                handle = flow_lm.out_norm.register_forward_hook(
                    lambda m, a, out, i=i: self._add(out, i)
                )
            elif point == len(layers):
                handle = flow_lm.transformer.register_forward_hook(
                    lambda m, a, out, i=i: self._add(out, i)
                )
            else:
                handle = layers[point].register_forward_pre_hook(
                    lambda m, args, i=i: (self._add(args[0], i), *args[1:])
                )
            self._handles.append(handle)

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def call_kinds(
    model, steer: Steer | None, n_text: int, n_cond: int, latents: torch.Tensor
) -> torch.Tensor:
    """Position kinds of one TTSModel._run_flow_lm call.

    Calls carry either the voice prompt (audio_conditioning alone), the text
    (text tokens, plus the soft token passed as audio_conditioning when the
    site inserts one), or one latent (NaN = BOS).
    """
    if n_text > 0:
        assert n_cond <= 1 and (n_cond == 0 or (steer is not None and steer.site.inserts))
        cond = [SOFT] * n_cond
    else:
        cond = [VOICE] * n_cond
        if n_cond and model.flow_lm.insert_bos_before_voice:
            cond[0] = BOS_BEFORE_VOICE
    lat = [
        BOS if bool(torch.isnan(latents[0, i]).all()) else AUDIO for i in range(latents.shape[1])
    ]
    kinds = [TEXT] * n_text + cond + lat
    return torch.tensor(kinds, device=latents.device)[None].expand(latents.shape[0], -1)


@contextlib.contextmanager
def steered(model, steer: Steer | None):
    """Apply `steer` to every flow-LM call of a pocket_tts TTSModel."""
    if steer is None:
        yield
        return
    orig = model._run_flow_lm

    def run(model_state, text_tokens, backbone_input_latents, audio_conditioning):
        steer.kinds = call_kinds(
            model, steer, text_tokens.shape[1], audio_conditioning.shape[1], backbone_input_latents
        )
        try:
            return orig(
                model_state=model_state,
                text_tokens=text_tokens,
                backbone_input_latents=backbone_input_latents,
                audio_conditioning=audio_conditioning,
            )
        finally:
            steer.kinds = None

    steer.attach(model.flow_lm)
    model._run_flow_lm = run
    try:
        yield
    finally:
        del model._run_flow_lm  # back to the class method
        steer.detach_hooks()


def make_steer(model, site_name: str, theta: torch.Tensor, scale: float | list[float]) -> Steer:
    site = all_sites(len(model.flow_lm.transformer.layers))[site_name]
    steer = Steer(site, model.flow_lm, scale=scale)
    steer.theta.data.copy_(theta.reshape(steer.theta.shape))  # older checkpoints store a 1-d theta
    return steer.to(next(model.flow_lm.parameters()).device)


def load_steer(path, model) -> Steer:
    blob = torch.load(path, map_location="cpu")
    return make_steer(model, blob["site"], blob["theta"], blob["scale"])


def save_steer(path, steer: Steer, **extra) -> None:
    torch.save(
        {
            "site": steer.site.name,
            "theta": steer.theta.detach().cpu(),
            "scale": steer.scale.tolist(),
            **extra,
        },
        path,
    )

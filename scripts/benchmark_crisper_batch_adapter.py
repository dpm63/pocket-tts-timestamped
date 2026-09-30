# ruff: noqa: ANN001, ANN202
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BatchTranscription:
    text: str
    words: list[dict]


def _generated_ids(sequence, prompt: list[int], *, eot_id: int | None, pad_id: int | None):
    ids = [int(token) for token in sequence.tolist()]
    if ids[: len(prompt)] == prompt:
        ids = ids[len(prompt) :]
    stop_ids = {token for token in (eot_id, pad_id) if token is not None}
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[:index]
    return ids


def _attention_for_sample(cross_attentions, heads, sample_index: int, token_count: int):
    import torch

    rows = []
    for step in cross_attentions[:token_count]:
        encoder_frames = step[0].shape[-1]
        row = torch.zeros(encoder_frames, dtype=torch.float32, device=step[0].device)
        for layer, head in heads:
            row += step[layer][sample_index, head, -1].float()
        rows.append(row / len(heads))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.stack(rows).detach().cpu().numpy().astype(np.float32)


def _serialize_result(result) -> BatchTranscription:
    return BatchTranscription(
        text=str(getattr(result, "text", "")),
        words=[
            {"word": str(word.word).strip(), "start": float(word.start), "end": float(word.end)}
            for word in (getattr(result, "words", None) or [])
        ],
    )


def _transcribe_group(model, records: list[dict]) -> dict[int, BatchTranscription]:
    import torch
    from crisperwhisper.fallback import decode_with_coverage_fallback
    from crisperwhisper.hallucination import DEFAULT_REPAIR_THRESHOLDS, find_token_loop
    from crisperwhisper.prompt import strip_prompt_artifacts
    from crisperwhisper.word_timing import extract_word_timings
    from transformers import LogitsProcessorList

    engine = model._engine
    engine.enable_attention(None)
    features = torch.cat([record["features"] for record in records], dim=0)
    prompts = [record["prompt"] for record in records]
    decoder_input_ids = torch.tensor(prompts, device=engine.device, dtype=torch.long)
    suppress_tokens = list(engine._resolve_suppress(None))
    with torch.no_grad():
        generated = engine.model.generate(
            features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
            use_cache=True,
            suppress_tokens=suppress_tokens,
            logits_processor=LogitsProcessorList(),
        )

    heads = engine._resolved_alignment_heads()
    eot_id = getattr(engine, "eot_id", None)
    pad_id = getattr(engine.model.generation_config, "pad_token_id", None)
    output = {}
    for batch_index, record in enumerate(records):
        gen_ids = _generated_ids(
            generated.sequences[batch_index], record["prompt"], eot_id=eot_id, pad_id=pad_id
        )
        attention = _attention_for_sample(
            generated.cross_attentions, heads, batch_index, len(gen_ids)
        )
        if find_token_loop(gen_ids, reps=DEFAULT_REPAIR_THRESHOLDS) is not None:
            result = model.transcribe(
                str(record["path"]),
                language=record["language"],
                mode="intended",
                word_timestamps=True,
            )
            output[record["index"]] = _serialize_result(result)
            continue

        sample_features = features[batch_index : batch_index + 1]
        gen_ids, attention = decode_with_coverage_fallback(
            engine,
            sample_features,
            record["mel"],
            record["prompt"],
            gen_ids,
            attention,
            max_length=256,
            want_attention=True,
            enabled=True,
            ref_prompt_tokens=record["sibling_prompt"],
            suppress_tokens=None,
        )
        text = strip_prompt_artifacts(engine.decode_tokens(gen_ids, skip_special=True))
        words = extract_word_timings(
            engine, gen_ids, attention, record["mel"], audio_duration_s=record["duration"]
        )
        output[record["index"]] = BatchTranscription(
            text=text,
            words=[
                {"word": str(word.word).strip(), "start": float(word.start), "end": float(word.end)}
                for word in words
            ],
        )
    return output


def transcribe_short_batch(model, items: list[tuple[Path, dict]]) -> list[BatchTranscription]:
    """Batch CrisperWhisper Transformers generation for evaluation clips up to 30 seconds.

    The upstream high-level API currently exposes only single-audio transcription. This adapter
    keeps its prompt, timing, collapse-recovery, and loop-repair behavior while batching the
    expensive HuggingFace decoder pass. Long clips fall back to the upstream single-file path.
    """
    if model.backend != "transformers" or model.model_version != 2:
        raise ValueError("Batched transcription requires a CrisperWhisper v2 Transformers model")

    from crisperwhisper.audio import SAMPLE_RATE, get_duration, load_audio
    from crisperwhisper.prompt import PromptBuilder

    results: dict[int, BatchTranscription] = {}
    groups: dict[int, list[dict]] = {}
    engine = model._engine
    for index, (path, metadata) in enumerate(items):
        language = metadata["_evaluation_language"]
        audio = load_audio(str(path))
        if get_duration(audio) > 30.0:
            results[index] = _serialize_result(
                model.transcribe(
                    str(path), language=language, mode="intended", word_timestamps=True
                )
            )
            continue
        prompt_builder = PromptBuilder(engine, language=language)
        prompt = prompt_builder.intended(hotwords=None)
        features, mel = engine.extract_features_with_mel(audio)
        record = {
            "index": index,
            "path": path,
            "language": language,
            "duration": min(len(audio) / SAMPLE_RATE, 30.0),
            "features": features,
            "mel": mel,
            "prompt": prompt,
            "sibling_prompt": prompt_builder.verbatim(hotwords=None),
        }
        groups.setdefault(len(prompt), []).append(record)

    for records in groups.values():
        results.update(_transcribe_group(model, records))
    return [results[index] for index in range(len(items))]

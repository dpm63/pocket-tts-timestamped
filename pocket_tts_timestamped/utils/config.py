"""Configuration models for loading YAML config files."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from pocket_tts_timestamped.utils.utils import download_if_necessary

CONFIGS_DIR = Path(__file__).parent.parent / "config"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    dim: int
    depth: int
    # "lsd" (2 time conditions, 1-step decode) or "flow_matching" (1 time
    # condition, Euler integration; needs >= 16 decode steps).
    type: str = "lsd"


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str


# Root configuration
class FlowLMConfig(StrictModel):
    """Root configuration model for YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None
    insert_bos_before_voice: bool = False


# SEANet configuration
class SEANetConfig(StrictModel):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    dimension: int
    output_dimension: int


# Root configuration
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: str | None = None
    inner_dim: int | None = None
    outer_dim: int | None = None


class TimestampHeadConfig(StrictModel):
    layer: int
    head: int


class Config(StrictModel):
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None
    pad_with_spaces_for_short_inputs: bool = False
    remove_semicolons: bool = False
    append_terminal_punctuation: bool = True
    model_recommended_frames_after_eos: int | None = None
    default_temperature: float = 0.7
    timestamp_heads: list[TimestampHeadConfig] | None = None

    @model_validator(mode="after")
    def validate_timestamp_heads(self) -> Self:
        if self.timestamp_heads is None:
            return self
        if not self.timestamp_heads:
            raise ValueError("timestamp_heads must contain at least one layer/head pair")
        seen: set[tuple[int, int]] = set()
        for selected in self.timestamp_heads:
            pair = (selected.layer, selected.head)
            if selected.layer < 0 or selected.layer >= self.flow_lm.transformer.num_layers:
                raise ValueError(
                    f"Timestamp layer {selected.layer} is outside the FlowLM layer range"
                )
            if selected.head < 0 or selected.head >= self.flow_lm.transformer.num_heads:
                raise ValueError(f"Timestamp head {selected.head} is outside the FlowLM head range")
            if pair in seen:
                raise ValueError(f"Duplicate timestamp head L{selected.layer}H{selected.head}")
            seen.add(pair)
        return self


def load_config(yaml_path: str | Path) -> Config:
    yaml_path = download_if_necessary(str(yaml_path))

    if not yaml_path.exists():
        if yaml_path.is_relative_to(CONFIGS_DIR):
            raise FileNotFoundError(
                f"Config file not found: {yaml_path}. "
                f"Did you make a typo? Available languages: {[p.stem for p in CONFIGS_DIR.glob('*.yaml')]}"
            )
        raise FileNotFoundError(f"Config file not found: {yaml_path}. Did you make a typo?")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

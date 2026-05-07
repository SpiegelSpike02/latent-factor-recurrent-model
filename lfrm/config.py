from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LFRMConfig:
    belief_dim: int = 0
    num_slots: int = 64
    num_heads: int = 4
    cell_attention_layers: int = 1
    cell_attention_type: str = "gated_dual"
    l_cycles: int = 3
    latent_processor_layers: int = 1
    symbol_context_mode: str = "cell_symbol_tokens"
    slot_readout_mode: str = "cell_symbol_attention"
    belief_temperature: float = 1.0
    belief_step_size: float = 0.25
    belief_floor: float = 1e-5
    assignment_temperature: float = 1.0
    use_condition_type_embedding: bool = True


@dataclass(frozen=True)
class TRMConfig:
    h_cycles: int = 3
    l_cycles: int = 6
    l_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 4
    mlp_t: bool = True
    attention_type: str = "standard"
    local_mixing: bool = False
    local_mixing_kernel: int = 3
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    pos_encodings: str = "none"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_exploration_prob: float = 0.1
    no_act_continue: bool = True


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    model_type: str = "lfrm"
    num_puzzle_identifiers: int = 1
    seq_len: int = 81
    grid_height: int = 9
    grid_width: int = 9
    d_model: int = 256
    num_steps: int = 6
    dropout_rate: float = 0.0
    lfrm: LFRMConfig | None = None
    trm: TRMConfig | None = None

    @property
    def lfrm_config(self) -> LFRMConfig:
        return self.lfrm or LFRMConfig()

    @property
    def trm_config(self) -> TRMConfig:
        return self.trm or TRMConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    lr_min_ratio: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    flatten_optimizer: bool = False


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    eval_batch_size: int = 0
    max_steps: int = 500
    log_every: int = 10
    eval_every: int = 100
    eval_batches: int = 20
    trm_train_mode: str = "act"
    dense_loss_weight: float = 0.5
    final_loss_weight: float = 0.5
    sequence_loss_weight: float = 0.0
    sequence_loss_temperature: float = 0.5
    q_loss_weight: float = 0.0
    terminal_residual_weight: float = 0.0
    slot_consistency_weight: float = 0.0
    slot_usage_weight: float = 0.0
    slot_diversity_weight: float = 0.0
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    ema: EMAConfig | None = None

    @property
    def use_ema(self) -> bool:
        return (self.ema or EMAConfig()).enabled

    @property
    def ema_decay(self) -> float:
        return (self.ema or EMAConfig()).decay


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "latent-factor-recurrent-model"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    data: DataConfig
    runtime: RuntimeConfig
    wandb: WandbConfig

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.train.checkpoint_dir)

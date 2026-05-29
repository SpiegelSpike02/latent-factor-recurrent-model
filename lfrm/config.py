from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    type: str = "sudoku"


@dataclass(frozen=True)
class TRMConfig:
    h_cycles: int = 3
    l_cycles: int = 6
    l_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 4
    mlp_t: bool = False
    local_mixing: bool = False
    local_mixing_kernel: int = 3
    puzzle_embed_ndim: int = 0
    puzzle_embed_len: int = 16
    position_encoding: str = "none"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_exploration_prob: float = 0.1
    no_act_continue: bool = True
    step_loss_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class BeliefDynamicsConfig:
    commit_steps: int = 6
    h_cycles: int = 1
    l_cycles: int = 2
    l_layers: int = 1
    hidden_state_dim: int = 0
    num_heads: int = 4
    mlp_ratio: int = 2
    local_kernel: int = 3
    attn_scale: float = 0.2
    local_scale: float = 0.2
    position_encoding: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_exploration_prob: float = 0.1
    step_loss_schedule: str = "uniform"
    update_rule: str = "proposal"
    draft_view: str = "auto"
    prediction_view: str = "auto"
    update_step_size: float = 0.3


@dataclass(frozen=True)
class URMConfig:
    recurrent_steps: int = 16
    h_cycles: int = 2
    l_cycles: int = 6
    l_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    conv_kernel: int = 2
    puzzle_embed_ndim: int = 512
    puzzle_embed_len: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_exploration_prob: float = 0.1
    step_loss_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    input_vocab_size: int | None = None
    model_type: str = "bdr"
    num_puzzle_identifiers: int = 1
    seq_len: int = 81
    grid_height: int = 9
    grid_width: int = 9
    d_model: int = 256
    rollout_steps: int = 6
    dropout_rate: float = 0.0
    loss_type: str = "softmax"
    task: TaskConfig | None = None
    trm: TRMConfig | None = None
    bdr: BeliefDynamicsConfig | None = None
    urm: URMConfig | None = None

    @property
    def task_type(self) -> str:
        return (self.task or TaskConfig()).type

    @property
    def trm_config(self) -> TRMConfig:
        return self.trm or TRMConfig()

    @property
    def bdr_config(self) -> BeliefDynamicsConfig:
        return self.bdr or BeliefDynamicsConfig()

    @property
    def urm_config(self) -> URMConfig:
        return self.urm or URMConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: str = "adamw"
    learning_rate: float = 3e-4
    puzzle_embed_learning_rate: float = 0.0
    lr_min_ratio: float = 0.1
    lr_mid_ratio: float = 0.0
    lr_mid_fraction: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    puzzle_embed_weight_decay: float = 0.0
    puzzle_embed_coalesce_updates: bool = True
    lr_warmup_steps: int = 100
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    epochs: int = 500
    optimizer_updates: int = 0
    log_epochs: int = 10
    log_interval_updates: int = 0
    train_mode: str = "act"
    halt_loss_weight: float = 0.0
    terminal_residual_weight: float = 0.0
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
class EvalConfig:
    batch_size: int = 0
    nums: int = 10
    interval_updates: int = 0
    diagnostics: bool = False
    full_dataset: bool = True


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    compute_dtype: str = "bfloat16"
    data_parallel_devices: int = 1
    prefetch_depth: int = 4
    prefetch_workers: int = 2
    profile_enabled: bool = False
    profile_start_step: int = 1000
    profile_steps: int = 20
    profile_dir: str = "profile"


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
    task: TaskConfig = field(default_factory=TaskConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.train.checkpoint_dir)

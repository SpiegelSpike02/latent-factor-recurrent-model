from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TRMConfig:
    deep_recursion: int = 3
    latent_recursion: int = 6
    block_layers: int = 2
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
class BRCSudokuConfig:
    recurrent_steps: int = 6
    block_layers: int = 1
    latent_dim: int = 128
    num_heads: int = 4
    mlp_ratio: int = 2
    position_encoding: str = "learned"
    step_loss_weights: tuple[float, ...] | None = None
    latent_fit_steps: int = 4
    latent_lr: float = 0.1
    latent_grad_clip_norm: float = 1.0
    latent_update_clip_norm: float = 0.5
    denoise_initial_prob: float = 0.4
    denoise_teacher_reveal_prob: float = 0.25
    denoise_mode_weights: tuple[float, ...] = (0.35, 0.20, 0.30, 0.15)
    verifier_loss_weight: float = 0.2
    meta_loss_weight: float = 0.0
    fit_given_weight: float = 0.2
    fit_energy_weight: float = 1.0
    fit_consistency_weight: float = 0.1
    fit_prior_weight: float = 0.02
    verifier_layers: int = 4
    verifier_margin: float = 1.0




@dataclass(frozen=True)
class URMConfig:
    recurrent_steps: int = 16
    deep_recursion: int = 2
    latent_recursion: int = 6
    block_layers: int = 4
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
    task_type: str = "sudoku"
    supervision: str = "unknown_only"
    model_type: str = "brc_sudoku"
    num_puzzle_identifiers: int = 1
    seq_len: int = 81
    grid_height: int = 9
    grid_width: int = 9
    d_model: int = 256
    rollout_steps: int = 6
    dropout_rate: float = 0.0
    loss_type: str = "softmax"
    clamp_given: bool = False
    trm: TRMConfig | None = None
    brc: BRCSudokuConfig | None = None
    urm: URMConfig | None = None

    @property
    def trm_config(self) -> TRMConfig:
        return self.trm or TRMConfig()

    @property
    def brc_config(self) -> BRCSudokuConfig:
        return self.brc or BRCSudokuConfig()

    @property
    def urm_config(self) -> URMConfig:
        return self.urm or URMConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: str = "adamw"
    learning_rate: float = 3e-4
    puzzle_embed_learning_rate: float = 0.0
    lr_min_ratio: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    puzzle_embed_weight_decay: float = 0.0
    lr_warmup_steps: int = 100
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
    gradient_accumulation_steps: int = 1
    epochs: int = 500
    optimizer_updates: int = 0
    log_epochs: int = 10
    log_interval_updates: int = 0
    eval_epochs: int = 100
    eval_interval_updates: int = 0
    trm_train_mode: str = "act"
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
class DataConfig:
    dataset_path: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    compute_dtype: str = "bfloat16"
    data_parallel_devices: int = 1
    prefetch_depth: int = 4
    prefetch_workers: int = 2
    profile_enabled: bool = True
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

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.train.checkpoint_dir)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransitionConfig:
    type: str = "residual"
    hidden_dim: int = 128


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int = 4


@dataclass(frozen=True)
class RelationConfig:
    include_global: bool = True


@dataclass(frozen=True)
class ClueConfig:
    use_type_embedding: bool = True
    fix_outputs: bool = True
    freeze_state: bool = False


@dataclass(frozen=True)
class ComputeConfig:
    # Inner compute repeats happen inside each outer reasoning step. Early inner
    # steps can be stop-gradient while still contributing forward refinement.
    inner_steps: int = 1
    layers_per_step: int = 1
    grad_inner_steps: int = 1
    reinject_input: bool = False


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    model_type: str = "universal_transformer"
    communication_type: str = "relation"
    seq_len: int = 81
    grid_height: int = 9
    grid_width: int = 9
    d_model: int = 256
    d_ff: int = 1024
    num_steps: int = 6
    dropout_rate: float = 0.0
    transition: TransitionConfig | None = None
    attention: AttentionConfig | None = None
    relation: RelationConfig | None = None
    clues: ClueConfig | None = None
    compute: ComputeConfig | None = None

    @property
    def transition_hidden_dim(self) -> int:
        return (self.transition or TransitionConfig()).hidden_dim

    @property
    def transition_type(self) -> str:
        return (self.transition or TransitionConfig()).type

    @property
    def uses_damped_transition(self) -> bool:
        return self.transition_type == "damped"

    @property
    def num_heads(self) -> int:
        return (self.attention or AttentionConfig()).num_heads

    @property
    def include_global_relation(self) -> bool:
        return (self.relation or RelationConfig()).include_global

    @property
    def use_clue_type_embedding(self) -> bool:
        return (self.clues or ClueConfig()).use_type_embedding

    @property
    def fix_clue_outputs(self) -> bool:
        return (self.clues or ClueConfig()).fix_outputs

    @property
    def freeze_clue_state(self) -> bool:
        return (self.clues or ClueConfig()).freeze_state

    @property
    def inner_steps(self) -> int:
        return (self.compute or ComputeConfig()).inner_steps

    @property
    def layers_per_step(self) -> int:
        return (self.compute or ComputeConfig()).layers_per_step

    @property
    def grad_inner_steps(self) -> int:
        return (self.compute or ComputeConfig()).grad_inner_steps

    @property
    def reinject_input(self) -> bool:
        return (self.compute or ComputeConfig()).reinject_input


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 500
    log_every: int = 10
    eval_every: int = 100
    eval_batches: int = 20
    validity_loss_weight: float = 0.01
    step_loss_weighting: str = "uniform"
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
    project: str = "recurrent-grid-reasoning"
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

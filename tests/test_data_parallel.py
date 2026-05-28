from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_bdr_step_carry_runs_with_two_data_parallel_devices() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from lfrm.config import (
    BDRConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.runtime.sharding import (
    batch_sharding,
    data_parallel_mesh,
    device_put_batch_sharded,
    place_module_replicated,
    place_tree,
    replicated_sharding,
)
from lfrm.training.bdr import build_bdr_step_carry_train_step_runner
from lfrm.training.factory import create_model, create_optimizer

config = ExperimentConfig(
    task=TaskConfig(type="sudoku"),
    model=ModelConfig(
        vocab_size=9,
        input_vocab_size=10,
        model_type="bdr",
        seq_len=81,
        grid_height=9,
        grid_width=9,
        d_model=64,
        bdr=BDRConfig(
            commit_steps=2,
            refine_steps=1,
            block_depth=1,
            hidden_state_dim=64,
            num_heads=4,
            mlp_ratio=1,
            update_rule="velocity",
            fixed_point_update_weight=1e-3,
            wrong_attractor_direction_weight=0.0,
            wrong_attractor_nonzero_weight=0.0,
            corrupted_recovery_weight=0.0,
            early_stop_require_constraints=False,
        ),
    ),
    optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
    train=TrainConfig(batch_size=4, train_mode="step_carry"),
    eval=EvalConfig(batch_size=4),
    data=DataConfig(dataset_path="unused"),
    runtime=RuntimeConfig(data_parallel_devices=2),
    wandb=WandbConfig(enabled=False),
)

mesh = data_parallel_mesh(config)
assert mesh is not None
data_sharding = batch_sharding(mesh)
state_sharding = replicated_sharding(mesh)
model = create_model(config)
optimizer = create_optimizer(model, config)
place_module_replicated(model, state_sharding)
place_module_replicated(optimizer, state_sharding)

rng = np.random.default_rng(0)
batch = device_put_batch_sharded(
    {
        "inputs": rng.integers(0, 10, size=(4, 81), dtype=np.int32),
        "labels": rng.integers(0, 9, size=(4, 81), dtype=np.int32),
        "puzzle_identifiers": np.zeros((4,), dtype=np.int32),
    },
    data_sharding,
)
carry = place_tree(model.initial_carry(batch), data_sharding)
train_step = build_bdr_step_carry_train_step_runner()

with jax.sharding.set_mesh(mesh):
    metrics, new_carry = train_step(
        model,
        optimizer,
        carry,
        batch,
        jax.random.key(0),
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready(metrics["loss"])

assert batch["inputs"].sharding.spec == jax.sharding.PartitionSpec("data")
assert new_carry["z"].sharding.spec == jax.sharding.PartitionSpec("data", None, None)
assert new_carry["current_example_mask"].sharding.spec == jax.sharding.PartitionSpec("data")
assert metrics["loss"].sharding.spec == jax.sharding.PartitionSpec()
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_trm_act_runs_with_two_data_parallel_devices() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from lfrm.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
    TrainConfig,
    TRMConfig,
    WandbConfig,
)
from lfrm.runtime.sharding import (
    batch_sharding,
    data_parallel_mesh,
    device_put_batch_sharded,
    place_module_replicated,
    place_tree,
    replicated_sharding,
)
from lfrm.training.factory import create_model, create_optimizer
from lfrm.training.recurrent import build_trm_act_train_step_runner

config = ExperimentConfig(
    task=TaskConfig(type="sudoku"),
    model=ModelConfig(
        vocab_size=9,
        input_vocab_size=10,
        model_type="trm",
        seq_len=81,
        grid_height=9,
        grid_width=9,
        d_model=64,
        trm=TRMConfig(
            h_cycles=1,
            l_cycles=1,
            l_layers=1,
            num_heads=4,
            mlp_ratio=1,
            puzzle_embed_len=0,
            position_encoding="none",
        ),
    ),
    optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
    train=TrainConfig(batch_size=4, train_mode="act", halt_loss_weight=0.0),
    eval=EvalConfig(batch_size=4),
    data=DataConfig(dataset_path="unused"),
    runtime=RuntimeConfig(data_parallel_devices=2),
    wandb=WandbConfig(enabled=False),
)

mesh = data_parallel_mesh(config)
assert mesh is not None
data_sharding = batch_sharding(mesh)
state_sharding = replicated_sharding(mesh)
model = create_model(config)
optimizer = create_optimizer(model, config)
place_module_replicated(model, state_sharding)
place_module_replicated(optimizer, state_sharding)

rng = np.random.default_rng(1)
batch = device_put_batch_sharded(
    {
        "inputs": rng.integers(0, 10, size=(4, 81), dtype=np.int32),
        "labels": rng.integers(0, 9, size=(4, 81), dtype=np.int32),
        "puzzle_identifiers": np.zeros((4,), dtype=np.int32),
    },
    data_sharding,
)
carry = place_tree(model.initial_carry(batch), data_sharding)
train_step = build_trm_act_train_step_runner(config, halt_loss_weight=0.0)

with jax.sharding.set_mesh(mesh):
    metrics, new_carry = train_step(
        model,
        optimizer,
        carry,
        batch,
        jax.random.key(0),
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready(metrics["loss"])

assert batch["inputs"].sharding.spec == jax.sharding.PartitionSpec("data")
assert new_carry["halted"].sharding.spec == jax.sharding.PartitionSpec("data")
assert metrics["loss"].sharding.spec == jax.sharding.PartitionSpec()
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_urm_act_runs_with_two_data_parallel_devices() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from lfrm.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
    TrainConfig,
    URMConfig,
    WandbConfig,
)
from lfrm.runtime.sharding import (
    batch_sharding,
    data_parallel_mesh,
    device_put_batch_sharded,
    place_module_replicated,
    place_tree,
    replicated_sharding,
)
from lfrm.training.factory import create_model, create_optimizer
from lfrm.training.recurrent import build_trm_act_train_step_runner

config = ExperimentConfig(
    task=TaskConfig(type="sudoku"),
    model=ModelConfig(
        vocab_size=9,
        input_vocab_size=10,
        model_type="urm",
        seq_len=81,
        grid_height=9,
        grid_width=9,
        d_model=64,
        urm=URMConfig(
            recurrent_steps=2,
            h_cycles=1,
            l_cycles=1,
            l_layers=1,
            num_heads=4,
            mlp_ratio=1,
            puzzle_embed_len=0,
        ),
    ),
    optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
    train=TrainConfig(batch_size=4, train_mode="act", halt_loss_weight=0.0),
    eval=EvalConfig(batch_size=4),
    data=DataConfig(dataset_path="unused"),
    runtime=RuntimeConfig(data_parallel_devices=2),
    wandb=WandbConfig(enabled=False),
)

mesh = data_parallel_mesh(config)
assert mesh is not None
data_sharding = batch_sharding(mesh)
state_sharding = replicated_sharding(mesh)
model = create_model(config)
optimizer = create_optimizer(model, config)
place_module_replicated(model, state_sharding)
place_module_replicated(optimizer, state_sharding)

rng = np.random.default_rng(2)
batch = device_put_batch_sharded(
    {
        "inputs": rng.integers(0, 10, size=(4, 81), dtype=np.int32),
        "labels": rng.integers(0, 9, size=(4, 81), dtype=np.int32),
        "puzzle_identifiers": np.zeros((4,), dtype=np.int32),
    },
    data_sharding,
)
carry = place_tree(model.initial_carry(batch), data_sharding)
train_step = build_trm_act_train_step_runner(config, halt_loss_weight=0.0)

with jax.sharding.set_mesh(mesh):
    metrics, new_carry = train_step(
        model,
        optimizer,
        carry,
        batch,
        jax.random.key(0),
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready(metrics["loss"])

assert batch["inputs"].sharding.spec == jax.sharding.PartitionSpec("data")
assert new_carry["halted"].sharding.spec == jax.sharding.PartitionSpec("data")
assert metrics["loss"].sharding.spec == jax.sharding.PartitionSpec()
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

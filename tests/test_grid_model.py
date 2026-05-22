from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.cli import load_toml_config
from lfrm.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    BRCConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
    TRMConfig,
    TrainConfig,
    URMConfig,
    WandbConfig,
)
from lfrm.runtime import (
    apply_epoch_budget,
    resolve_resume_checkpoint,
    schedule_learning_rate,
    small_metric_items,
    updates_from_epochs,
)
from lfrm.models import BRCModel, TinyRecursiveModel
from lfrm.training import (
    build_train_step_runner,
    build_trm_act_train_step_runner,
    create_model,
    create_optimizer,
    load_checkpoint,
    loss_and_metrics,
    save_checkpoint,
    stablemax_cross_entropy_with_integer_labels,
    trm_dense_unroll_loss_and_metrics,
)
from lfrm.training.optim import scheduled_lr


class GridModelTests(unittest.TestCase):
    def test_updates_from_epochs_matches_official_floor_conversion(self) -> None:
        dataset = SimpleNamespace(
            spec=SimpleNamespace(total_groups=1000, mean_puzzle_examples=1.0)
        )
        self.assertEqual(updates_from_epochs(dataset, batch_size=768, epochs=5000), 6510)

    def test_epoch_budget_uses_batch_as_global_batch(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=11),
            optimizer=OptimizerConfig(),
            train=TrainConfig(batch_size=1024, epochs=10, log_epochs=1),
            eval=EvalConfig(epochs=2),
            data=DataConfig(dataset_path="unused"),
            runtime=RuntimeConfig(),
            wandb=WandbConfig(),
        )
        dataset = SimpleNamespace(
            spec=SimpleNamespace(total_groups=2048, mean_puzzle_examples=1.0)
        )
        budgeted = apply_epoch_budget(config, dataset)
        self.assertEqual(budgeted.train.optimizer_updates, 20)
        self.assertEqual(budgeted.train.log_interval_updates, 2)
        self.assertEqual(budgeted.eval.interval_updates, 4)

    def test_small_metric_items_drops_large_leaves(self) -> None:
        metrics = {
            "loss": jnp.asarray(1.0),
            "per_step_loss": jnp.ones((16,), dtype=jnp.float32),
            "logits": jnp.ones((2, 81, 11), dtype=jnp.float32),
        }
        filtered = small_metric_items(metrics, max_elements=32)
        self.assertEqual(set(filtered), {"loss", "per_step_loss"})

    def test_lr_schedule_matches_official_zero_based_warmup(self) -> None:
        schedule = scheduled_lr(
            peak_value=1e-4,
            min_ratio=1.0,
            warmup_steps=2000,
            optimizer_updates=10000,
        )
        self.assertEqual(float(schedule(jnp.asarray(0, dtype=jnp.int32))), 0.0)
        self.assertAlmostEqual(float(schedule(jnp.asarray(1, dtype=jnp.int32))), 5e-8, places=12)
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=11),
            optimizer=OptimizerConfig(learning_rate=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000),
            train=TrainConfig(optimizer_updates=10000),
            data=DataConfig(dataset_path="unused"),
            runtime=RuntimeConfig(),
            wandb=WandbConfig(),
        )
        self.assertEqual(schedule_learning_rate(config, 1), 0.0)

    def test_solved_rate_metric(self) -> None:
        class FixedModel:
            def forward_all_steps_with_diagnostics(self, inputs, train: bool, dropout_key=None):
                del inputs, train, dropout_key
                logits = jnp.full((2, 4, 11), -10.0)
                logits = logits.at[0, 0, 2].set(10.0)
                logits = logits.at[0, 1, 3].set(10.0)
                logits = logits.at[1, 0, 2].set(10.0)
                logits = logits.at[1, 1, 1].set(10.0)
                diagnostics = {"hidden_delta_mean": jnp.asarray([0.0], dtype=jnp.float32)}
                return logits[None, :, :, :], diagnostics

        batch = {
            "inputs": jnp.zeros((2, 4), dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 0, 0], [2, 3, 0, 0]], dtype=jnp.int32),
        }
        _, metrics = loss_and_metrics(FixedModel(), batch, False, None)
        self.assertAlmostEqual(float(metrics["accuracy"]), 0.75, places=6)
        self.assertAlmostEqual(float(metrics["exact_accuracy"]), 0.5, places=6)

    def test_create_model_supports_trm_and_brc(self) -> None:
        trm_config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=1,
                trm=TRMConfig(num_heads=3),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(trm_config), TinyRecursiveModel)
        brc_config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="brc",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=1,
                brc=BRCConfig(belief_steps=1, num_heads=4),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(brc_config), BRCModel)
        invalid = ExperimentConfig(
            model=ModelConfig(vocab_size=11, model_type="legacy_shared_block"),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        with self.assertRaisesRegex(ValueError, "brc"):
            create_model(invalid)

    def test_stablemax_forward_backward_are_finite(self) -> None:
        logits = jnp.asarray([[100.0, -100.0, 0.0], [-2.0, 3.0, 0.5]], dtype=jnp.float32)
        targets = jnp.asarray([0, 1], dtype=jnp.int32)

        def objective(x):
            return jnp.mean(stablemax_cross_entropy_with_integer_labels(x, targets))

        loss, grad = jax.value_and_grad(objective)(logits)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_trm_forward_and_losses_are_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1, num_heads=3, mlp_ratio=2),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(21),
        )
        batch = {
            "inputs": jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 3, 4, 5, 4, 6, 5, 7]], dtype=jnp.int32),
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            batch["inputs"],
            puzzle_identifiers=batch["puzzle_identifiers"],
            train=False,
        )
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(diagnostics["halt_logits"].shape, (2, 1))
        self.assertEqual(diagnostics["hidden_delta_mean"].shape, (2,))
        _, metrics = loss_and_metrics(model, batch, True, jax.random.key(2), halt_loss_weight=0.1)
        for key in ("loss", "halt_loss", "selected_lm_loss", "accuracy"):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))
        dense_loss, dense_metrics = trm_dense_unroll_loss_and_metrics(
            model,
            batch,
            True,
            jax.random.key(3),
        )
        self.assertIn("lm_loss", dense_metrics)
        self.assertIn("step_loss_weights", dense_metrics)
        self.assertEqual(dense_metrics["step_loss_weights"].shape, (2,))
        self.assertTrue(bool(jnp.isfinite(dense_metrics["lm_loss"])))
        self.assertAlmostEqual(
            float(dense_loss),
            float(dense_metrics["lm_loss"]),
            places=5,
        )

    def test_trm_mlp_t_keeps_channel_mlp(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=1,
                trm=TRMConfig(
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    mlp_t=True,
                    position_encoding="none",
                    puzzle_embed_len=0,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(210),
        )
        self.assertTrue(hasattr(model.blocks[0], "token_mlp"))
        self.assertTrue(hasattr(model.blocks[0], "channel_mlp"))
        logits, _ = model.forward_all_steps_with_diagnostics(
            jnp.ones((1, 9), dtype=jnp.int32),
            train=False,
            collect_diagnostics=False,
        )
        self.assertEqual(logits.shape, (1, 1, 9, 11))

    def test_brc_forward_is_finite_without_optional_energy(self) -> None:
        model = BRCModel(
            ModelConfig(
                vocab_size=11,
                model_type="brc",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=2,
                brc=BRCConfig(
                    belief_steps=2,
                    num_heads=4,
                    mlp_ratio=1,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(31),
        )
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        tokens = jnp.asarray([[int(ch) + 1 for ch in puzzle]], dtype=jnp.int32)
        labels = jnp.asarray([[int(ch) + 1 for ch in solution]], dtype=jnp.int32)
        self.assertTrue(bool(jnp.all(model.context_mask(tokens) == (tokens > 1))))
        logits, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        final_only_logits, final_only_diagnostics = model.run_diffusion(
            tokens,
            train=False,
            return_final_only=True,
        )
        self.assertEqual(logits.shape, (2, 1, 81, 11))
        self.assertEqual(final_only_logits.shape, (1, 81, 11))
        self.assertTrue(bool(jnp.allclose(final_only_logits, logits[-1], rtol=1e-5, atol=1e-5)))
        belief = jnp.arange(2 * 81 * 11, dtype=jnp.float32).reshape(2, 81, 11) / 100.0
        self.assertTrue(
            bool(
                jnp.allclose(
                    model._belief_to_token_logits(belief, tokens, jnp.asarray(0, dtype=jnp.int32)),
                    model._belief_to_token_logits(belief, tokens, jnp.asarray(1, dtype=jnp.int32)),
                )
            )
        )
        self.assertEqual(diagnostics["diffusion_filled_ratio"].shape, (2,))
        self.assertEqual(final_only_diagnostics["diffusion_filled_ratio"].shape, (2,))

    def test_brc_losses_are_finite(self) -> None:
        model = BRCModel(
            ModelConfig(
                vocab_size=11,
                model_type="brc",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=2,
                brc=BRCConfig(
                    belief_steps=2,
                    num_heads=4,
                    mlp_ratio=1,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(32),
        )
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        tokens = jnp.asarray([[int(ch) + 1 for ch in puzzle]], dtype=jnp.int32)
        labels = jnp.asarray([[int(ch) + 1 for ch in solution]], dtype=jnp.int32)
        batch = {
            "inputs": tokens,
            "labels": labels,
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        _, metrics = loss_and_metrics(
            model,
            batch,
            True,
            jax.random.key(33),
        )
        legacy_keys = {
            "blank_ce_loss",
            "step_weighted_ce_loss",
            "final_blank_ce_loss",
            "mean_blank_ce_loss",
            "blank_cell_accuracy",
            "solved_rate",
            "solved_count",
            "invalid_board_rate",
            "conflict_count",
        }
        self.assertFalse(legacy_keys & set(metrics))
        for key in (
            "loss",
            "lm_loss",
            "final_lm_loss",
            "mean_lm_loss",
            "context_accuracy",
            "query_accuracy",
            "context_target_probability",
            "query_target_probability",
            "context_consistency",
            "invalid_rate",
            "conflicts",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))
        self.assertEqual(metrics["per_step_loss"].shape, (2,))
        self.assertEqual(metrics["step_loss_weights"].shape, (2,))
    def test_brc_arc_canvas_belief_loss_is_finite(self) -> None:
        model = BRCModel(
            ModelConfig(
                vocab_size=12,
                model_type="brc",
                task=TaskConfig(type="arc"),
                seq_len=16,
                grid_height=4,
                grid_width=4,
                d_model=16,
                rollout_steps=2,
                loss_type="stablemax",
                brc=BRCConfig(
                    belief_steps=2,
                    num_heads=4,
                    mlp_ratio=1,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(36),
        )
        inputs = jnp.asarray([[2, 3, 1, 0, 4, 5, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
        labels = jnp.asarray([[6, 7, 1, 0, 8, 9, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(inputs, train=False)
        self.assertEqual(logits.shape, (2, 1, 16, 12))
        self.assertEqual(diagnostics["diffusion_filled_ratio"].shape, (2,))
        batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        _, metrics = loss_and_metrics(model, batch, True, jax.random.key(37))
        for key in ("loss", "lm_loss", "accuracy", "exact_accuracy"):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))

    def test_brc_train_step_updates_parameters(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="brc",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=1,
                brc=BRCConfig(
                    belief_steps=1,
                    num_heads=4,
                    mlp_ratio=1,
                ),
            ),
            optimizer=OptimizerConfig(learning_rate=1e-4, lr_warmup_steps=1),
            train=TrainConfig(batch_size=1, epochs=1, optimizer_updates=1),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        self.assertNotIn("init_hidden", str(nnx.state(model, nnx.Param)))
        optimizer = create_optimizer(model, config)
        train_step = build_train_step_runner()
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        tokens = jnp.asarray([[int(ch) + 1 for ch in puzzle]], dtype=jnp.int32)
        labels = jnp.asarray([[int(ch) + 1 for ch in solution]], dtype=jnp.int32)
        batch = {
            "inputs": tokens,
            "labels": labels,
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        before = model.input_to_hidden.kernel[...]
        metrics = train_step(model, optimizer, batch, jax.random.key(34))
        metrics = train_step(model, optimizer, batch, jax.random.key(35))
        after = model.input_to_hidden.kernel[...]
        self.assertTrue(bool(jnp.isfinite(metrics["loss"])))
        self.assertGreater(float(jnp.sum(jnp.abs(after - before))), 0.0)

    def test_trm_breaks_blank_symbol_symmetry(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1, num_heads=3, mlp_ratio=2),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(21),
        )
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=jnp.asarray([0], dtype=jnp.int32),
            train=False,
        )
        blank_digit_logits = logits[-1, 0, 1, 2:]
        self.assertGreater(float(jnp.std(blank_digit_logits)), 1e-6)
        self.assertIn("halt_logits", diagnostics)

    def test_trm_rope_position_forward_is_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_embed_len=0,
                    position_encoding="rope",
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(23),
        )
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=jnp.asarray([0], dtype=jnp.int32),
            train=False,
        )
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(model.rope_cos.shape, (9, 4))
        self.assertEqual(model.rope_sin.shape, (9, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertIn("halt_logits", diagnostics)

    def test_trm_local_mixing_forward_is_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_embed_len=0,
                    position_encoding="rope",
                    local_mixing=True,
                    local_mixing_kernel=3,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(24),
        )
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=jnp.asarray([0], dtype=jnp.int32),
            train=False,
        )
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertIn("halt_logits", diagnostics)

    def test_trm_local_mixing_requires_odd_kernel(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_mixing_kernel"):
            TinyRecursiveModel(
                ModelConfig(
                    vocab_size=11,
                    model_type="trm",
                    seq_len=9,
                    grid_height=3,
                    grid_width=3,
                    d_model=12,
                    rollout_steps=1,
                    trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1,
                        num_heads=3,
                        mlp_ratio=2,
                        local_mixing=True,
                        local_mixing_kernel=2,
                    ),
                ),
                RuntimeConfig(compute_dtype="float32"),
                rngs=nnx.Rngs(25),
            )

    def test_trm_puzzle_embedding_uses_unified_optimizer_update(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="trm",
                num_puzzle_identifiers=1,
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1, num_heads=3, mlp_ratio=2, puzzle_embed_len=2),
            ),
            optimizer=OptimizerConfig(
                optimizer_type="adam_atan2",
                learning_rate=1e-4,
                puzzle_embed_learning_rate=1e-4,
                lr_min_ratio=1.0,
                weight_decay=1.0,
                puzzle_embed_weight_decay=1.0,
                lr_warmup_steps=1,
            ),
            train=TrainConfig(batch_size=2, epochs=2, optimizer_updates=2, halt_loss_weight=0.5),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_step = build_trm_act_train_step_runner(config, config.train.halt_loss_weight)
        batch = {
            "inputs": jnp.asarray(
                [[2, 1, 3, 1, 1, 4, 1, 5, 1], [3, 1, 2, 1, 1, 5, 1, 4, 1]],
                dtype=jnp.int32,
            ),
            "labels": jnp.asarray(
                [[2, 3, 3, 4, 5, 4, 6, 5, 7], [3, 4, 2, 5, 6, 5, 7, 4, 8]],
                dtype=jnp.int32,
            ),
            "puzzle_identifiers": jnp.asarray([0, 0], dtype=jnp.int32),
        }
        before = model.puzzle_embed.weights[...]
        carry = model.initial_carry(batch)
        metrics, carry = train_step(
            model,
            optimizer,
            carry,
            batch,
            jax.random.key(0),
            jnp.asarray(0, dtype=jnp.int32),
        )
        metrics, _carry = train_step(
            model,
            optimizer,
            carry,
            batch,
            jax.random.key(1),
            jnp.asarray(1, dtype=jnp.int32),
        )
        after = model.puzzle_embed.weights[...]

        self.assertTrue(bool(jnp.isfinite(metrics["loss"])))
        self.assertGreater(float(jnp.sum(jnp.abs(after - before))), 0.0)

    def test_trm_puzzle_embedding_shape(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                num_puzzle_identifiers=5,
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=2,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_embed_ndim=12,
                    puzzle_embed_len=2,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(22),
        )
        params = nnx.state(model, nnx.Param)
        self.assertEqual(model.prefix_len, 2)
        self.assertEqual(model.puzzle_embed.weights[...].shape, (5, 12))
        self.assertNotIn("state_init", str(params))

    def test_urm_muon_train_step_finite(self) -> None:
        config = ExperimentConfig(
            task=TaskConfig(type="arc"),
            model=ModelConfig(
                vocab_size=12,
                model_type="urm",
                num_puzzle_identifiers=1,
                seq_len=16,
                grid_height=4,
                grid_width=4,
                d_model=8,
                urm=URMConfig(
                    recurrent_steps=2,
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=2,
                    mlp_ratio=2,
                    conv_kernel=2,
                    puzzle_embed_ndim=8,
                    puzzle_embed_len=1,
                ),
            ),
            optimizer=OptimizerConfig(
                optimizer_type="muon",
                learning_rate=1e-4,
                puzzle_embed_learning_rate=1e-2,
                lr_min_ratio=1.0,
                beta2=0.95,
                lr_warmup_steps=1,
            ),
            train=TrainConfig(batch_size=2, epochs=2, optimizer_updates=2, halt_loss_weight=0.5),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_step = build_trm_act_train_step_runner(config, config.train.halt_loss_weight)
        batch = {
            "inputs": jnp.full((2, 16), 2, dtype=jnp.int32),
            "labels": jnp.full((2, 16), 3, dtype=jnp.int32),
            "puzzle_identifiers": jnp.asarray([0, 0], dtype=jnp.int32),
        }

        metrics, _carry = train_step(
            model,
            optimizer,
            model.initial_carry(batch),
            batch,
            jax.random.key(0),
            jnp.asarray(0, dtype=jnp.int32),
        )

        self.assertTrue(bool(jnp.isfinite(metrics["loss"])))

    def test_halt_selected_metrics_use_first_positive_halt_step(self) -> None:
        class HaltSelectedModel:
            def forward_all_steps_with_diagnostics(self, inputs, train: bool, dropout_key=None):
                del inputs, train, dropout_key
                logits = jnp.full((2, 1, 4, 11), -10.0)
                logits = logits.at[0, 0, 0, 2].set(10.0)
                logits = logits.at[0, 0, 1, 1].set(10.0)
                logits = logits.at[1, 0, 0, 2].set(10.0)
                logits = logits.at[1, 0, 1, 3].set(10.0)
                diagnostics = {
                    "hidden_delta_mean": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
                    "halt_logits": jnp.asarray([[1.0], [4.0]], dtype=jnp.float32),
                }
                return logits, diagnostics

        batch = {
            "inputs": jnp.zeros((1, 4), dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 0, 0]], dtype=jnp.int32),
        }
        _, metrics = loss_and_metrics(HaltSelectedModel(), batch, False, None, halt_loss_weight=0.1)
        self.assertAlmostEqual(float(metrics["accuracy"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["selected_accuracy"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["selected_step"]), 1.0, places=6)

    def test_num_heads_must_divide_d_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            BRCModel(
                ModelConfig(
                    vocab_size=11,
                    model_type="brc",
                    seq_len=81,
                    grid_height=9,
                    grid_width=9,
                    d_model=10,
                    rollout_steps=1,
                    brc=BRCConfig(belief_steps=1, num_heads=4),
                ),
                RuntimeConfig(compute_dtype="float32"),
                rngs=nnx.Rngs(0),
            )

    def test_config_loader_rejects_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "legacy.toml"
            config_path.write_text(
                "[model]\n"
                "model_type = \"brc\"\n"
                "legacy_field = \"old\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported \\[model\\] field"):
                load_toml_config(str(config_path))

            train_legacy_path = Path(tmpdir) / "legacy_train.toml"
            train_legacy_path.write_text(
                "[train]\n"
                "microbatch_size = 8\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported \\[train\\] field"):
                load_toml_config(str(train_legacy_path))

    def test_config_loader_accepts_brc_and_trm_step_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "brc.toml"
            config_path.write_text(
                "[task]\n"
                "type = \"sudoku\"\n"
                "\n"
                "[model]\n"
                "model_type = \"brc\"\n"
                "d_model = 16\n"
                "\n"
                "[model.brc]\n"
                "belief_steps = 2\n"
                "h_cycles = 2\n"
                "l_cycles = 2\n"
                "hidden_state_dim = 16\n"
                "l_layers = 1\n"
                "num_heads = 4\n"
                "halt_exploration_prob = 0.2\n"
                "step_loss_schedule = \"linear\"\n",
                encoding="utf-8",
            )
            loaded = load_toml_config(str(config_path))
            self.assertEqual(loaded["model_type"], "brc")
            self.assertEqual(loaded["task_type"], "sudoku")
            self.assertEqual(loaded["brc_belief_steps"], 2)
            self.assertEqual(loaded["brc_h_cycles"], 2)
            self.assertEqual(loaded["brc_l_cycles"], 2)
            self.assertEqual(loaded["brc_hidden_state_dim"], 16)
            self.assertEqual(loaded["brc_l_layers"], 1)
            self.assertEqual(loaded["brc_num_heads"], 4)
            self.assertEqual(loaded["brc_halt_exploration_prob"], 0.2)
            self.assertEqual(loaded["brc_step_loss_schedule"], "linear")

            eval_config_path = Path(tmpdir) / "eval.toml"
            eval_config_path.write_text(
                "[eval]\n"
                "batch_size = 32\n"
                "epochs = 10\n"
                "diagnostics = true\n"
                "full_dataset = true\n",
                encoding="utf-8",
            )
            eval_loaded = load_toml_config(str(eval_config_path))
            self.assertEqual(eval_loaded["eval_batch_size"], 32)
            self.assertEqual(eval_loaded["eval_epochs"], 10)
            self.assertTrue(eval_loaded["eval_diagnostics"])
            self.assertTrue(eval_loaded["eval_full_dataset"])

            trm_config_path = Path(tmpdir) / "trm.toml"
            trm_config_path.write_text(
                "[task]\n"
                "type = \"maze\"\n"
                "\n"
                "[model]\n"
                "model_type = \"trm\"\n"
                "d_model = 16\n"
                "rollout_steps = 3\n"
                "\n"
                "[model.trm]\n"
                "num_heads = 4\n"
                "step_loss_weights = [1.0, 2.0, 3.0]\n",
                encoding="utf-8",
            )
            trm_loaded = load_toml_config(str(trm_config_path))
            self.assertEqual(trm_loaded["task_type"], "maze")
            self.assertEqual(trm_loaded["rollout_steps"], 3)
            self.assertEqual(trm_loaded["trm_step_loss_weights"], [1.0, 2.0, 3.0])

    def test_checkpoint_round_trip_restores_step(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                rollout_steps=1,
                trm=TRMConfig(
                h_cycles=1,
                l_cycles=1,
                l_layers=1, num_heads=3, mlp_ratio=2),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_model(config)
            optimizer = create_optimizer(model, config)
            save_checkpoint(tmpdir, model, optimizer, 7)

            restored_model = create_model(config)
            restored_optimizer = create_optimizer(restored_model, config)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                step = load_checkpoint(Path(tmpdir) / "step_7", restored_model, restored_optimizer)
            self.assertEqual(step, 7)
            self.assertFalse(
                any("Sharding info not provided" in str(item.message) for item in caught)
            )

            checkpoint_path, run_dir = resolve_resume_checkpoint(tmpdir)
            self.assertEqual(checkpoint_path.name, "step_7")
            self.assertEqual(run_dir, Path(tmpdir).resolve())


if __name__ == "__main__":
    unittest.main()

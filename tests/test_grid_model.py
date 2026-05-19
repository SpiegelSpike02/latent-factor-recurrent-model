from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.cli import load_toml_config, resolve_resume_checkpoint
from lfrm.config import (
    DataConfig,
    ExperimentConfig,
    BRCSudokuConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TRMConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.models import BRCSudokuModel, TinyRecursiveModel
from lfrm.jax_defaults import apply_jax_defaults
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
from lfrm.training.steps import _clamp_logits_to_given


class GridModelTests(unittest.TestCase):
    def test_jax_defaults_set_gpu_startup_flags_without_overriding_user_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80",
                "XLA_FLAGS": "--some_existing_flag=true",
            },
            clear=True,
        ):
            apply_jax_defaults()
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "true")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.80")
            self.assertNotIn("TF_GPU_ALLOCATOR", os.environ)
            self.assertIn("--some_existing_flag=true", os.environ["XLA_FLAGS"].split())
            self.assertIn("--xla_gpu_triton_gemm_any=true", os.environ["XLA_FLAGS"].split())
            self.assertIn("--xla_gpu_enable_latency_hiding_scheduler=true", os.environ["XLA_FLAGS"].split())

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
            "given_mask": jnp.asarray([[False, False, True, True], [False, False, True, True]], dtype=bool),
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
                model_type="brc_sudoku",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=1,
                brc=BRCSudokuConfig(recurrent_steps=1, latent_dim=16, num_heads=4, verifier_layers=1),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(brc_config), BRCSudokuModel)
        invalid = ExperimentConfig(
            model=ModelConfig(vocab_size=11, model_type="legacy_shared_block"),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        with self.assertRaisesRegex(ValueError, "brc_sudoku"):
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1, num_heads=3, mlp_ratio=2),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(21),
        )
        batch = {
            "inputs": jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 3, 4, 5, 4, 6, 5, 7]], dtype=jnp.int32),
            "given_mask": jnp.asarray([[True, False, True, False, False, True, False, True, False]], dtype=bool),
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

    def test_brc_forward_and_verifier_are_finite(self) -> None:
        model = BRCSudokuModel(
            ModelConfig(
                vocab_size=11,
                model_type="brc_sudoku",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=2,
                brc=BRCSudokuConfig(
                    recurrent_steps=2,
                    latent_dim=16,
                    num_heads=4,
                    mlp_ratio=1,
                    latent_fit_steps=1,
                    verifier_layers=1,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(31),
        )
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        tokens = jnp.asarray([[int(ch) + 1 for ch in puzzle]], dtype=jnp.int32)
        labels = jnp.asarray([[int(ch) + 1 for ch in solution]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        final_only_logits, final_only_diagnostics = model.run_diffusion(
            tokens,
            train=False,
            return_final_only=True,
        )
        self.assertEqual(logits.shape, (2, 1, 81, 11))
        self.assertEqual(final_only_logits.shape, (1, 81, 11))
        self.assertTrue(bool(jnp.allclose(final_only_logits, logits[-1], rtol=1e-5, atol=1e-5)))
        self.assertEqual(diagnostics["hidden_delta_mean"].shape, (2,))
        self.assertEqual(diagnostics["diffusion_filled_ratio"].shape, (2,))
        self.assertEqual(diagnostics["draft"].shape, (1, 81))
        self.assertEqual(diagnostics["belief_logits"].shape, (1, 81, 9))
        self.assertEqual(final_only_diagnostics["belief_logits"].shape, (1, 81, 9))
        self.assertEqual(model.relation_masks.shape, (4, 81, 81))
        predictions = jnp.argmax(logits[-1], axis=-1)
        given_mask = tokens != 1
        self.assertTrue(bool(jnp.all(predictions[given_mask] == tokens[given_mask])))
        energy = model.verifier_energy(tokens, labels)
        self.assertEqual(energy.shape, (1,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(energy))))
        soft_candidate = jax.nn.one_hot(labels, 11)
        soft_energy = model.verifier_energy_from_probs(tokens, soft_candidate)
        self.assertEqual(soft_energy.shape, (1,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(soft_energy))))
        refined_belief, refine_metrics = model.refine_belief_with_verifier(
            tokens,
            diagnostics["belief_logits"],
            steps=1,
        )
        self.assertEqual(refined_belief.shape, (1, 81, 9))
        self.assertTrue(bool(jnp.isfinite(refine_metrics["belief_refine_loss"])))

    def test_brc_losses_are_finite(self) -> None:
        model = BRCSudokuModel(
            ModelConfig(
                vocab_size=11,
                model_type="brc_sudoku",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=2,
                brc=BRCSudokuConfig(
                    recurrent_steps=2,
                    latent_dim=16,
                    num_heads=4,
                    mlp_ratio=1,
                    latent_fit_steps=1,
                    latent_lr=0.05,
                    verifier_layers=1,
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
            "given_mask": tokens != 1,
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
            "verifier_ranking_accuracy",
            "invalid_board_rate",
            "conflict_count",
        }
        self.assertFalse(legacy_keys & set(metrics))
        for key in (
            "loss",
            "lm_loss",
            "final_lm_loss",
            "mean_lm_loss",
            "latent_fit_loss",
            "fit_given_loss",
            "fit_energy",
            "fit_consistency_loss",
            "fit_prior_loss",
            "latent_update_norm",
            "latent_grad_norm",
            "latent_step_norm",
            "meta_outer_loss",
            "verifier_loss",
            "verifier_accuracy",
            "given_consistency",
            "invalid_rate",
            "conflicts",
            "belief_init_noise_rate",
            "belief_init_uniform_rate",
            "belief_init_teacher_rate",
            "belief_init_corrupt_rate",
            "belief_init_soft_rate",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))
        self.assertEqual(metrics["per_step_loss"].shape, (2,))
        self.assertEqual(metrics["step_loss_weights"].shape, (2,))

    def test_brc_train_step_updates_parameters(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="brc_sudoku",
                seq_len=81,
                grid_height=9,
                grid_width=9,
                d_model=16,
                rollout_steps=1,
                brc=BRCSudokuConfig(
                    recurrent_steps=1,
                    latent_dim=16,
                    num_heads=4,
                    mlp_ratio=1,
                    latent_fit_steps=1,
                    latent_lr=0.05,
                    meta_loss_weight=0.5,
                    verifier_layers=1,
                ),
            ),
            optimizer=OptimizerConfig(learning_rate=1e-4, warmup_epochs=1, warmup_updates=1),
            train=TrainConfig(batch_size=1, epochs=1, optimizer_updates=1),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_step = build_train_step_runner()
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        tokens = jnp.asarray([[int(ch) + 1 for ch in puzzle]], dtype=jnp.int32)
        labels = jnp.asarray([[int(ch) + 1 for ch in solution]], dtype=jnp.int32)
        batch = {
            "inputs": tokens,
            "labels": labels,
            "given_mask": tokens != 1,
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        before = model.z_global[...]
        metrics = train_step(model, optimizer, batch, jax.random.key(34))
        after = model.z_global[...]
        self.assertTrue(bool(jnp.isfinite(metrics["loss"])))
        self.assertTrue(bool(jnp.isfinite(metrics["meta_outer_loss"])))
        self.assertTrue(bool(jnp.isfinite(metrics["latent_fit_loss"])))
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1, num_heads=3, mlp_ratio=2),
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

    def test_trm_rel2d_position_bias_forward_is_finite(self) -> None:
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_embed_len=0,
                    position_encoding="rel2d",
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
        attention_bias = model._attention_bias()
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(model.rel2d_row_bias[...].shape, (3, 5))
        self.assertEqual(model.rel2d_col_bias[...].shape, (3, 5))
        self.assertEqual(attention_bias.shape, (3, 9, 9))
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_embed_len=0,
                    position_encoding="rel2d",
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1,
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1, num_heads=3, mlp_ratio=2, puzzle_embed_len=2),
            ),
            optimizer=OptimizerConfig(
                optimizer_type="adam_atan2",
                learning_rate=1e-4,
                puzzle_embed_learning_rate=1e-4,
                lr_min_ratio=1.0,
                weight_decay=1.0,
                puzzle_embed_weight_decay=1.0,
                warmup_epochs=1,
                warmup_updates=1,
            ),
            train=TrainConfig(batch_size=2, epochs=2, optimizer_updates=2, halt_loss_weight=0.5),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_step = build_trm_act_train_step_runner(config.train.halt_loss_weight)
        batch = {
            "inputs": jnp.asarray(
                [[2, 1, 3, 1, 1, 4, 1, 5, 1], [3, 1, 2, 1, 1, 5, 1, 4, 1]],
                dtype=jnp.int32,
            ),
            "labels": jnp.asarray(
                [[2, 3, 3, 4, 5, 4, 6, 5, 7], [3, 4, 2, 5, 6, 5, 7, 4, 8]],
                dtype=jnp.int32,
            ),
            "given_mask": jnp.asarray(
                [
                    [True, False, True, False, False, True, False, True, False],
                    [True, False, True, False, False, True, False, True, False],
                ],
                dtype=bool,
            ),
            "puzzle_identifiers": jnp.asarray([0, 0], dtype=jnp.int32),
        }
        before = model.puzzle_embed.weights[...]
        metrics, _carry = train_step(
            model,
            optimizer,
            model.initial_carry(batch),
            batch,
            jax.random.key(0),
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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1,
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
            "given_mask": jnp.asarray([[False, False, True, True]], dtype=bool),
        }
        _, metrics = loss_and_metrics(HaltSelectedModel(), batch, False, None, halt_loss_weight=0.1)
        self.assertAlmostEqual(float(metrics["accuracy"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["selected_accuracy"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["selected_step"]), 1.0, places=6)

    def test_clamp_logits_to_given_uses_given_mask_not_blank_id(self) -> None:
        logits = jnp.zeros((2, 1, 4, 6), dtype=jnp.float32)
        inputs = jnp.asarray([[2, 1, 3, 4]], dtype=jnp.int32)
        given_mask = jnp.asarray([[False, True, False, True]], dtype=bool)

        clamped = _clamp_logits_to_given(logits, inputs, given_mask, vocab_size=6)
        predictions = jnp.argmax(clamped, axis=-1)

        self.assertTrue(bool(jnp.all(predictions[:, 0, 1] == 1)))
        self.assertTrue(bool(jnp.all(predictions[:, 0, 3] == 4)))
        self.assertTrue(bool(jnp.all(clamped[:, 0, 0] == logits[:, 0, 0])))

    def test_num_heads_must_divide_d_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            BRCSudokuModel(
                ModelConfig(
                    vocab_size=11,
                    model_type="brc_sudoku",
                    seq_len=81,
                    grid_height=9,
                    grid_width=9,
                    d_model=10,
                    rollout_steps=1,
                    brc=BRCSudokuConfig(recurrent_steps=1, latent_dim=16, num_heads=4),
                ),
                RuntimeConfig(compute_dtype="float32"),
                rngs=nnx.Rngs(0),
            )

    def test_config_loader_rejects_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "legacy.toml"
            config_path.write_text(
                "[model]\n"
                "model_type = \"brc_sudoku\"\n"
                "legacy_field = \"old\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported \\[model\\] field"):
                load_toml_config(str(config_path))

    def test_config_loader_accepts_brc_and_trm_step_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "brc.toml"
            config_path.write_text(
                "[task]\n"
                "type = \"sudoku\"\n"
                "supervision = \"unknown_only\"\n"
                "clamp_given = true\n"
                "\n"
                "[model]\n"
                "model_type = \"brc_sudoku\"\n"
                "d_model = 16\n"
                "\n"
                "[model.brc]\n"
                "recurrent_steps = 2\n"
                "block_layers = 1\n"
                "latent_dim = 16\n"
                "num_heads = 4\n"
                "step_loss_weights = [1.0, 2.0]\n"
                "latent_fit_steps = 1\n"
                "meta_loss_weight = 0.5\n"
                "fit_energy_weight = 0.75\n"
                "denoise_initial_prob = 0.4\n"
                "denoise_teacher_reveal_prob = 0.25\n"
                "denoise_mode_weights = [0.35, 0.20, 0.30, 0.15]\n",
                encoding="utf-8",
            )
            loaded = load_toml_config(str(config_path))
            self.assertEqual(loaded["model_type"], "brc_sudoku")
            self.assertEqual(loaded["task_type"], "sudoku")
            self.assertEqual(loaded["supervision"], "unknown_only")
            self.assertTrue(loaded["clamp_given"])
            self.assertEqual(loaded["brc_recurrent_steps"], 2)
            self.assertEqual(loaded["brc_block_layers"], 1)
            self.assertEqual(loaded["brc_latent_dim"], 16)
            self.assertEqual(loaded["brc_num_heads"], 4)
            self.assertEqual(loaded["brc_step_loss_weights"], [1.0, 2.0])
            self.assertEqual(loaded["brc_meta_loss_weight"], 0.5)
            self.assertEqual(loaded["brc_fit_energy_weight"], 0.75)
            self.assertEqual(loaded["brc_denoise_initial_prob"], 0.4)
            self.assertEqual(loaded["brc_denoise_mode_weights"], [0.35, 0.20, 0.30, 0.15])

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
                deep_recursion=1,
                latent_recursion=1,
                block_layers=1, num_heads=3, mlp_ratio=2),
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

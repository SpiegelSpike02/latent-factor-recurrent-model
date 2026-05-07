from __future__ import annotations

import inspect
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
    LFRMConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TRMConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.models import LatentFactorRecurrentModel, TinyRecursiveModel
import lfrm.models.lfrm as lfrm_module
from lfrm.jax_defaults import apply_jax_defaults
from lfrm.training import (
    build_trm_act_train_step_runner,
    create_model,
    create_optimizer,
    load_checkpoint,
    loss_and_metrics,
    save_checkpoint,
    trm_dense_unroll_loss_and_metrics,
)


class LFRMModelTests(unittest.TestCase):
    def _make_lfrm_model(
        self,
        *,
        num_steps: int = 3,
        num_slots: int = 5,
    ) -> LatentFactorRecurrentModel:
        return LatentFactorRecurrentModel(
            ModelConfig(
                vocab_size=11,
                model_type="lfrm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=num_steps,
                lfrm=LFRMConfig(
                    belief_dim=9,
                    num_slots=num_slots,
                    num_heads=4,
                    latent_processor_layers=1,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(12),
        )

    def test_forward_all_steps_shape_and_diagnostics(self) -> None:
        model = self._make_lfrm_model(num_steps=2)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(diagnostics["digit_logits"].shape, (1, 9, 9))
        self.assertEqual(diagnostics["q"].shape, (1, 9, 9))
        self.assertEqual(diagnostics["slots"].shape, (1, 5, 12))
        self.assertEqual(diagnostics["hidden_delta_mean"].shape, (2,))
        self.assertEqual(diagnostics["quality_logits"].shape, (2, 1))
        self.assertEqual(int(diagnostics["unroll_steps"]), 2)

    def test_given_cells_are_clamped(self) -> None:
        model = self._make_lfrm_model(num_steps=2)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        predictions = jnp.argmax(logits[-1], axis=-1)
        self.assertEqual(int(predictions[0, 0]), 2)
        self.assertEqual(int(predictions[0, 2]), 3)
        self.assertEqual(int(predictions[0, 5]), 4)
        self.assertEqual(int(predictions[0, 7]), 5)
        given_q = jnp.take(diagnostics["q"][0], jnp.asarray([0, 2, 5, 7]), axis=0)
        expected_digits = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
        self.assertTrue(bool(jnp.all(jnp.argmax(given_q, axis=-1) == expected_digits)))

    def test_lfrm_uses_no_sudoku_specific_relations(self) -> None:
        source = inspect.getsource(lfrm_module)
        self.assertNotIn("tasks.sudoku", source)
        self.assertNotIn("sudoku_relation", source)
        self.assertNotIn("sudoku_unit", source)
        self.assertNotIn("row_unit_matrix", source)
        self.assertNotIn("box_relation", source)
        self.assertFalse(hasattr(self._make_lfrm_model(), "row_unit_matrix"))

    def test_symbol_equivariance(self) -> None:
        model = self._make_lfrm_model(num_steps=2, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        permutation = jnp.asarray([2, 0, 1, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        digit_ids = tokens - 2
        permuted_digits = permutation[jnp.clip(digit_ids, 0, 8)] + 2
        permuted_tokens = jnp.where(tokens == 1, tokens, permuted_digits)
        logits, _ = model.forward_all_steps_with_diagnostics(tokens, train=False)
        permuted_logits, _ = model.forward_all_steps_with_diagnostics(permuted_tokens, train=False)
        restored_digit_logits = permuted_logits[..., 2:][..., permutation]
        self.assertTrue(bool(jnp.allclose(logits[..., 2:], restored_digit_logits, atol=1e-4, rtol=1e-4)))

    def test_initial_hidden_is_symbol_invariant(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        permutation = jnp.asarray([2, 0, 1, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        digit_ids = tokens - 2
        permuted_digits = permutation[jnp.clip(digit_ids, 0, 8)] + 2
        permuted_tokens = jnp.where(tokens == 1, tokens, permuted_digits)
        state, _, _, _ = model._initial_state(tokens, train=False, dropout_key=None)
        permuted_state, _, _, _ = model._initial_state(permuted_tokens, train=False, dropout_key=None)
        self.assertTrue(bool(jnp.allclose(state[0], permuted_state[0], atol=1e-6, rtol=1e-6)))

    def test_symbol_conditioned_slot_readout_shape(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        state, initial_logits, initial_q, condition_mask = model._initial_state(tokens, train=False, dropout_key=None)
        h, _, logits, q, slots = state
        given_channels = model._given_channels(initial_q, condition_mask)
        micro_tokens, symbol_context = model._cell_symbol_context(h, logits, q, given_channels)
        message, routing = model._slots_to_cell_symbols(micro_tokens, slots)
        self.assertEqual(micro_tokens.shape, (1, 9, 9, 12))
        self.assertEqual(symbol_context.shape, (1, 9, 12))
        self.assertEqual(message.shape, (1, 9, 9, 12))
        self.assertEqual(routing.shape, (1, 9, 9, 4))

    def test_training_losses_are_finite(self) -> None:
        model = self._make_lfrm_model(num_steps=2, num_slots=4)
        batch = {
            "inputs": jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 3, 4, 5, 4, 6, 5, 7]], dtype=jnp.int32),
            "given_mask": jnp.asarray([[True, False, True, False, False, True, False, True, False]], dtype=bool),
            "puzzle_identifiers": jnp.asarray([0], dtype=jnp.int32),
        }
        _, metrics = loss_and_metrics(
            model,
            batch,
            True,
            jax.random.key(1),
            dense_loss_weight=0.0,
            final_loss_weight=1.0,
            terminal_residual_weight=0.1,
            slot_consistency_weight=0.01,
            slot_usage_weight=0.001,
        )
        self.assertNotIn("validity_loss", metrics)
        for key in (
            "loss",
            "q_loss",
            "q_selected_blank_ce_loss",
            "q_selected_blank_cell_accuracy",
            "q_selected_solved_rate",
            "q_selected_step",
            "oracle_step",
            "slot_consistency_loss",
            "slot_usage_entropy",
            "terminal_belief_delta",
            "terminal_belief_mse",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))
        self.assertEqual(metrics["per_step_loss"].shape, (2,))
        self.assertEqual(metrics["per_step_hidden_delta"].shape, (2,))
        self.assertEqual(metrics["per_step_quality_score"].shape, (2,))

    def test_zero_weight_training_skips_expensive_diagnostics(self) -> None:
        model = self._make_lfrm_model(num_steps=2, num_slots=4)
        batch = {
            "inputs": jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 3, 4, 5, 4, 6, 5, 7]], dtype=jnp.int32),
            "given_mask": jnp.asarray([[True, False, True, False, False, True, False, True, False]], dtype=bool),
        }
        _, metrics = loss_and_metrics(
            model,
            batch,
            True,
            jax.random.key(1),
            dense_loss_weight=0.0,
            final_loss_weight=1.0,
            terminal_residual_weight=0.0,
        )
        self.assertNotIn("terminal_belief_delta", metrics)
        self.assertNotIn("terminal_belief_mse", metrics)

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
            self.assertEqual(os.environ["TF_GPU_ALLOCATOR"], "cuda_malloc_async")
            self.assertIn("--some_existing_flag=true", os.environ["XLA_FLAGS"].split())
            self.assertIn("--xla_gpu_triton_gemm_any=true", os.environ["XLA_FLAGS"].split())

    def test_gradient_path_finite(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_slots=3)
        batch = {
            "inputs": jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 3, 4, 5, 4, 6, 5, 7]], dtype=jnp.int32),
            "given_mask": jnp.asarray([[True, False, True, False, False, True, False, True, False]], dtype=bool),
        }

        def objective(m):
            return loss_and_metrics(
                m,
                batch,
                True,
                jax.random.key(1),
                dense_loss_weight=0.0,
                final_loss_weight=1.0,
                terminal_residual_weight=0.1,
                slot_consistency_weight=0.01,
                slot_usage_weight=0.001,
            )[0]

        value, grads = nnx.value_and_grad(objective)(model)
        grad_leaves = [leaf for leaf in jax.tree.leaves(grads) if hasattr(leaf, "shape")]
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves))

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
        self.assertAlmostEqual(float(metrics["blank_cell_accuracy"]), 0.75, places=6)
        self.assertAlmostEqual(float(metrics["solved_rate"]), 0.5, places=6)

    def test_create_model_is_lfrm_only(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="lfrm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=1,
                lfrm=LFRMConfig(
                    belief_dim=9,
                    num_slots=3,
                ),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(config), LatentFactorRecurrentModel)
        trm_config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=1,
                trm=TRMConfig(),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(trm_config), TinyRecursiveModel)
        invalid = ExperimentConfig(
            model=ModelConfig(vocab_size=11, model_type="legacy_shared_block"),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        with self.assertRaisesRegex(ValueError, "Only model_type='lfrm' or model_type='trm'"):
            create_model(invalid)

    def test_trm_forward_and_losses_are_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=2,
                trm=TRMConfig(h_cycles=1, l_cycles=1, l_layers=1, num_heads=3, mlp_ratio=2),
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
            include_layer_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(diagnostics["quality_logits"].shape, (2, 1))
        self.assertEqual(diagnostics["l_logits"].shape, (2, 1, 9, 11))
        self.assertEqual(diagnostics["h_hidden_delta_mean"].shape, (2,))
        self.assertEqual(diagnostics["l_hidden_delta_mean"].shape, (2,))
        _, metrics = loss_and_metrics(model, batch, True, jax.random.key(2), q_loss_weight=0.1)
        for key in ("loss", "q_loss", "q_selected_blank_ce_loss", "blank_cell_accuracy"):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))
        dense_loss, dense_metrics = trm_dense_unroll_loss_and_metrics(
            model,
            batch,
            True,
            jax.random.key(3),
            sequence_loss_weight=0.1,
        )
        self.assertIn("sequence_loss", dense_metrics)
        self.assertNotIn("per_step_h_loss", dense_metrics)
        self.assertNotIn("per_step_l_loss", dense_metrics)
        self.assertTrue(bool(jnp.isfinite(dense_metrics["sequence_loss"])))
        self.assertAlmostEqual(
            float(dense_loss),
            float(dense_metrics["blank_ce_loss"] + 0.1 * dense_metrics["sequence_loss"]),
            places=5,
        )

    def test_trm_breaks_blank_symbol_symmetry(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=2,
                trm=TRMConfig(h_cycles=1, l_cycles=1, l_layers=1, num_heads=3, mlp_ratio=2),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(21),
        )
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=jnp.asarray([0], dtype=jnp.int32),
            train=False,
            include_layer_diagnostics=True,
        )
        blank_digit_logits = logits[-1, 0, 1, 2:]
        self.assertGreater(float(jnp.std(blank_digit_logits)), 1e-6)
        self.assertIn("quality_logits", diagnostics)

    def test_trm_rel2d_position_bias_forward_is_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=2,
                trm=TRMConfig(
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_emb_len=0,
                    pos_encodings="rel2d",
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
            include_layer_diagnostics=True,
        )
        attention_bias = model._attention_bias()
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(model.rel2d_row_bias[...].shape, (3, 5))
        self.assertEqual(model.rel2d_col_bias[...].shape, (3, 5))
        self.assertEqual(attention_bias.shape, (3, 9, 9))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertIn("quality_logits", diagnostics)

    def test_trm_local_mixing_forward_is_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=2,
                trm=TRMConfig(
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_emb_len=0,
                    pos_encodings="rel2d",
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
            include_layer_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertIn("l_logits", diagnostics)

    def test_trm_gated_dual_attention_forward_is_finite(self) -> None:
        model = TinyRecursiveModel(
            ModelConfig(
                vocab_size=11,
                model_type="trm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=2,
                trm=TRMConfig(
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    mlp_t=False,
                    puzzle_emb_len=0,
                    pos_encodings="rel2d",
                    attention_type="gated_dual",
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(26),
        )
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=jnp.asarray([0], dtype=jnp.int32),
            train=False,
            include_layer_diagnostics=True,
        )
        attention = model.blocks[0].attention
        self.assertEqual(logits.shape, (2, 1, 9, 11))
        self.assertEqual(attention.local_distance.shape, (9, 9))
        self.assertEqual(attention.local_distance_logit[...].shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertIn("l_logits", diagnostics)

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
                    num_steps=1,
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
                num_steps=2,
                trm=TRMConfig(h_cycles=1, l_cycles=1, l_layers=1, num_heads=3, mlp_ratio=2, puzzle_emb_len=2),
            ),
            optimizer=OptimizerConfig(
                learning_rate=1e-4,
                lr_min_ratio=1.0,
                weight_decay=1.0,
                warmup_steps=1,
            ),
            train=TrainConfig(batch_size=2, max_steps=2, q_loss_weight=0.5),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        train_step = build_trm_act_train_step_runner(config.train.q_loss_weight)
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
        before = model.puzzle_emb.weights[...]
        metrics, _carry = train_step(
            model,
            optimizer,
            model.initial_carry(batch),
            batch,
            jax.random.key(0),
        )
        after = model.puzzle_emb.weights[...]

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
                num_steps=2,
                trm=TRMConfig(
                    h_cycles=1,
                    l_cycles=1,
                    l_layers=1,
                    num_heads=3,
                    mlp_ratio=2,
                    puzzle_emb_ndim=12,
                    puzzle_emb_len=2,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(22),
        )
        params = nnx.state(model, nnx.Param)
        self.assertEqual(model.prefix_len, 2)
        self.assertEqual(model.puzzle_emb.weights[...].shape, (5, 12))
        self.assertNotIn("h_init", str(params))
        self.assertNotIn("l_init", str(params))

    def test_q_selected_metrics_use_quality_head_step(self) -> None:
        class QualitySelectedModel:
            def forward_all_steps_with_diagnostics(self, inputs, train: bool, dropout_key=None):
                del inputs, train, dropout_key
                logits = jnp.full((2, 1, 4, 11), -10.0)
                logits = logits.at[0, 0, 0, 2].set(10.0)
                logits = logits.at[0, 0, 1, 1].set(10.0)
                logits = logits.at[1, 0, 0, 2].set(10.0)
                logits = logits.at[1, 0, 1, 3].set(10.0)
                diagnostics = {
                    "hidden_delta_mean": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
                    "quality_logits": jnp.asarray([[0.0], [4.0]], dtype=jnp.float32),
                }
                return logits, diagnostics

        batch = {
            "inputs": jnp.zeros((1, 4), dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 0, 0]], dtype=jnp.int32),
            "given_mask": jnp.asarray([[False, False, True, True]], dtype=bool),
        }
        _, metrics = loss_and_metrics(QualitySelectedModel(), batch, False, None)
        self.assertAlmostEqual(float(metrics["blank_cell_accuracy"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["q_selected_blank_cell_accuracy"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["q_selected_step"]), 2.0, places=6)

    def test_num_heads_must_divide_d_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            LatentFactorRecurrentModel(
                ModelConfig(
                    vocab_size=11,
                    model_type="lfrm",
                    seq_len=9,
                    grid_height=3,
                    grid_width=3,
                    d_model=10,
                    num_steps=1,
                    lfrm=LFRMConfig(belief_dim=9, num_heads=4),
                ),
                RuntimeConfig(compute_dtype="float32"),
                rngs=nnx.Rngs(0),
            )

    def test_config_loader_rejects_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "legacy.toml"
            config_path.write_text(
                "[model]\n"
                "model_type = \"lfrm\"\n"
                "legacy_field = \"old\"\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported \\[model\\] field"):
                load_toml_config(str(config_path))

    def test_checkpoint_round_trip_restores_step(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                vocab_size=11,
                model_type="lfrm",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=12,
                num_steps=1,
                lfrm=LFRMConfig(belief_dim=9, num_slots=3),
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

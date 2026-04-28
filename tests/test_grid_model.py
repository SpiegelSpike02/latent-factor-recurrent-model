from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.cli import load_toml_config
from lfrm.config import (
    DataConfig,
    ExperimentConfig,
    LFRMConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.models import LatentFactorRecurrentModel
import lfrm.models.lfrm as lfrm_module
from lfrm.training import create_model, loss_and_metrics


class LFRMModelTests(unittest.TestCase):
    def _make_lfrm_model(
        self,
        *,
        num_steps: int = 3,
        num_branches: int = 3,
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
                    num_branches=num_branches,
                    num_heads=4,
                    latent_processor_layers=1,
                    energy_hidden_dim=16,
                    freeze_conditioned_state=True,
                ),
            ),
            RuntimeConfig(compute_dtype="float32"),
            rngs=nnx.Rngs(12),
        )

    def test_forward_final_only_shape_and_branch_metrics(self) -> None:
        model = self._make_lfrm_model(num_steps=2)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        logits, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        self.assertEqual(logits.shape, (1, 1, 9, 11))
        self.assertEqual(diagnostics["branch_logits"].shape, (1, 3, 9, 11))
        self.assertEqual(diagnostics["branch_digit_logits"].shape, (1, 3, 9, 9))
        self.assertEqual(diagnostics["branch_q"].shape, (1, 3, 9, 9))
        self.assertEqual(diagnostics["branch_slots"].shape, (1, 3, 5, 12))
        self.assertEqual(diagnostics["branch_symbol_context"].shape, (1, 3, 9, 12))
        self.assertEqual(diagnostics["branch_energy"].shape, (1, 3))
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
        given_q = jnp.take(diagnostics["branch_q"][0], jnp.asarray([0, 2, 5, 7]), axis=1)
        expected_digits = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
        self.assertTrue(bool(jnp.all(jnp.argmax(given_q, axis=-1) == expected_digits[None, :])))

    def test_lfrm_uses_no_sudoku_specific_relations(self) -> None:
        source = inspect.getsource(lfrm_module)
        self.assertNotIn("tasks.sudoku", source)
        self.assertNotIn("sudoku_relation", source)
        self.assertNotIn("sudoku_unit", source)
        self.assertNotIn("row_unit_matrix", source)
        self.assertNotIn("box_relation", source)
        self.assertFalse(hasattr(self._make_lfrm_model(), "row_unit_matrix"))

    def test_symbol_equivariance(self) -> None:
        model = self._make_lfrm_model(num_steps=2, num_branches=2, num_slots=4)
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
        model = self._make_lfrm_model(num_steps=1, num_branches=2, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        permutation = jnp.asarray([2, 0, 1, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        digit_ids = tokens - 2
        permuted_digits = permutation[jnp.clip(digit_ids, 0, 8)] + 2
        permuted_tokens = jnp.where(tokens == 1, tokens, permuted_digits)
        state, _, _, _ = model._initial_state(tokens, train=False, dropout_key=None)
        permuted_state, _, _, _ = model._initial_state(permuted_tokens, train=False, dropout_key=None)
        self.assertTrue(bool(jnp.allclose(state[0], permuted_state[0], atol=1e-6, rtol=1e-6)))

    def test_symbol_conditioned_slot_readout_shape(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_branches=2, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        state, initial_logits, initial_q, condition_mask = model._initial_state(tokens, train=False, dropout_key=None)
        h, logits, q, slots = state
        given_channels = model._given_channels(initial_q, condition_mask, h.shape[1])
        micro_tokens, symbol_context = model._cell_symbol_context(h, logits, q, given_channels)
        message, routing = model._slots_to_cell_symbols(micro_tokens, slots)
        self.assertEqual(micro_tokens.shape, (1, 2, 9, 9, 12))
        self.assertEqual(symbol_context.shape, (1, 2, 9, 12))
        self.assertEqual(message.shape, (1, 2, 9, 9, 12))
        self.assertEqual(routing.shape, (1, 2, 9, 9, 4))

    def test_energy_is_symbol_invariant(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_branches=2, num_slots=4)
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)
        permutation = jnp.asarray([2, 0, 1, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        _, diagnostics = model.forward_all_steps_with_diagnostics(tokens, train=False)
        q = diagnostics["branch_q"]
        symbol_context = diagnostics["branch_symbol_context"]
        energy = model._energy(q, diagnostics["branch_h"], diagnostics["branch_slots"], symbol_context)
        permuted_energy = model._energy(
            q[..., permutation],
            diagnostics["branch_h"],
            diagnostics["branch_slots"],
            symbol_context[:, :, permutation, :],
        )
        self.assertTrue(bool(jnp.allclose(energy, permuted_energy, atol=1e-5, rtol=1e-5)))

    def test_training_losses_are_finite(self) -> None:
        model = self._make_lfrm_model(num_steps=2, num_branches=2, num_slots=4)
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
            step_loss_weighting="final",
            terminal_residual_weight=0.1,
            energy_loss_weight=0.05,
            slot_consistency_weight=0.01,
            slot_usage_weight=0.001,
        )
        self.assertNotIn("validity_loss", metrics)
        for key in (
            "loss",
            "branch_min_ce",
            "branch_mean_ce",
            "energy_margin_loss",
            "slot_consistency_loss",
            "slot_usage_entropy",
            "terminal_belief_delta",
            "terminal_belief_mse",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(bool(jnp.isfinite(metrics[key])))

    def test_gradient_path_finite(self) -> None:
        model = self._make_lfrm_model(num_steps=1, num_branches=2, num_slots=3)
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
                step_loss_weighting="final",
                terminal_residual_weight=0.1,
                energy_loss_weight=0.05,
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
                    num_branches=2,
                    freeze_conditioned_state=True,
                ),
            ),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        self.assertIsInstance(create_model(config), LatentFactorRecurrentModel)
        invalid = ExperimentConfig(
            model=ModelConfig(vocab_size=11, model_type="legacy_shared_block"),
            optimizer=OptimizerConfig(),
            train=TrainConfig(),
            data=DataConfig(),
            runtime=RuntimeConfig(compute_dtype="float32"),
            wandb=WandbConfig(),
        )
        with self.assertRaisesRegex(ValueError, "Only model_type='lfrm'"):
            create_model(invalid)

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


if __name__ == "__main__":
    unittest.main()

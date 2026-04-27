from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from config import AttentionConfig, ClueConfig, ComputeConfig, ModelConfig, RelationConfig, RuntimeConfig, TransitionConfig
from models import GridReasoningModel
from tasks.sudoku import apply_given_logits
from training import loss_and_metrics


class GridModelTests(unittest.TestCase):
    def test_apply_given_logits_fixes_clues(self) -> None:
        logits = jnp.zeros((1, 4, 11), dtype=jnp.float32)
        inputs = jnp.asarray([[2, 1, 5, 1]], dtype=jnp.int32)
        given_mask = jnp.asarray([[True, False, True, False]], dtype=bool)
        effective_logits = apply_given_logits(logits, inputs, given_mask)
        predictions = jnp.argmax(effective_logits, axis=-1)
        self.assertEqual(int(predictions[0, 0]), 2)
        self.assertEqual(int(predictions[0, 2]), 5)

    def test_forward_shapes(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.zeros((2, 9), dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
            ),
            runtime,
            rngs=nnx.Rngs(0),
        )
        logits = model(tokens, train=True, dropout_key=jax.random.key(0))
        self.assertEqual(logits.shape, (2, 9, 11))
        step_logits = model.forward_all_steps(tokens, train=True, dropout_key=jax.random.key(1))
        self.assertEqual(step_logits.shape, (2, 2, 9, 11))

    def test_recurrent_transformer_shapes(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.zeros((2, 9), dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                model_type="recurrent_transformer",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
                attention=AttentionConfig(num_heads=3),
                clues=ClueConfig(freeze_state=False),
            ),
            runtime,
            rngs=nnx.Rngs(4),
        )
        step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            train=True,
            dropout_key=jax.random.key(2),
        )
        self.assertEqual(len(model.blocks), 2)
        self.assertIsNot(model.blocks[0], model.blocks[1])
        self.assertEqual(step_logits.shape, (2, 2, 9, 11))
        self.assertIn("hidden_delta_mean", diagnostics)
        self.assertNotIn("rho_mean", diagnostics)
        self.assertNotIn("alpha_mean", diagnostics)

    def test_recurrent_transformer_rejects_damped_transition(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        with self.assertRaisesRegex(ValueError, "transition.type"):
            GridReasoningModel(
                ModelConfig(
                    vocab_size=11,
                    model_type="recurrent_transformer",
                    seq_len=9,
                    grid_height=3,
                    grid_width=3,
                    d_model=18,
                    d_ff=36,
                    num_steps=2,
                    transition=TransitionConfig(type="damped"),
                    attention=AttentionConfig(num_heads=3),
                ),
                runtime,
                rngs=nnx.Rngs(7),
            )

    def test_universal_transformer_can_use_residual_transition(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.zeros((2, 9), dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                model_type="universal_transformer",
                communication_type="attention",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
                transition=TransitionConfig(type="residual"),
                attention=AttentionConfig(num_heads=3),
            ),
            runtime,
            rngs=nnx.Rngs(5),
        )
        step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            train=True,
            dropout_key=jax.random.key(3),
        )
        self.assertEqual(step_logits.shape, (2, 2, 9, 11))
        self.assertIn("hidden_delta_mean", diagnostics)
        self.assertNotIn("rho_mean", diagnostics)
        self.assertNotIn("alpha_mean", diagnostics)

    def test_universal_transformer_inner_compute_depth(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.zeros((2, 9), dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                model_type="universal_transformer",
                communication_type="attention",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
                attention=AttentionConfig(num_heads=3),
                compute=ComputeConfig(
                    inner_steps=2,
                    layers_per_step=2,
                    grad_inner_steps=1,
                    reinject_input=True,
                ),
            ),
            runtime,
            rngs=nnx.Rngs(9),
        )
        step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            train=True,
            dropout_key=jax.random.key(5),
        )
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(step_logits.shape, (2, 2, 9, 11))
        self.assertEqual(diagnostics["hidden_delta_mean"].shape, (2,))

    def test_universal_transformer_freezes_clue_state_with_residual_transition(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.asarray([[2, 1, 3, 1, 1, 4, 1, 5, 1]], dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                model_type="universal_transformer",
                communication_type="attention",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
                transition=TransitionConfig(type="residual"),
                attention=AttentionConfig(num_heads=3),
                clues=ClueConfig(freeze_state=True),
            ),
            runtime,
            rngs=nnx.Rngs(8),
        )
        initial_hidden, _, given_mask = model.embeddings(tokens, train=False, dropout_key=None)
        _, step_hidden, _, _, _ = model._prepare_recurrence_inputs(tokens, train=False, dropout_key=None)
        first_step_delta = jnp.linalg.norm((step_hidden[0] - initial_hidden).astype(jnp.float32), axis=-1)
        self.assertTrue(bool(jnp.allclose(first_step_delta[given_mask], 0.0)))

    def test_inner_compute_gradient_horizon_path_runs(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        tokens = jnp.zeros((2, 9), dtype=jnp.int32)

        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                model_type="universal_transformer",
                communication_type="attention",
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=3,
                attention=AttentionConfig(num_heads=3),
                compute=ComputeConfig(inner_steps=2, grad_inner_steps=1),
            ),
            runtime,
            rngs=nnx.Rngs(6),
        )
        step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
            tokens,
            train=True,
            dropout_key=jax.random.key(4),
        )
        self.assertEqual(step_logits.shape, (3, 2, 9, 11))
        self.assertEqual(diagnostics["hidden_delta_mean"].shape, (3,))

    def test_given_mask_and_relation_wiring(self) -> None:
        runtime = RuntimeConfig(compute_dtype="float32")
        model = GridReasoningModel(
            ModelConfig(
                vocab_size=11,
                seq_len=9,
                grid_height=3,
                grid_width=3,
                d_model=18,
                d_ff=36,
                num_steps=2,
                relation=RelationConfig(include_global=True),
            ),
            runtime,
            rngs=nnx.Rngs(3),
        )
        tokens = jnp.asarray([[1, 2, 1, 3, 4, 1, 5, 6, 7]], dtype=jnp.int32)
        hidden, cell_type_embedding, given_mask = model.embeddings(
            tokens,
            train=False,
            dropout_key=None,
        )
        self.assertEqual(hidden.shape, (1, 9, 18))
        self.assertEqual(cell_type_embedding.shape, (1, 9, 18))
        self.assertEqual(given_mask.shape, (1, 9))
        self.assertFalse(jnp.array_equal(cell_type_embedding[:, 0], cell_type_embedding[:, 1]))

        self.assertEqual(model.row_relation.shape, (9, 9))
        self.assertEqual(model.col_relation.shape, (9, 9))
        self.assertEqual(model.box_relation.shape, (9, 9))
        self.assertEqual(model.global_relation.shape, (9, 9))
        self.assertEqual(model.num_box_units, 9)
        self.assertEqual(model.row_unit_matrix.shape, (3, 9))
        self.assertEqual(model.col_unit_matrix.shape, (3, 9))
        self.assertEqual(model.box_unit_matrix.shape, (9, 9))
        self.assertTrue(jnp.allclose(jnp.diag(model.row_relation), 0.0))
        self.assertTrue(jnp.allclose(jnp.diag(model.col_relation), 0.0))
        self.assertTrue(jnp.allclose(jnp.diag(model.box_relation), 0.0))
        self.assertTrue(jnp.allclose(jnp.diag(model.global_relation), 0.0))
        self.assertAlmostEqual(float(jnp.sum(model.row_relation[0])), 1.0, places=6)
        self.assertAlmostEqual(float(jnp.sum(model.col_relation[0])), 1.0, places=6)
        self.assertAlmostEqual(float(jnp.sum(model.global_relation[0])), 1.0, places=6)
        self.assertTrue(jnp.allclose(model.box_relation, 0.0))

    def test_solved_rate_metric(self) -> None:
        class FixedModel:
            class Config:
                num_steps = 1
                seq_len = 4
                grid_height = 2
                grid_width = 2
                clues = ClueConfig(fix_outputs=True)

                @property
                def fix_clue_outputs(self):
                    return self.clues.fix_outputs

            config = Config()
            row_ids = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
            col_ids = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
            box_ids = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
            row_unit_matrix = jax.nn.one_hot(row_ids, 2, dtype=jnp.float32).T
            col_unit_matrix = jax.nn.one_hot(col_ids, 2, dtype=jnp.float32).T
            box_unit_matrix = jax.nn.one_hot(box_ids, 2, dtype=jnp.float32).T

            def forward_all_steps_with_diagnostics(self, inputs, train: bool, dropout_key=None):
                del train, dropout_key
                logits = jnp.full((2, 4, 11), -10.0)
                # Example 0 solves both blank cells, example 1 misses one.
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


if __name__ == "__main__":
    unittest.main()

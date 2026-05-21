from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from lfrm.models import BRCModel, TinyRecursiveModel, UnifiedReasoningModel
from lfrm.training.factory import GridReasoningModel
from lfrm.training.losses import (
    supervised_loss_mask,
    target_probability as token_target_probability,
    token_cross_entropy,
)
from lfrm.training.metrics import maybe_path_metrics


def _maybe_path_metrics(
    model: GridReasoningModel,
    predictions: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
) -> dict[str, jax.Array]:
    return maybe_path_metrics(
        task_type=getattr(getattr(model, "config", None), "task_type", "sudoku"),
        predictions=predictions,
        targets=targets,
        loss_mask=loss_mask,
    )


def _example_mask(batch: dict[str, jax.Array], targets: jax.Array) -> jax.Array:
    return batch.get("example_mask", jnp.ones((targets.shape[0],), dtype=jnp.float32)).astype(jnp.float32)


def _apply_example_mask(loss_mask: jax.Array, example_mask: jax.Array) -> jax.Array:
    return loss_mask.astype(jnp.float32) * example_mask[:, None]


def _masked_example_mean(values: jax.Array, example_mask: jax.Array) -> jax.Array:
    return jnp.sum(values.astype(jnp.float32) * example_mask) / jnp.maximum(jnp.sum(example_mask), 1.0)


def _permute_sudoku_digits(
    inputs: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    *,
    train: bool,
) -> tuple[jax.Array, jax.Array]:
    if not train:
        return inputs, targets
    perms = jax.vmap(jax.random.permutation, in_axes=(0, None))(
        jax.random.split(key, inputs.shape[0]),
        jnp.arange(9, dtype=jnp.int32),
    )

    def apply_perm(tokens: jax.Array) -> jax.Array:
        digit_ids = jnp.clip(tokens - 2, 0, 8)
        permuted = jnp.take_along_axis(perms, digit_ids, axis=1) + 2
        return jnp.where(tokens >= 2, permuted, tokens).astype(jnp.int32)

    return apply_perm(inputs), apply_perm(targets)


def _mix_training_belief(
    model: BRCModel,
    inputs: jax.Array,
    targets: jax.Array,
    self_belief: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mode_key, mask_key, teacher_corrupt_key, self_corrupt_key = jax.random.split(key, 4)
    brc = model.brc
    mode_weights = jnp.asarray(brc.denoise_mode_weights, dtype=jnp.float32)
    mode_logits = jnp.log(mode_weights / jnp.maximum(jnp.sum(mode_weights), 1e-6))
    mode = jax.random.categorical(mode_key, mode_logits, shape=(inputs.shape[0],))

    active = targets != 0
    prior_belief = model.initial_belief_logits(inputs)
    teacher_source = model.belief_logits_from_tokens(inputs, targets, active)
    teacher_belief = _corrupt_belief(
        model,
        teacher_source,
        active,
        teacher_corrupt_key,
        replace_prob=1.0 - brc.denoise_teacher_reveal_prob,
        mask_prob=0.20,
    )
    self_belief = _corrupt_belief(
        model,
        jax.lax.stop_gradient(self_belief.astype(jnp.float32)),
        active,
        self_corrupt_key,
        replace_prob=0.15,
        mask_prob=0.10,
    )

    stacked = jnp.stack([prior_belief, teacher_belief, self_belief], axis=0)
    selector = jax.nn.one_hot(mode, 3, dtype=jnp.float32).T[:, :, None, None]
    mixed_belief = jnp.sum(stacked * selector, axis=0)
    mixed_belief = jnp.where(active[..., None], mixed_belief, prior_belief)
    return mixed_belief.astype(jnp.float32), mode


def _corrupt_belief(
    model: BRCModel,
    source_belief: jax.Array,
    active: jax.Array,
    key: jax.Array,
    *,
    replace_prob: float,
    mask_prob: float,
) -> jax.Array:
    replace_key, token_key, mask_key, force_key = jax.random.split(key, 4)
    random_class_ids = jax.random.randint(
        token_key,
        active.shape,
        minval=0,
        maxval=model.config.vocab_size,
        dtype=jnp.int32,
    )
    random_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(random_class_ids, model.belief_vocab_size)
    replace = active & (jax.random.uniform(replace_key, active.shape) < replace_prob)
    mask = active & (jax.random.uniform(mask_key, active.shape) < mask_prob)
    active_scores = jnp.where(active, jax.random.uniform(force_key, active.shape), -jnp.inf)
    forced_index = jnp.argmax(active_scores, axis=-1)
    forced = jnp.zeros_like(active).at[jnp.arange(active.shape[0]), forced_index].set(True)
    replace = replace | (forced & active)
    corrupted = jnp.where(replace[..., None], random_logits.astype(jnp.float32), source_belief.astype(jnp.float32))
    corrupted = jnp.where(mask[..., None], 0.0, corrupted)
    return model._normalize_belief_logits(corrupted, jnp.zeros_like(active, dtype=jnp.int32))


def _sudoku_board_metrics(
    model: BRCModel,
    predictions: jax.Array,
    inputs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    context_mask = inputs != 0
    context_f32 = context_mask.astype(jnp.float32)
    context_consistency = (
        jnp.sum(((predictions == inputs) & context_mask).astype(jnp.float32))
        / jnp.maximum(jnp.sum(context_f32), 1.0)
    )
    if model.config.task_type != "sudoku":
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return context_consistency, zero, zero
    if model.config.grid_height != 9 or model.config.grid_width != 9 or model.config.vocab_size < 11:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return context_consistency, zero, zero

    valid_digits = (predictions >= 2) & (predictions <= 10)
    one_hot = jax.nn.one_hot(jnp.clip(predictions - 2, 0, 8), 9)
    grid = one_hot.reshape(predictions.shape[0], 9, 9, 9)
    row_counts = jnp.sum(grid, axis=2)
    col_counts = jnp.sum(grid, axis=1)
    box_values = jnp.take(one_hot, model.box_indices, axis=1)
    box_counts = jnp.sum(box_values, axis=2)
    unit_valid = (
        jnp.all(row_counts == 1.0, axis=(1, 2))
        & jnp.all(col_counts == 1.0, axis=(1, 2))
        & jnp.all(box_counts == 1.0, axis=(1, 2))
        & jnp.all(valid_digits, axis=1)
    )
    row_conflicts = jnp.sum(jnp.maximum(row_counts - 1.0, 0.0), axis=(1, 2))
    col_conflicts = jnp.sum(jnp.maximum(col_counts - 1.0, 0.0), axis=(1, 2))
    box_conflicts = jnp.sum(jnp.maximum(box_counts - 1.0, 0.0), axis=(1, 2))
    conflicts = jnp.mean(row_conflicts + col_conflicts + box_conflicts)
    invalid_rate = 1.0 - jnp.mean(unit_valid.astype(jnp.float32))
    return context_consistency, invalid_rate, conflicts


def _normalized_step_loss_weights(configured: tuple[float, ...] | None, rollout_steps: int) -> jax.Array:
    if configured is None:
        weights = jnp.arange(1, rollout_steps + 1, dtype=jnp.float32)
    else:
        weights = jnp.asarray(configured, dtype=jnp.float32)
    return weights / jnp.maximum(jnp.sum(weights), 1e-6)


def _refine_belief_step(
    model: BRCModel,
    inputs: jax.Array,
    belief_logits: jax.Array,
    base_embeddings: jax.Array,
    time_embedding: jax.Array,
    step_index: jax.Array,
    *,
    train: bool,
    dropout_key: jax.Array | None,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    cell_input = model._cell_embeddings(
        inputs,
        belief_logits,
        base_embeddings,
        time_embedding,
        train=train,
        dropout_key=dropout_key,
    )
    scratch, block_diagnostics = model._scratch_refine(cell_input)
    raw_delta = model.lm_head(scratch.astype(model.dtype))
    alpha = jax.nn.sigmoid(model.step_gate(scratch.astype(model.dtype)).astype(jnp.float32))
    next_belief = model._belief_update(inputs, belief_logits, raw_delta, alpha, step_index)
    return next_belief, scratch, block_diagnostics


def _brc_step_loss_weights(model: BRCModel, rollout_steps: int) -> jax.Array:
    return _normalized_step_loss_weights(model.brc.step_loss_weights, rollout_steps)


def _brc_compact_training_rollout(
    model: BRCModel,
    inputs: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
    initial_belief: jax.Array,
    dropout_key: jax.Array,
    *,
    trajectory_noise: bool,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Run BRC training without materializing [steps, batch, cells, vocab] logits."""
    base_embeddings, context = model.context_memory(inputs)
    query_mask = (~context).astype(jnp.float32)
    query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)
    loss_mask_f32 = loss_mask.astype(jnp.float32)
    loss_normalizer = jnp.maximum(jnp.sum(loss_mask_f32), 1.0)
    candidate_init = jnp.zeros_like(inputs)
    early_index = jnp.asarray(0, dtype=jnp.int32)
    mid_index = jnp.asarray(model.recurrent_steps // 2, dtype=jnp.int32)

    def scan_step(carry, scan_inputs):
        belief_logits, early_candidate, mid_candidate, trajectory_noise_count = carry
        step_index, step_dropout_key, noise_key, active_key, time_embedding = scan_inputs
        if trajectory_noise:
            noisy_belief, _noise_mode = _mix_training_belief(
                model,
                inputs,
                targets,
                belief_logits,
                noise_key,
            )
            active = targets != 0
            noise_active = (
                jax.random.uniform(active_key, (inputs.shape[0],)) < model.brc.denoise_trajectory_prob
            )
            refine_input_belief = jnp.where(noise_active[:, None, None], noisy_belief, belief_logits)
            trajectory_noise_count = trajectory_noise_count + jnp.mean(noise_active.astype(jnp.float32))
        else:
            refine_input_belief = belief_logits
            active = targets != 0
        next_belief, scratch, block_diagnostics = _refine_belief_step(
            model,
            inputs,
            refine_input_belief,
            base_embeddings,
            time_embedding,
            step_index,
            train=True,
            dropout_key=step_dropout_key,
        )
        logits = model._belief_to_token_logits(next_belief, inputs, step_index)
        token_loss = token_cross_entropy(model, logits, targets)
        per_step_loss = jnp.sum(token_loss * loss_mask_f32) / loss_normalizer
        predictions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        early_candidate = jnp.where(step_index == early_index, predictions, early_candidate)
        mid_candidate = jnp.where(step_index == mid_index, predictions, mid_candidate)

        scratch_norm = jnp.sum(
            jnp.linalg.norm(scratch.astype(jnp.float32), axis=-1) * query_mask
        ) / query_normalizer
        belief_probs = jax.nn.softmax(next_belief, axis=-1)
        confidence = jnp.max(belief_probs, axis=-1)
        filled_ratio = jnp.sum(confidence * query_mask) / query_normalizer
        (
            energy,
            kl_delta,
            entropy,
            mean_confidence,
            update_norm,
            context_weight_reg,
            context_weight_mean,
        ) = model._fixed_point_stats(
            refine_input_belief,
            next_belief,
            base_embeddings,
        )
        return (next_belief, early_candidate, mid_candidate, trajectory_noise_count), (
            per_step_loss,
            scratch_norm,
            filled_ratio,
            block_diagnostics["step_gate_mean"],
            block_diagnostics["step_gate_std"],
            energy,
            kl_delta,
            entropy,
            mean_confidence,
            update_norm,
            context_weight_reg,
            context_weight_mean,
        )

    step_indices = jnp.arange(model.recurrent_steps, dtype=jnp.int32)
    step_dropout_keys, step_noise_keys, step_active_keys = jax.random.split(dropout_key, 3)
    step_dropout_keys = jax.random.split(step_dropout_keys, model.recurrent_steps)
    step_noise_keys = jax.random.split(step_noise_keys, model.recurrent_steps)
    step_active_keys = jax.random.split(step_active_keys, model.recurrent_steps)
    time_embeddings = model.time_embed(step_indices)
    initial_carry = (initial_belief.astype(jnp.float32), candidate_init, candidate_init, jnp.asarray(0.0, dtype=jnp.float32))
    (belief_final, early_candidate, mid_candidate, trajectory_noise_count), scan_outputs = jax.lax.scan(
        scan_step,
        initial_carry,
        (step_indices, step_dropout_keys, step_noise_keys, step_active_keys, time_embeddings),
    )
    (
        per_step_loss,
        scratch_norm,
        filled_ratio,
        gate_mean,
        gate_std,
        energy,
        kl_delta,
        entropy,
        confidence,
        update_norm,
        context_weight_reg,
        context_weight_mean,
    ) = scan_outputs
    final_step = jnp.asarray(model.recurrent_steps - 1, dtype=jnp.int32)
    final_logits = model._belief_to_token_logits(belief_final, inputs, final_step)
    final_candidate = jnp.argmax(final_logits, axis=-1).astype(jnp.int32)
    diagnostics = {
        "scratch_norm": scratch_norm,
        "diffusion_filled_ratio": filled_ratio,
        "step_gate_mean": jnp.mean(gate_mean),
        "step_gate_std": jnp.mean(gate_std),
        "denoise_energy": jnp.mean(energy),
        "belief_kl_delta": jnp.mean(kl_delta),
        "belief_entropy": jnp.mean(entropy),
        "belief_confidence": jnp.mean(confidence),
        "belief_update_norm": jnp.mean(update_norm),
        "context_weight_reg": jnp.mean(context_weight_reg),
        "context_weight_mean": jnp.mean(context_weight_mean),
        "trajectory_noise_rate": trajectory_noise_count / jnp.asarray(model.recurrent_steps, dtype=jnp.float32),
        "per_step_denoise_energy": energy,
        "per_step_belief_entropy": entropy,
        "unroll_steps": jnp.asarray(model.recurrent_steps, dtype=jnp.float32),
        "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
        "belief_logits": belief_final,
        "early_candidate": early_candidate,
        "mid_candidate": mid_candidate,
        "final_candidate": final_candidate,
    }
    return final_logits, per_step_loss, diagnostics


def _trm_step_loss_weights(model: GridReasoningModel, rollout_steps: int) -> jax.Array:
    recurrent_config = getattr(model, "trm", None) or getattr(model, "urm", None)
    return _normalized_step_loss_weights(getattr(recurrent_config, "step_loss_weights", None), rollout_steps)


def _trm_selected_step(halt_logits: jax.Array) -> jax.Array:
    step_halted = halt_logits > 0.0
    step_halted = step_halted.at[-1, :].set(True)
    return jnp.argmax(step_halted.astype(jnp.int32), axis=0)


def brc_loss_and_metrics(
    model: BRCModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if dropout_key is None:
        dropout_key = jax.random.key(0)
    augment_key, solve_key, denoise_key = jax.random.split(dropout_key, 3)
    if model.config.task_type == "sudoku":
        inputs, targets = _permute_sudoku_digits(
            batch["inputs"],
            batch["labels"],
            augment_key,
            train=train,
        )
    else:
        inputs, targets = batch["inputs"], batch["labels"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, jnp.zeros_like(inputs, dtype=bool), targets), example_mask)
    brc = model.brc
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    initial_belief = model.initial_belief_logits(inputs)
    belief_init_noise_rate = zero
    belief_init_prior_rate = zero
    belief_init_teacher_rate = zero
    belief_init_self_rate = zero
    if train and brc.denoise_initial_prob > 0.0:
        denoise_predict_key, denoise_mix_key, denoise_active_key = jax.random.split(denoise_key, 3)
        self_logits, _self_diagnostics = model.run_diffusion(
            inputs,
            initial_belief=initial_belief,
            train=False,
            dropout_key=denoise_predict_key,
            return_final_only=True,
        )
        self_belief = jax.lax.stop_gradient(model._normalize_belief_logits(self_logits, inputs))
        denoise_belief, denoise_mode = _mix_training_belief(
            model,
            inputs,
            targets,
            self_belief,
            denoise_mix_key,
        )
        denoise_active = jax.random.uniform(denoise_active_key, (inputs.shape[0],)) < brc.denoise_initial_prob
        initial_belief = jnp.where(denoise_active[:, None, None], denoise_belief, initial_belief)
        denoise_active_f32 = denoise_active.astype(jnp.float32)
        belief_init_noise_rate = jnp.mean(denoise_active_f32)
        active_normalizer = jnp.maximum(jnp.sum(denoise_active_f32), 1.0)
        belief_init_prior_rate = jnp.sum(((denoise_mode == 0).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
        belief_init_teacher_rate = jnp.sum(((denoise_mode == 1).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
        belief_init_self_rate = jnp.sum(((denoise_mode == 2).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
    normalizer = jnp.maximum(jnp.sum(loss_mask.astype(jnp.float32)), 1.0)
    if train:
        final_logits, per_step_loss, diagnostics = _brc_compact_training_rollout(
            model,
            inputs,
            targets,
            loss_mask,
            initial_belief,
            solve_key,
            trajectory_noise=brc.denoise_trajectory_prob > 0.0,
        )
    else:
        step_logits, diagnostics = model.run_diffusion(
            inputs,
            initial_belief=initial_belief,
            train=train,
            dropout_key=solve_key,
        )
        step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
        step_loss_mask = loss_mask[None, :, :]
        token_loss = token_cross_entropy(model, step_logits, step_targets)
        per_step_loss = jnp.sum(token_loss * step_loss_mask.astype(jnp.float32), axis=(1, 2)) / normalizer
        final_logits = step_logits[-1]
    step_loss_weights = _brc_step_loss_weights(model, per_step_loss.shape[0])
    step_ce_loss = jnp.sum(step_loss_weights * per_step_loss)
    solution_loss = per_step_loss[-1]
    supervised_cells_per_example = jnp.sum(loss_mask.astype(jnp.float32), axis=-1)
    fixed_point_loss = brc.fixed_point_loss_weight * diagnostics["denoise_energy"]
    context_weight_reg_loss = brc.context_weight_reg_weight * diagnostics["context_weight_reg"]
    loss = step_ce_loss + fixed_point_loss + context_weight_reg_loss

    predictions = jnp.argmax(final_logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask.astype(jnp.float32)
    cell_accuracy = jnp.sum(correct) / normalizer
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask.astype(jnp.float32), axis=-1), 1.0)
    correct_per_example = jnp.sum(correct, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_f32 = exact_examples.astype(jnp.float32)
    exact_accuracy = _masked_example_mean(exact_f32, example_mask)
    exact_count = jnp.sum(exact_f32 * example_mask)
    target_probability = token_target_probability(model, final_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask.astype(jnp.float32)) / normalizer
    if train:
        oracle_step = jnp.argmin(per_step_loss)
    else:
        per_step_example_loss = jnp.sum(token_loss * step_loss_mask.astype(jnp.float32), axis=-1) / per_example_normalizer[None, :]
        oracle_step = _masked_example_mean(jnp.argmin(per_step_example_loss, axis=0).astype(jnp.float32), example_mask)
    context_consistency, invalid_rate, conflicts = _sudoku_board_metrics(model, predictions, inputs)

    metrics = {
        "loss": loss,
        "lm_loss": step_ce_loss,
        "fixed_point_loss": fixed_point_loss,
        "context_weight_reg_loss": context_weight_reg_loss,
        "final_lm_loss": solution_loss,
        "mean_lm_loss": jnp.mean(per_step_loss),
        "final_target_probability": target_probability,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "exact_count": exact_count,
        "oracle_step": oracle_step.astype(jnp.float32) + 1.0,
        "context_consistency": context_consistency,
        "invalid_rate": invalid_rate,
        "conflicts": conflicts,
        "belief_init_noise_rate": belief_init_noise_rate,
        "belief_init_prior_rate": belief_init_prior_rate,
        "belief_init_teacher_rate": belief_init_teacher_rate,
        "belief_init_self_rate": belief_init_self_rate,
        "trajectory_noise_rate": diagnostics.get("trajectory_noise_rate", zero),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_loss_weights,
        "scratch_norm": jnp.mean(diagnostics["scratch_norm"]),
        "diffusion_filled_ratio": diagnostics["diffusion_filled_ratio"],
        "step_gate_mean": diagnostics["step_gate_mean"],
        "step_gate_std": diagnostics["step_gate_std"],
        "denoise_energy": diagnostics["denoise_energy"],
        "belief_kl_delta": diagnostics["belief_kl_delta"],
        "belief_entropy": diagnostics["belief_entropy"],
        "belief_confidence": diagnostics["belief_confidence"],
        "belief_update_norm": diagnostics["belief_update_norm"],
        "context_weight_reg": diagnostics["context_weight_reg"],
        "context_weight_mean": diagnostics["context_weight_mean"],
        "per_step_denoise_energy": diagnostics["per_step_denoise_energy"],
        "per_step_belief_entropy": diagnostics["per_step_belief_entropy"],
    }
    return loss, metrics


def loss_and_metrics(
    model: GridReasoningModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if isinstance(model, BRCModel):
        del halt_loss_weight, terminal_residual_weight
        return brc_loss_and_metrics(
            model,
            batch,
            train,
            dropout_key,
        )

    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, given_mask, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)

    model_key = dropout_key

    compute_terminal_residual = terminal_residual_weight != 0.0
    model_forward_kwargs = {}
    if isinstance(model, (TinyRecursiveModel, UnifiedReasoningModel)):
        model_forward_kwargs = {
            "puzzle_identifiers": batch["puzzle_identifiers"],
        }
        if isinstance(model, TinyRecursiveModel):
            model_forward_kwargs["compute_terminal_residual"] = compute_terminal_residual
    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=train,
        dropout_key=model_key,
        **model_forward_kwargs,
    )
    effective_step_logits = step_logits
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = token_cross_entropy(model, effective_step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    rollout_steps = effective_step_logits.shape[0]
    step_weights = _trm_step_loss_weights(model, rollout_steps)
    lm_loss_value = jnp.sum(step_weights * per_step_loss)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    step_predictions = jnp.argmax(effective_step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        supervised_cells_per_example[None, :] > 0,
        step_correct_per_example == supervised_cells_per_example[None, :],
        True,
    )
    halt_logits = diagnostics.get("halt_logits")
    halt_loss = jnp.asarray(0.0, dtype=jnp.float32)
    selected_logits = effective_step_logits[-1]
    selected_step = jnp.full((inputs.shape[0],), effective_step_logits.shape[0] - 1, dtype=jnp.int32)
    if halt_logits is not None and halt_loss_weight != 0.0:
        halt_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        halt_loss_per_example = optax.sigmoid_binary_cross_entropy(halt_logits, halt_targets)
        halt_loss = jnp.sum(halt_loss_per_example * example_mask[None, :]) / jnp.maximum(
            jnp.sum(example_mask) * halt_logits.shape[0],
            1.0,
        )
        selected_step = _trm_selected_step(halt_logits)
        gather_index = selected_step[None, :, None, None]
        selected_logits = jnp.take_along_axis(
            effective_step_logits,
            jnp.broadcast_to(gather_index, (1, inputs.shape[0], effective_step_logits.shape[2], effective_step_logits.shape[3])),
            axis=0,
        ).squeeze(0)
    terminal_residual = diagnostics.get(
        "terminal_belief_mse",
        diagnostics.get("terminal_belief_delta", jnp.asarray(0.0, dtype=jnp.float32)),
    )
    loss = (
        lm_loss_value
        + halt_loss_weight * halt_loss
        + terminal_residual_weight * terminal_residual
    )

    final_logits = effective_step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    target_probability = token_target_probability(model, final_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    cell_accuracy = jnp.sum(correct) / metric_normalizer

    correct_per_example = jnp.sum(correct, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_f32 = exact_examples.astype(jnp.float32)
    exact_accuracy = _masked_example_mean(exact_f32, example_mask)
    exact_count = jnp.sum(exact_f32 * example_mask)
    if halt_loss_weight != 0.0:
        selected_token_loss = token_cross_entropy(model, selected_logits, targets)
        selected_ce_loss = jnp.sum(selected_token_loss * loss_mask) / normalizer
        selected_predictions = jnp.argmax(selected_logits, axis=-1)
        selected_correct = (selected_predictions == targets).astype(jnp.float32) * loss_mask
        selected_accuracy = jnp.sum(selected_correct) / metric_normalizer
        selected_correct_per_example = jnp.sum(selected_correct, axis=-1)
        selected_exact_examples = jnp.where(
            supervised_cells_per_example > 0,
            selected_correct_per_example == supervised_cells_per_example,
            True,
        )
        selected_exact_accuracy = _masked_example_mean(selected_exact_examples.astype(jnp.float32), example_mask)
    else:
        selected_ce_loss = per_step_loss[-1]
        selected_accuracy = cell_accuracy
        selected_exact_accuracy = exact_accuracy
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)

    metrics = {
        "loss": loss,
        "lm_loss": lm_loss_value,
        "halt_loss": halt_loss,
        "final_lm_loss": per_step_loss[-1],
        "final_target_probability": target_probability,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "exact_count": exact_count,
        "selected_lm_loss": selected_ce_loss,
        "selected_accuracy": selected_accuracy,
        "selected_exact_accuracy": selected_exact_accuracy,
        "selected_step": _masked_example_mean(selected_step.astype(jnp.float32) + 1.0, example_mask),
        "oracle_step": _masked_example_mean(oracle_step.astype(jnp.float32) + 1.0, example_mask),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    metrics.update(_maybe_path_metrics(model, predictions, targets, loss_mask))
    if halt_loss_weight != 0.0:
        halt_path_metrics = _maybe_path_metrics(model, selected_predictions, targets, loss_mask)
        metrics.update(
            {
                f"selected_{key}": value
                for key, value in halt_path_metrics.items()
                if key in ("path_precision", "path_recall", "path_f1")
            }
        )
    if halt_logits is not None and halt_loss_weight != 0.0:
        metrics["per_step_halt_probability"] = (
            jnp.sum(jax.nn.sigmoid(halt_logits) * example_mask[None, :], axis=1)
            / jnp.maximum(jnp.sum(example_mask), 1.0)
        )
    if "terminal_belief_delta" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_delta"] = diagnostics.get("terminal_belief_delta", terminal_residual)
    if "terminal_belief_mse" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_mse"] = terminal_residual
    if "rho_mean" in diagnostics:
        metrics["per_step_rho"] = diagnostics["rho_mean"]
    if "alpha_mean" in diagnostics:
        metrics["per_step_alpha"] = diagnostics["alpha_mean"]
    for key in (
        "unroll_steps",
        "terminal_belief_delta",
        "terminal_belief_mse",
        "belief_entropy",
        "belief_confidence",
        "halt_loss",
        "selected_lm_loss",
        "selected_accuracy",
        "selected_exact_accuracy",
        "selected_step",
        "oracle_step",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    return loss, metrics


def trm_act_loss_and_metrics(
    model: TinyRecursiveModel,
    carry: dict[str, jax.Array],
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    halt_loss_weight: float = 0.5,
    puzzle_embeddings: jax.Array | None = None,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    new_carry, logits, diagnostics = model.forward_act_step(
        carry,
        batch,
        train=train,
        dropout_key=dropout_key,
        puzzle_embeddings=puzzle_embeddings,
    )
    targets = new_carry["current_labels"]
    given_mask = new_carry["current_given_mask"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, given_mask, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
    token_loss = token_cross_entropy(model, logits, targets)
    per_example_loss = jnp.sum(token_loss * loss_mask, axis=-1) / per_example_normalizer
    lm_loss_value = jnp.mean(per_example_loss)

    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    current_cell_accuracy = jnp.sum(correct) / metric_normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_targets = jax.lax.stop_gradient(exact_examples.astype(jnp.float32))
    current_exact_accuracy = jnp.mean(exact_targets)
    current_exact_count = jnp.sum(exact_targets)
    valid_halted = new_carry["halted"] & (supervised_cells_per_example > 0)
    valid_halted_f32 = valid_halted.astype(jnp.float32)
    valid_halted_count = jnp.sum(valid_halted_f32)
    valid_halted_normalizer = jnp.maximum(valid_halted_count, 1.0)
    per_example_accuracy = correct_per_example / jnp.maximum(supervised_cells_per_example, 1.0)
    cell_accuracy = jnp.sum(per_example_accuracy * valid_halted_f32) / valid_halted_normalizer
    exact_accuracy = jnp.sum(exact_targets * valid_halted_f32) / valid_halted_normalizer
    exact_count = jnp.sum(exact_targets * valid_halted_f32)
    q_halt_correct = ((diagnostics["halt_logits"] >= 0.0) == exact_examples) & valid_halted
    q_halt_accuracy = jnp.sum(q_halt_correct.astype(jnp.float32)) / valid_halted_normalizer
    steps = jnp.where(valid_halted_count > 0.0, diagnostics["act_step"], 0.0)

    halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(diagnostics["halt_logits"], exact_targets))
    loss = lm_loss_value + halt_loss_weight * halt_loss

    target_probability = token_target_probability(model, logits, targets)
    per_example_target_probability = (
        jnp.sum(target_probability * loss_mask, axis=-1)
        / jnp.maximum(supervised_cells_per_example, 1.0)
    )
    target_probability = (
        jnp.sum(per_example_target_probability * valid_halted_f32)
        / valid_halted_normalizer
    )
    metrics = {
        "loss": loss,
        "lm_loss": lm_loss_value,
        "q_halt_loss": halt_loss,
        "count": valid_halted_count,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "q_halt_accuracy": q_halt_accuracy,
        "steps": steps,
        "final_lm_loss": lm_loss_value,
        "halted_target_probability": target_probability,
        "current_accuracy": current_cell_accuracy,
        "current_exact_accuracy": current_exact_accuracy,
        "current_exact_count": current_exact_count,
        "halt_loss": halt_loss,
        "act_step": diagnostics["act_step"],
        "halted_rate": diagnostics["halted_rate"],
        "reset_rate": diagnostics["reset_rate"],
    }
    metrics.update(_maybe_path_metrics(model, predictions, targets, loss_mask))
    return loss, (metrics, new_carry)


def trm_dense_unroll_loss_and_metrics(
    model: TinyRecursiveModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    *,
    halt_loss_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, given_mask, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    model_forward_kwargs = {"puzzle_identifiers": batch["puzzle_identifiers"]}
    if isinstance(model, TinyRecursiveModel):
        model_forward_kwargs["compute_terminal_residual"] = False
    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=train,
        dropout_key=dropout_key,
        **model_forward_kwargs,
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = token_cross_entropy(model, step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    rollout_steps = per_step_loss.shape[0]
    step_weights = _trm_step_loss_weights(model, rollout_steps)
    lm_loss_value = jnp.sum(step_weights * per_step_loss)
    mean_lm_loss_value = jnp.mean(per_step_loss)
    final_lm_loss_value = per_step_loss[-1]
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / metric_normalizer
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        supervised_cells_per_example[None, :] > 0,
        step_correct_per_example == supervised_cells_per_example[None, :],
        True,
    )
    final_logits = step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    target_probability = token_target_probability(model, final_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    cell_accuracy = jnp.sum(correct) / metric_normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_f32 = exact_examples.astype(jnp.float32)
    exact_accuracy = _masked_example_mean(exact_f32, example_mask)
    exact_count = jnp.sum(exact_f32 * example_mask)

    halt_logits = diagnostics.get("halt_logits")
    halt_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if halt_logits is not None and halt_loss_weight != 0.0:
        halt_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        halt_loss_per_example = optax.sigmoid_binary_cross_entropy(halt_logits, halt_targets)
        halt_loss = jnp.sum(halt_loss_per_example * example_mask[None, :]) / jnp.maximum(
            jnp.sum(example_mask) * halt_logits.shape[0],
            1.0,
        )

    loss = lm_loss_value + halt_loss_weight * halt_loss
    oracle_step = jnp.argmin(
        jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :],
        axis=0,
    )
    metrics = {
        "loss": loss,
        "lm_loss": lm_loss_value,
        "mean_lm_loss": mean_lm_loss_value,
        "final_lm_loss": final_lm_loss_value,
        "final_target_probability": target_probability,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "exact_count": exact_count,
        "halt_loss": halt_loss,
        "oracle_step": _masked_example_mean(oracle_step.astype(jnp.float32) + 1.0, example_mask),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.float32),
    }
    metrics.update(_maybe_path_metrics(model, predictions, targets, loss_mask))
    if halt_logits is not None and halt_loss_weight != 0.0:
        metrics["per_step_halt_probability"] = (
            jnp.sum(jax.nn.sigmoid(halt_logits) * example_mask[None, :], axis=1)
            / jnp.maximum(jnp.sum(example_mask), 1.0)
        )
    return loss, metrics


def trm_eval_loss_and_metrics(
    model: TinyRecursiveModel,
    batch: dict[str, jax.Array],
    halt_loss_weight: float = 0.5,
    *,
    collect_diagnostics: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, given_mask, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    model_forward_kwargs = {"puzzle_identifiers": batch["puzzle_identifiers"]}
    if isinstance(model, TinyRecursiveModel):
        model_forward_kwargs["compute_terminal_residual"] = False
        model_forward_kwargs["collect_diagnostics"] = collect_diagnostics
    elif isinstance(model, UnifiedReasoningModel):
        model_forward_kwargs["collect_diagnostics"] = collect_diagnostics
    step_logits, model_diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=False,
        dropout_key=None,
        **model_forward_kwargs,
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = token_cross_entropy(model, step_logits, step_targets)

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    per_step_example_solved = jnp.where(
        supervised_cells_per_example[None, :] > 0,
        step_correct_per_example == supervised_cells_per_example[None, :],
        True,
    )
    halt_logits = model_diagnostics["halt_logits"]
    selected_step = _trm_selected_step(halt_logits)
    gather_index = selected_step[None, :, None, None]
    selected_logits = jnp.take_along_axis(
        step_logits,
        jnp.broadcast_to(gather_index, (1, inputs.shape[0], step_logits.shape[2], step_logits.shape[3])),
        axis=0,
    ).squeeze(0)
    selected_token_loss = token_cross_entropy(model, selected_logits, targets)
    selected_lm_loss_value = jnp.sum(selected_token_loss * loss_mask) / normalizer
    selected_exact_targets = jnp.take_along_axis(
        per_step_example_solved.astype(jnp.float32),
        selected_step[None, :],
        axis=0,
    ).squeeze(0)
    selected_exact_count = jnp.sum(selected_exact_targets * example_mask)
    halt_loss_per_example = optax.sigmoid_binary_cross_entropy(
        halt_logits,
        jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32)),
    )
    halt_loss = jnp.sum(halt_loss_per_example * example_mask[None, :]) / jnp.maximum(
        jnp.sum(example_mask) * halt_logits.shape[0],
        1.0,
    )
    final_logits = step_logits[-1]
    final_token_loss = token_cross_entropy(model, final_logits, targets)
    final_lm_loss_value = jnp.sum(final_token_loss * loss_mask) / normalizer
    loss = final_lm_loss_value + halt_loss_weight * halt_loss

    predictions = jnp.argmax(selected_logits, axis=-1)
    target_probability = token_target_probability(model, selected_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    cell_accuracy = jnp.sum(correct) / metric_normalizer
    exact_accuracy = _masked_example_mean(selected_exact_targets, example_mask)
    selected_q_halt_logits = jnp.take_along_axis(halt_logits, selected_step[None, :], axis=0).squeeze(0)
    selected_q_halt_correct = (selected_q_halt_logits >= 0.0) == selected_exact_targets.astype(bool)
    q_halt_accuracy = _masked_example_mean(selected_q_halt_correct.astype(jnp.float32), example_mask)
    final_predictions = jnp.argmax(final_logits, axis=-1)
    final_correct = (final_predictions == targets).astype(jnp.float32) * loss_mask
    final_cell_accuracy = jnp.sum(final_correct) / metric_normalizer
    final_correct_per_example = jnp.sum(final_correct, axis=-1)
    final_exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        final_correct_per_example == supervised_cells_per_example,
        True,
    )
    final_exact_f32 = final_exact_examples.astype(jnp.float32)
    final_exact_count = jnp.sum(final_exact_f32 * example_mask)
    count = jnp.sum(example_mask)
    metrics = {
        "loss": loss,
        "lm_loss": final_lm_loss_value,
        "q_halt_loss": halt_loss,
        "count": count,
        "accuracy": final_cell_accuracy,
        "exact_accuracy": _masked_example_mean(final_exact_f32, example_mask),
        "q_halt_accuracy": q_halt_accuracy,
        "steps": jnp.asarray(step_logits.shape[0], dtype=jnp.float32),
        "halt_loss": halt_loss,
        "final_lm_loss": final_lm_loss_value,
        "final_accuracy": final_cell_accuracy,
        "final_exact_accuracy": _masked_example_mean(final_exact_f32, example_mask),
        "halted_target_probability": target_probability,
        "exact_count": final_exact_count,
        "selected_lm_loss": selected_lm_loss_value,
        "selected_accuracy": cell_accuracy,
        "selected_exact_accuracy": exact_accuracy,
        "selected_exact_count": selected_exact_count,
        "selected_step": _masked_example_mean(selected_step.astype(jnp.float32) + 1.0, example_mask),
        "final_exact_count": final_exact_count,
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.int32),
    }
    if collect_diagnostics:
        per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
        per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]
        per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / metric_normalizer
        oracle_step = jnp.argmin(per_step_example_loss, axis=0)
        metrics.update(
            {
                "oracle_step": _masked_example_mean(oracle_step.astype(jnp.float32) + 1.0, example_mask),
                "per_step_loss": per_step_loss,
                "per_step_accuracy": per_step_accuracy,
                "per_step_hidden_delta": model_diagnostics["hidden_delta_mean"],
                "per_step_halt_probability": (
                    jnp.sum(jax.nn.sigmoid(halt_logits) * example_mask[None, :], axis=1)
                    / jnp.maximum(count, 1.0)
                ),
            }
        )
    selected_path_metrics = _maybe_path_metrics(model, predictions, targets, loss_mask)
    metrics.update(selected_path_metrics)
    metrics.update(
        {
            f"selected_{key}": value
            for key, value in selected_path_metrics.items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    final_path_metrics = _maybe_path_metrics(model, final_predictions, targets, loss_mask)
    metrics.update(
        {
            f"final_{key}": value
            for key, value in final_path_metrics.items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    return loss, metrics

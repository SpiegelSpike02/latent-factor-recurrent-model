from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training import orbax_utils
from orbax.checkpoint import (
    Checkpointer,
    PyTreeCheckpointHandler,
    args as ocp_args,
)

from lfrm.config import ExperimentConfig
from lfrm.models import BRCSudokuModel, TinyRecursiveModel


GridReasoningModel = BRCSudokuModel | TinyRecursiveModel


def build_optimizer(config: ExperimentConfig) -> optax.GradientTransformation:
    optimizer_steps = max(1, config.train.max_steps)
    warmup_steps = max(1, config.optimizer.warmup_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(optimizer_steps, warmup_steps + 1),
        end_value=config.optimizer.learning_rate * config.optimizer.lr_min_ratio,
    )
    optimizer_schedule = lambda count: schedule(count + jnp.asarray(1, dtype=count.dtype))
    transforms = []
    if config.optimizer.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.optimizer.grad_clip_norm))
    transforms.append(
        optax.adamw(
            learning_rate=optimizer_schedule,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            weight_decay=config.optimizer.weight_decay,
        )
    )
    optimizer = optax.chain(*transforms)
    if config.optimizer.flatten_optimizer:
        optimizer = optax.flatten(optimizer)
    return optimizer


def create_model(config: ExperimentConfig) -> GridReasoningModel:
    if config.model.model_type == "trm":
        return TinyRecursiveModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if config.model.model_type == "brc_sudoku":
        return BRCSudokuModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    raise ValueError("Only model_type='trm' or model_type='brc_sudoku' is supported")


def create_optimizer(model: GridReasoningModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config), wrt=nnx.Param)


def create_ema_model(model: GridReasoningModel, config: ExperimentConfig) -> GridReasoningModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model))
    return ema_model


def ema_param_filter(config: ExperimentConfig):
    return nnx.Param


def _masked_token_ce(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f32 = mask.astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(mask_f32), 1.0)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(token_loss * mask_f32) / normalizer


def _target_loss_weights(model: GridReasoningModel, targets: jax.Array) -> jax.Array:
    path_weight = getattr(getattr(model, "config", None), "path_loss_weight", 1.0)
    return jnp.where(targets == 5, jnp.asarray(path_weight, dtype=jnp.float32), 1.0)


def _weighted_mask_normalizer(mask: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.maximum(jnp.sum(mask.astype(jnp.float32) * weights.astype(jnp.float32)), 1.0)


def _path_metrics(predictions: jax.Array, targets: jax.Array, loss_mask: jax.Array) -> dict[str, jax.Array]:
    mask = loss_mask.astype(bool)
    predicted_path = (predictions == 5) & mask
    target_path = (targets == 5) & mask
    true_positive = jnp.sum((predicted_path & target_path).astype(jnp.float32))
    predicted_positive = jnp.sum(predicted_path.astype(jnp.float32))
    target_positive = jnp.sum(target_path.astype(jnp.float32))
    mask_total = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    precision = true_positive / jnp.maximum(predicted_positive, 1.0)
    recall = true_positive / jnp.maximum(target_positive, 1.0)
    f1 = jnp.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )
    return {
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": f1,
        "path_positive_rate": predicted_positive / mask_total,
        "target_path_rate": target_positive / mask_total,
    }


def _clamp_logits_to_given(
    logits: jax.Array,
    inputs: jax.Array,
    given_mask: jax.Array,
    vocab_size: int,
) -> jax.Array:
    given_logits = jnp.full_like(logits, -1.0e4)
    given_logits = given_logits + 2.0e4 * jax.nn.one_hot(inputs.astype(jnp.int32), vocab_size)
    while given_mask.ndim < logits.ndim - 1:
        given_mask = given_mask[None, ...]
    return jnp.where(given_mask[..., None], given_logits, logits)


def _should_clamp_given(model: GridReasoningModel) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "clamp_given", False))


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
    model: BRCSudokuModel,
    inputs: jax.Array,
    targets: jax.Array,
    given_mask: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mode_key, mask_key, corrupt_key, soft_candidate_key, soft_strength_key = jax.random.split(key, 5)
    brc = model.brc
    mode_weights = jnp.asarray(brc.denoise_mode_weights, dtype=jnp.float32)
    mode_logits = jnp.log(mode_weights / jnp.maximum(jnp.sum(mode_weights), 1e-6))
    mode = jax.random.categorical(mode_key, mode_logits, shape=(inputs.shape[0],))

    unknown = ~given_mask
    teacher_reveal = unknown & (jax.random.uniform(mask_key, inputs.shape) < brc.denoise_teacher_reveal_prob)
    full_mask_belief = model.initial_belief_logits(inputs)
    teacher_belief = model.belief_logits_from_tokens(inputs, targets, teacher_reveal)

    corrupt_candidate = _corrupt_solution(model, inputs, targets, given_mask, corrupt_key)
    corrupt_belief = model.belief_logits_from_tokens(inputs, corrupt_candidate, unknown)

    soft_candidate = _corrupt_solution(
        model,
        inputs,
        targets,
        given_mask,
        soft_candidate_key,
        corruption_prob=0.20,
    )
    soft_digit_ids = jnp.clip(soft_candidate - 2, 0, 8)
    soft_strength = jax.random.uniform(
        soft_strength_key,
        (*inputs.shape, 1),
        minval=1.0,
        maxval=4.0,
        dtype=jnp.float32,
    )
    soft_belief = soft_strength * jax.nn.one_hot(soft_digit_ids.astype(jnp.int32), 9)
    soft_belief = model._clamp_belief_logits(jnp.where(unknown[..., None], soft_belief, 0.0), inputs)

    stacked = jnp.stack([full_mask_belief, teacher_belief, corrupt_belief, soft_belief], axis=0)
    selector = jax.nn.one_hot(mode, 4, dtype=jnp.float32).T[:, :, None, None]
    mixed_belief = jnp.sum(stacked * selector, axis=0)
    mixed_belief = jnp.where(unknown[..., None], mixed_belief, full_mask_belief)
    return mixed_belief.astype(jnp.float32), mode


def _corrupt_solution(
    model: BRCSudokuModel,
    inputs: jax.Array,
    targets: jax.Array,
    given_mask: jax.Array,
    key: jax.Array,
    *,
    corruption_prob: float = 0.08,
) -> jax.Array:
    select_key, digit_key, force_key = jax.random.split(key, 3)
    random_digits = jax.random.randint(
        digit_key,
        targets.shape,
        minval=2,
        maxval=min(model.config.vocab_size, 11),
        dtype=jnp.int32,
    )
    change = (~given_mask) & (jax.random.uniform(select_key, targets.shape) < corruption_prob)
    unknown_scores = jnp.where(~given_mask, jax.random.uniform(force_key, targets.shape), -jnp.inf)
    forced_index = jnp.argmax(unknown_scores, axis=-1)
    forced = jnp.zeros_like(given_mask).at[jnp.arange(targets.shape[0]), forced_index].set(True)
    change = change | (forced & ~given_mask)
    corrupted = jnp.where(change, random_digits, targets)
    return jnp.where(given_mask, inputs, corrupted).astype(jnp.int32)


def _zero_brc_fit_metrics(zero: jax.Array) -> dict[str, jax.Array]:
    return {
        "latent_fit_loss": zero,
        "fit_given_loss": zero,
        "fit_energy": zero,
        "fit_consistency_loss": zero,
        "fit_prior_loss": zero,
        "latent_update_norm": zero,
        "latent_grad_norm": zero,
        "latent_step_norm": zero,
    }


def _sudoku_board_metrics(
    model: BRCSudokuModel,
    predictions: jax.Array,
    inputs: jax.Array,
    given_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    given_f32 = given_mask.astype(jnp.float32)
    given_consistency = (
        jnp.sum(((predictions == inputs) & given_mask).astype(jnp.float32))
        / jnp.maximum(jnp.sum(given_f32), 1.0)
    )
    if model.config.grid_height != 9 or model.config.grid_width != 9 or model.config.vocab_size < 11:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return given_consistency, zero, zero

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
    conflict_count = jnp.mean(row_conflicts + col_conflicts + box_conflicts)
    invalid_rate = 1.0 - jnp.mean(unit_valid.astype(jnp.float32))
    return given_consistency, invalid_rate, conflict_count


def _verifier_guided_latent_fit(
    model: BRCSudokuModel,
    inputs: jax.Array,
    given_mask: jax.Array,
    z0: jax.Array,
    *,
    train: bool,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    brc = model.brc
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if brc.latent_fit_steps <= 0:
        return z0, _zero_brc_fit_metrics(zero)

    step_keys = jax.random.split(key, (brc.latent_fit_steps, 2))

    def fit_components(latent_z: jax.Array, step_key_pair: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        key_a, key_b = step_key_pair
        logits_a, diagnostics_a = model.run_diffusion(
            inputs,
            z=latent_z,
            initial_belief=model.initial_belief_logits(inputs),
            train=train,
            dropout_key=key_a,
            return_raw_final_logits=True,
            return_final_only=True,
        )
        raw_final_logits_a = diagnostics_a["raw_final_logits"]
        probs_a = jax.nn.softmax(logits_a, axis=-1)
        given_loss = _masked_token_ce(raw_final_logits_a, inputs, given_mask)
        energy = jnp.mean(model.verifier_energy_from_probs(inputs, probs_a))

        final_logits_b, _ = model.run_diffusion(
            inputs,
            z=latent_z,
            initial_belief=model.initial_belief_logits(inputs),
            train=train,
            dropout_key=key_b,
            return_final_only=True,
        )
        probs_b = jax.nn.softmax(final_logits_b, axis=-1)
        unknown = (~given_mask).astype(jnp.float32)
        consistency = jnp.sum(jnp.square(probs_a - probs_b) * unknown[..., None])
        consistency = consistency / jnp.maximum(jnp.sum(unknown) * probs_a.shape[-1], 1.0)
        prior = jnp.mean(jnp.square(latent_z - z0))
        fit_loss = (
            brc.fit_given_weight * given_loss
            + brc.fit_energy_weight * energy
            + brc.fit_consistency_weight * consistency
            + brc.fit_prior_weight * prior
        )
        return fit_loss, {
            "fit_given_loss": given_loss,
            "fit_energy": energy,
            "fit_consistency_loss": consistency,
            "fit_prior_loss": prior,
        }

    def inner_step(latent_z: jax.Array, step_key_pair: jax.Array):
        (fit_loss, components), fit_grad = jax.value_and_grad(fit_components, has_aux=True)(
            latent_z,
            step_key_pair,
        )
        grad_norm = jnp.linalg.norm(fit_grad.astype(jnp.float32), axis=-1, keepdims=True)
        grad_scale = jnp.minimum(1.0, brc.latent_grad_clip_norm / (grad_norm + 1e-6))
        clipped_grad = fit_grad * grad_scale
        step_update = -brc.latent_lr * clipped_grad
        step_norm = jnp.linalg.norm(step_update.astype(jnp.float32), axis=-1, keepdims=True)
        step_scale = jnp.minimum(1.0, brc.latent_update_clip_norm / (step_norm + 1e-6))
        step_update = step_update * step_scale
        next_z = latent_z + step_update
        total_delta = next_z - z0
        total_norm = jnp.linalg.norm(total_delta.astype(jnp.float32), axis=-1, keepdims=True)
        total_scale = jnp.minimum(1.0, brc.latent_update_clip_norm / (total_norm + 1e-6))
        next_z = z0 + total_delta * total_scale
        return next_z, (
            fit_loss,
            components["fit_given_loss"],
            components["fit_energy"],
            components["fit_consistency_loss"],
            components["fit_prior_loss"],
            jnp.mean(grad_norm),
            jnp.mean(step_norm),
        )

    z, (fit_loss, fit_given, fit_energy, fit_consistency, fit_prior, grad_norm, step_norm) = jax.lax.scan(
        inner_step,
        z0,
        step_keys,
    )
    update_norm = jnp.mean(jnp.linalg.norm((z - z0).astype(jnp.float32), axis=-1))
    return z, {
        "latent_fit_loss": jnp.mean(fit_loss),
        "fit_given_loss": jnp.mean(fit_given),
        "fit_energy": jnp.mean(fit_energy),
        "fit_consistency_loss": jnp.mean(fit_consistency),
        "fit_prior_loss": jnp.mean(fit_prior),
        "latent_update_norm": update_norm,
        "latent_grad_norm": jnp.mean(grad_norm),
        "latent_step_norm": jnp.mean(step_norm),
    }


def _normalized_step_loss_weights(configured: tuple[float, ...] | None, num_steps: int) -> jax.Array:
    if configured is None:
        weights = jnp.arange(1, num_steps + 1, dtype=jnp.float32)
    else:
        weights = jnp.asarray(configured, dtype=jnp.float32)
    return weights / jnp.maximum(jnp.sum(weights), 1e-6)


def _brc_step_loss_weights(model: BRCSudokuModel, num_steps: int) -> jax.Array:
    return _normalized_step_loss_weights(model.brc.step_loss_weights, num_steps)


def _brc_compact_training_rollout(
    model: BRCSudokuModel,
    inputs: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
    z: jax.Array,
    initial_belief: jax.Array,
    dropout_key: jax.Array,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Run BRC training without materializing [steps, batch, cells, vocab] logits."""
    h, base_embeddings, given = model.initial_recurrent_state(inputs, z, initial_belief)
    unknown = (~given).astype(jnp.float32)
    unknown_normalizer = jnp.maximum(jnp.sum(unknown), 1.0)
    loss_mask_f32 = loss_mask.astype(jnp.float32)
    loss_normalizer = jnp.maximum(jnp.sum(loss_mask_f32), 1.0)
    candidate_init = jnp.zeros_like(inputs)
    early_index = jnp.asarray(0, dtype=jnp.int32)
    mid_index = jnp.asarray(model.recursion_steps // 2, dtype=jnp.int32)

    def scan_step(carry, scan_inputs):
        h_prev, belief_logits, early_candidate, mid_candidate = carry
        step_index, step_dropout_key, time_embedding = scan_inputs
        cell_input = model._cell_embeddings(
            inputs,
            belief_logits,
            base_embeddings,
            time_embedding,
            train=True,
            dropout_key=step_dropout_key,
        )
        h_next, block_diagnostics = model._solver_update(h_prev, cell_input, z)
        raw_logits = model.lm_head(h_next.astype(model.dtype))
        next_belief = model._belief_update(inputs, belief_logits, raw_logits, step_index)
        logits = model._belief_to_token_logits(next_belief, inputs, step_index)
        token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        per_step_loss = jnp.sum(token_loss * loss_mask_f32) / loss_normalizer
        predictions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        early_candidate = jnp.where(step_index == early_index, predictions, early_candidate)
        mid_candidate = jnp.where(step_index == mid_index, predictions, mid_candidate)

        hidden_delta = jnp.linalg.norm((h_next - h_prev).astype(jnp.float32), axis=-1)
        hidden_delta = jnp.sum(hidden_delta * unknown) / unknown_normalizer
        belief_probs = jax.nn.softmax(next_belief, axis=-1)
        confidence = jnp.max(belief_probs, axis=-1)
        filled_ratio = jnp.sum(confidence * unknown) / unknown_normalizer
        return (h_next, next_belief, early_candidate, mid_candidate), (
            per_step_loss,
            hidden_delta,
            filled_ratio,
            block_diagnostics["brc_gate_mean"],
            block_diagnostics["brc_gate_std"],
        )

    step_indices = jnp.arange(model.recursion_steps, dtype=jnp.int32)
    step_dropout_keys = jax.random.split(dropout_key, model.recursion_steps)
    time_embeddings = model.time_embed(step_indices)
    initial_carry = (h, initial_belief.astype(jnp.float32), candidate_init, candidate_init)
    (h_final, belief_final, early_candidate, mid_candidate), scan_outputs = jax.lax.scan(
        scan_step,
        initial_carry,
        (step_indices, step_dropout_keys, time_embeddings),
    )
    per_step_loss, hidden_delta, filled_ratio, gate_mean, gate_std = scan_outputs
    final_step = jnp.asarray(model.recursion_steps - 1, dtype=jnp.int32)
    final_logits = model._belief_to_token_logits(belief_final, inputs, final_step)
    final_candidate = jnp.argmax(final_logits, axis=-1).astype(jnp.int32)
    diagnostics = {
        "hidden_delta_mean": hidden_delta,
        "diffusion_filled_ratio": filled_ratio,
        "brc_gate_mean": jnp.mean(gate_mean),
        "brc_gate_std": jnp.mean(gate_std),
        "unroll_steps": jnp.asarray(model.recursion_steps, dtype=jnp.float32),
        "z": z,
        "h": h_final,
        "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
        "belief_logits": belief_final,
        "early_candidate": early_candidate,
        "mid_candidate": mid_candidate,
        "final_candidate": final_candidate,
    }
    return final_logits, per_step_loss, diagnostics


def _trm_step_loss_weights(model: GridReasoningModel, num_steps: int) -> jax.Array:
    trm_config = getattr(model, "trm", None)
    return _normalized_step_loss_weights(getattr(trm_config, "step_loss_weights", None), num_steps)


def _trm_halt_selected_step(halt_logits: jax.Array) -> jax.Array:
    step_halted = halt_logits > 0.0
    step_halted = step_halted.at[-1, :].set(True)
    return jnp.argmax(step_halted.astype(jnp.int32), axis=0)


def brc_loss_and_metrics(
    model: BRCSudokuModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if dropout_key is None:
        dropout_key = jax.random.key(0)
    augment_key, solve_key, denoise_key, verifier_key, fit_key, meta_key = jax.random.split(dropout_key, 6)
    inputs, targets = _permute_sudoku_digits(
        batch["inputs"],
        batch["labels"],
        augment_key,
        train=train,
    )
    given_mask = inputs != 1
    loss_mask = ~given_mask
    brc = model.brc
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    z0 = model.infer_latent(inputs)
    initial_belief = model.initial_belief_logits(inputs)
    belief_init_noise_rate = zero
    belief_init_uniform_rate = zero
    belief_init_teacher_rate = zero
    belief_init_corrupt_rate = zero
    belief_init_soft_rate = zero
    if train and brc.denoise_initial_prob > 0.0:
        denoise_mix_key, denoise_active_key = jax.random.split(denoise_key, 2)
        denoise_belief, denoise_mode = _mix_training_belief(
            model,
            inputs,
            targets,
            given_mask,
            denoise_mix_key,
        )
        denoise_active = jax.random.uniform(denoise_active_key, (inputs.shape[0],)) < brc.denoise_initial_prob
        initial_belief = jnp.where(denoise_active[:, None, None], denoise_belief, initial_belief)
        denoise_active_f32 = denoise_active.astype(jnp.float32)
        belief_init_noise_rate = jnp.mean(denoise_active_f32)
        active_normalizer = jnp.maximum(jnp.sum(denoise_active_f32), 1.0)
        belief_init_uniform_rate = jnp.sum(((denoise_mode == 0).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
        belief_init_teacher_rate = jnp.sum(((denoise_mode == 1).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
        belief_init_corrupt_rate = jnp.sum(((denoise_mode == 2).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
        belief_init_soft_rate = jnp.sum(((denoise_mode == 3).astype(jnp.float32)) * denoise_active_f32) / active_normalizer
    normalizer = jnp.maximum(jnp.sum(loss_mask.astype(jnp.float32)), 1.0)
    if train:
        final_logits, per_step_loss, diagnostics = _brc_compact_training_rollout(
            model,
            inputs,
            targets,
            loss_mask,
            z0,
            initial_belief,
            solve_key,
        )
    else:
        step_logits, diagnostics = model.run_diffusion(
            inputs,
            z=z0,
            initial_belief=initial_belief,
            train=train,
            dropout_key=solve_key,
        )
        step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
        step_loss_mask = loss_mask[None, :, :]
        token_loss = optax.softmax_cross_entropy_with_integer_labels(step_logits, step_targets)
        per_step_loss = jnp.sum(token_loss * step_loss_mask.astype(jnp.float32), axis=(1, 2)) / normalizer
        final_logits = step_logits[-1]
    step_loss_weights = _brc_step_loss_weights(model, per_step_loss.shape[0])
    step_ce_loss = jnp.sum(step_loss_weights * per_step_loss)
    solution_loss = per_step_loss[-1]
    blanks_per_example = jnp.sum(loss_mask.astype(jnp.float32), axis=-1)

    true_energy = zero
    fake_energy = zero
    verifier_loss = zero
    verifier_ranking_accuracy = zero
    if brc.verifier_loss_weight != 0.0:
        random_fake = _corrupt_solution(model, inputs, targets, given_mask, verifier_key)
        if train:
            early_fake = jax.lax.stop_gradient(diagnostics["early_candidate"])
            mid_fake = jax.lax.stop_gradient(diagnostics["mid_candidate"])
            final_fake = jax.lax.stop_gradient(diagnostics["final_candidate"])
        else:
            early_fake = jax.lax.stop_gradient(jnp.argmax(step_logits[0], axis=-1).astype(jnp.int32))
            mid_fake = jax.lax.stop_gradient(jnp.argmax(step_logits[step_logits.shape[0] // 2], axis=-1).astype(jnp.int32))
            final_fake = jax.lax.stop_gradient(jnp.argmax(step_logits[-1], axis=-1).astype(jnp.int32))
        candidates = jnp.stack([targets, random_fake, early_fake, mid_fake, final_fake], axis=1)
        flat_candidates = candidates.reshape(inputs.shape[0] * candidates.shape[1], inputs.shape[1])
        flat_inputs = jnp.repeat(inputs, candidates.shape[1], axis=0)
        candidate_energy = model.verifier_energy(flat_inputs, flat_candidates).reshape(inputs.shape[0], candidates.shape[1])
        true_energy_per_example = candidate_energy[:, 0]
        fake_energy_all = candidate_energy[:, 1:]
        fake_wrong = jnp.any(candidates[:, 1:] != targets[:, None, :], axis=-1)
        fake_wrong_f32 = fake_wrong.astype(jnp.float32)
        fake_normalizer = jnp.maximum(jnp.sum(fake_wrong_f32), 1.0)
        verifier_margin_loss = jax.nn.relu(brc.verifier_margin + true_energy_per_example[:, None] - fake_energy_all)
        verifier_loss = jnp.sum(verifier_margin_loss * fake_wrong_f32) / fake_normalizer
        verifier_ranking_accuracy = (
            jnp.sum((true_energy_per_example[:, None] < fake_energy_all).astype(jnp.float32) * fake_wrong_f32)
            / fake_normalizer
        )
        true_energy = jnp.mean(true_energy_per_example)
        fake_energy = jnp.sum(fake_energy_all * fake_wrong_f32) / fake_normalizer

    fit_metrics = _zero_brc_fit_metrics(zero)
    meta_outer_loss = zero
    if brc.meta_loss_weight != 0.0:
        z_fit, fit_metrics = _verifier_guided_latent_fit(
            model,
            inputs,
            given_mask,
            z0,
            train=train,
            key=fit_key,
        )
        meta_logits, _ = model.run_diffusion(
            inputs,
            z=z_fit,
            initial_belief=model.initial_belief_logits(inputs),
            train=train,
            dropout_key=meta_key,
            return_final_only=True,
        )
        meta_outer_loss = _masked_token_ce(meta_logits, targets, loss_mask)

    loss = (
        step_ce_loss
        + brc.verifier_loss_weight * verifier_loss
        + brc.meta_loss_weight * meta_outer_loss
    )

    predictions = jnp.argmax(final_logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask.astype(jnp.float32)
    blank_cell_accuracy = jnp.sum(correct) / normalizer
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask.astype(jnp.float32), axis=-1), 1.0)
    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))
    solved_count = jnp.sum(solved_examples.astype(jnp.float32))
    probs = jax.nn.softmax(final_logits, axis=-1)
    target_probability = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask.astype(jnp.float32)) / normalizer
    if train:
        oracle_step = jnp.argmin(per_step_loss)
    else:
        per_step_example_loss = jnp.sum(token_loss * step_loss_mask.astype(jnp.float32), axis=-1) / per_example_normalizer[None, :]
        oracle_step = jnp.mean(jnp.argmin(per_step_example_loss, axis=0).astype(jnp.float32))
    given_consistency, invalid_board_rate, conflict_count = _sudoku_board_metrics(model, predictions, inputs, given_mask)

    metrics = {
        "loss": loss,
        "blank_ce_loss": step_ce_loss,
        "step_weighted_ce_loss": step_ce_loss,
        "final_blank_ce_loss": solution_loss,
        "mean_blank_ce_loss": jnp.mean(per_step_loss),
        "latent_fit_loss": fit_metrics["latent_fit_loss"],
        "fit_given_loss": fit_metrics["fit_given_loss"],
        "fit_energy": fit_metrics["fit_energy"],
        "fit_consistency_loss": fit_metrics["fit_consistency_loss"],
        "fit_prior_loss": fit_metrics["fit_prior_loss"],
        "latent_update_norm": fit_metrics["latent_update_norm"],
        "latent_grad_norm": fit_metrics["latent_grad_norm"],
        "latent_step_norm": fit_metrics["latent_step_norm"],
        "meta_outer_loss": meta_outer_loss,
        "verifier_loss": verifier_loss,
        "verifier_ranking_accuracy": verifier_ranking_accuracy,
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": solved_count,
        "oracle_step": oracle_step.astype(jnp.float32) + 1.0,
        "given_consistency": given_consistency,
        "invalid_board_rate": invalid_board_rate,
        "conflict_count": conflict_count,
        "belief_init_noise_rate": belief_init_noise_rate,
        "belief_init_uniform_rate": belief_init_uniform_rate,
        "belief_init_teacher_rate": belief_init_teacher_rate,
        "belief_init_corrupt_rate": belief_init_corrupt_rate,
        "belief_init_soft_rate": belief_init_soft_rate,
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_loss_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "diffusion_filled_ratio": diagnostics["diffusion_filled_ratio"],
        "brc_gate_mean": diagnostics["brc_gate_mean"],
        "brc_gate_std": diagnostics["brc_gate_std"],
        "true_energy": true_energy,
        "fake_energy": fake_energy,
    }
    return loss, metrics


def build_ema_update_runner(decay: float, wrt=nnx.Param):
    def update_ema_model(ema_model: GridReasoningModel, model: GridReasoningModel) -> None:
        ema_params = nnx.state(ema_model, wrt)
        model_params = nnx.state(model, wrt)
        nnx.update(ema_model, nnx.state(model))
        updated_params = jax.tree.map(
            lambda ema, current: decay * ema + (1.0 - decay) * current,
            ema_params,
            model_params,
        )
        nnx.update(ema_model, updated_params)

    return nnx.jit(update_ema_model)


def loss_and_metrics(
    model: GridReasoningModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if isinstance(model, BRCSudokuModel):
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
    loss_mask = (~given_mask).astype(jnp.float32)
    loss_weights = _target_loss_weights(model, targets)
    weighted_loss_mask = loss_mask * loss_weights
    normalizer = _weighted_mask_normalizer(loss_mask, loss_weights)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)

    model_key = dropout_key

    compute_terminal_residual = terminal_residual_weight != 0.0
    model_forward_kwargs = {}
    if isinstance(model, TinyRecursiveModel):
        model_forward_kwargs = {
            "compute_terminal_residual": compute_terminal_residual,
            "puzzle_identifiers": batch["puzzle_identifiers"],
        }
    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=train,
        dropout_key=model_key,
        **model_forward_kwargs,
    )
    effective_step_logits = (
        _clamp_logits_to_given(step_logits, inputs, given_mask, model.config.vocab_size)
        if _should_clamp_given(model)
        else step_logits
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = weighted_loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(effective_step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    per_example_normalizer = jnp.maximum(jnp.sum(weighted_loss_mask, axis=-1), 1.0)
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    num_steps = effective_step_logits.shape[0]
    step_weights = _trm_step_loss_weights(model, num_steps)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    step_predictions = jnp.argmax(effective_step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    halt_logits = diagnostics.get("halt_logits")
    halt_loss = jnp.asarray(0.0, dtype=jnp.float32)
    halt_selected_logits = effective_step_logits[-1]
    halt_selected_step = jnp.full((inputs.shape[0],), effective_step_logits.shape[0] - 1, dtype=jnp.int32)
    if halt_logits is not None and halt_loss_weight != 0.0:
        halt_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(halt_logits, halt_targets))
        halt_selected_step = _trm_halt_selected_step(halt_logits)
        gather_index = halt_selected_step[None, :, None, None]
        halt_selected_logits = jnp.take_along_axis(
            effective_step_logits,
            jnp.broadcast_to(gather_index, (1, inputs.shape[0], effective_step_logits.shape[2], effective_step_logits.shape[3])),
            axis=0,
        ).squeeze(0)
    terminal_residual = diagnostics.get(
        "terminal_belief_mse",
        diagnostics.get("terminal_belief_delta", jnp.asarray(0.0, dtype=jnp.float32)),
    )
    loss = (
        blank_ce_loss
        + halt_loss_weight * halt_loss
        + terminal_residual_weight * terminal_residual
    )

    final_logits = effective_step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    final_probs = jax.nn.softmax(final_logits, axis=-1)
    target_probability = jnp.take_along_axis(final_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / metric_normalizer

    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))
    solved_count = jnp.sum(solved_examples.astype(jnp.float32))
    if halt_loss_weight != 0.0:
        halt_selected_token_loss = optax.softmax_cross_entropy_with_integer_labels(halt_selected_logits, targets)
        halt_selected_ce_loss = jnp.sum(halt_selected_token_loss * weighted_loss_mask) / normalizer
        halt_selected_predictions = jnp.argmax(halt_selected_logits, axis=-1)
        halt_selected_correct = (halt_selected_predictions == targets).astype(jnp.float32) * loss_mask
        halt_selected_accuracy = jnp.sum(halt_selected_correct) / metric_normalizer
        halt_selected_correct_per_example = jnp.sum(halt_selected_correct, axis=-1)
        halt_selected_solved_examples = jnp.where(
            blanks_per_example > 0,
            halt_selected_correct_per_example == blanks_per_example,
            True,
        )
        halt_selected_solved_rate = jnp.mean(halt_selected_solved_examples.astype(jnp.float32))
    else:
        halt_selected_ce_loss = per_step_loss[-1]
        halt_selected_accuracy = blank_cell_accuracy
        halt_selected_solved_rate = solved_rate
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)

    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "step_weighted_ce_loss": blank_ce_loss,
        "halt_loss": halt_loss,
        "final_blank_ce_loss": per_step_loss[-1],
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "halt_selected_blank_ce_loss": halt_selected_ce_loss,
        "halt_selected_blank_cell_accuracy": halt_selected_accuracy,
        "halt_selected_solved_rate": halt_selected_solved_rate,
        "halt_selected_step": jnp.mean(halt_selected_step.astype(jnp.float32) + 1.0),
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    metrics.update(_path_metrics(predictions, targets, loss_mask))
    if halt_loss_weight != 0.0:
        halt_path_metrics = _path_metrics(halt_selected_predictions, targets, loss_mask)
        metrics.update(
            {
                f"halt_selected_{key}": value
                for key, value in halt_path_metrics.items()
                if key in ("path_precision", "path_recall", "path_f1")
            }
        )
    if halt_logits is not None and halt_loss_weight != 0.0:
        metrics["per_step_halt_probability"] = jax.nn.sigmoid(jnp.mean(halt_logits, axis=1))
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
        "cell_attention_gate_mean",
        "cell_attention_gate_std",
        "cell_attention_gate_low_saturation",
        "cell_attention_gate_high_saturation",
        "cell_attention_local_distance_scale",
        "attention_gate_mean",
        "attention_gate_std",
        "attention_gate_low_saturation",
        "attention_gate_high_saturation",
        "attention_local_distance_scale",
        "halt_loss",
        "halt_selected_blank_ce_loss",
        "halt_selected_blank_cell_accuracy",
        "halt_selected_solved_rate",
        "halt_selected_step",
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
    loss_mask = (~given_mask).astype(jnp.float32)
    loss_weights = _target_loss_weights(model, targets)
    weighted_loss_mask = loss_mask * loss_weights
    normalizer = _weighted_mask_normalizer(loss_mask, loss_weights)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(weighted_loss_mask, axis=-1), 1.0)
    if _should_clamp_given(model):
        logits = _clamp_logits_to_given(logits, new_carry["current_inputs"], given_mask, model.config.vocab_size)

    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    per_example_loss = jnp.sum(token_loss * weighted_loss_mask, axis=-1) / per_example_normalizer
    blank_ce_loss = jnp.mean(per_example_loss)

    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / metric_normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_targets = jax.lax.stop_gradient(solved_examples.astype(jnp.float32))
    solved_rate = jnp.mean(solved_targets)
    solved_count = jnp.sum(solved_targets)

    halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(diagnostics["halt_logits"], solved_targets))
    loss = blank_ce_loss + halt_loss_weight * halt_loss

    probs = jax.nn.softmax(logits, axis=-1)
    target_probability = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "final_blank_ce_loss": blank_ce_loss,
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": solved_count,
        "halt_loss": halt_loss,
        "act_step": diagnostics["act_step"],
        "halted_rate": diagnostics["halted_rate"],
        "reset_rate": diagnostics["reset_rate"],
        "attention_gate_mean": diagnostics["attention_gate_mean"],
        "attention_gate_std": diagnostics["attention_gate_std"],
        "attention_gate_low_saturation": diagnostics["attention_gate_low_saturation"],
        "attention_gate_high_saturation": diagnostics["attention_gate_high_saturation"],
        "attention_local_distance_scale": diagnostics["attention_local_distance_scale"],
    }
    metrics.update(_path_metrics(predictions, targets, loss_mask))
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
    loss_mask = (~given_mask).astype(jnp.float32)
    loss_weights = _target_loss_weights(model, targets)
    weighted_loss_mask = loss_mask * loss_weights
    normalizer = _weighted_mask_normalizer(loss_mask, loss_weights)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(weighted_loss_mask, axis=-1), 1.0)

    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        puzzle_identifiers=batch["puzzle_identifiers"],
        train=train,
        dropout_key=dropout_key,
        compute_terminal_residual=False,
    )
    if _should_clamp_given(model):
        step_logits = _clamp_logits_to_given(step_logits, inputs, given_mask, model.config.vocab_size)
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = weighted_loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    num_steps = per_step_loss.shape[0]
    step_weights = _trm_step_loss_weights(model, num_steps)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    mean_blank_ce_loss = jnp.mean(per_step_loss)
    final_blank_ce_loss = per_step_loss[-1]
    blanks_per_example = jnp.sum(loss_mask, axis=-1)

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / metric_normalizer
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    final_logits = step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    final_probs = jax.nn.softmax(final_logits, axis=-1)
    target_probability = jnp.take_along_axis(final_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / metric_normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))
    solved_count = jnp.sum(solved_examples.astype(jnp.float32))

    halt_logits = diagnostics.get("halt_logits")
    halt_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if halt_logits is not None and halt_loss_weight != 0.0:
        halt_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(halt_logits, halt_targets))

    loss = blank_ce_loss + halt_loss_weight * halt_loss
    oracle_step = jnp.argmin(
        jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :],
        axis=0,
    )
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "step_weighted_ce_loss": blank_ce_loss,
        "mean_blank_ce_loss": mean_blank_ce_loss,
        "final_blank_ce_loss": final_blank_ce_loss,
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": solved_count,
        "halt_loss": halt_loss,
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.float32),
    }
    metrics.update(_path_metrics(predictions, targets, loss_mask))
    if halt_logits is not None and halt_loss_weight != 0.0:
        metrics["per_step_halt_probability"] = jax.nn.sigmoid(jnp.mean(halt_logits, axis=1))
    for key in (
        "attention_gate_mean",
        "attention_gate_std",
        "attention_gate_low_saturation",
        "attention_gate_high_saturation",
        "attention_local_distance_scale",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    return loss, metrics


def trm_eval_loss_and_metrics(
    model: TinyRecursiveModel,
    batch: dict[str, jax.Array],
    halt_loss_weight: float = 0.5,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    loss_weights = _target_loss_weights(model, targets)
    weighted_loss_mask = loss_mask * loss_weights
    normalizer = _weighted_mask_normalizer(loss_mask, loss_weights)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(weighted_loss_mask, axis=-1), 1.0)

    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        puzzle_identifiers=batch["puzzle_identifiers"],
        train=False,
        dropout_key=None,
        compute_terminal_residual=False,
    )
    if _should_clamp_given(model):
        step_logits = _clamp_logits_to_given(step_logits, inputs, given_mask, model.config.vocab_size)
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = weighted_loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / metric_normalizer
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    halt_logits = diagnostics["halt_logits"]
    selected_step = _trm_halt_selected_step(halt_logits)
    gather_index = selected_step[None, :, None, None]
    selected_logits = jnp.take_along_axis(
        step_logits,
        jnp.broadcast_to(gather_index, (1, inputs.shape[0], step_logits.shape[2], step_logits.shape[3])),
        axis=0,
    ).squeeze(0)
    selected_token_loss = optax.softmax_cross_entropy_with_integer_labels(selected_logits, targets)
    blank_ce_loss = jnp.sum(selected_token_loss * weighted_loss_mask) / normalizer
    selected_solved_targets = jnp.take_along_axis(
        per_step_example_solved.astype(jnp.float32),
        selected_step[None, :],
        axis=0,
    ).squeeze(0)
    selected_solved_count = jnp.sum(selected_solved_targets)
    halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(halt_logits, jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))))
    loss = blank_ce_loss + halt_loss_weight * halt_loss

    predictions = jnp.argmax(selected_logits, axis=-1)
    selected_probs = jax.nn.softmax(selected_logits, axis=-1)
    target_probability = jnp.take_along_axis(selected_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / metric_normalizer
    solved_rate = jnp.mean(selected_solved_targets)
    final_logits = step_logits[-1]
    final_token_loss = optax.softmax_cross_entropy_with_integer_labels(final_logits, targets)
    final_blank_ce_loss = jnp.sum(final_token_loss * weighted_loss_mask) / normalizer
    final_predictions = jnp.argmax(final_logits, axis=-1)
    final_correct = (final_predictions == targets).astype(jnp.float32) * loss_mask
    final_blank_cell_accuracy = jnp.sum(final_correct) / metric_normalizer
    final_correct_per_example = jnp.sum(final_correct, axis=-1)
    final_solved_examples = jnp.where(
        blanks_per_example > 0,
        final_correct_per_example == blanks_per_example,
        True,
    )
    final_solved_count = jnp.sum(final_solved_examples.astype(jnp.float32))
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "halt_loss": halt_loss,
        "final_blank_ce_loss": final_blank_ce_loss,
        "final_blank_cell_accuracy": final_blank_cell_accuracy,
        "final_solved_rate": jnp.mean(final_solved_examples.astype(jnp.float32)),
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": selected_solved_count,
        "halt_selected_blank_ce_loss": blank_ce_loss,
        "halt_selected_blank_cell_accuracy": blank_cell_accuracy,
        "halt_selected_solved_rate": solved_rate,
        "halt_selected_solved_count": selected_solved_count,
        "halt_selected_step": jnp.mean(selected_step.astype(jnp.float32) + 1.0),
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "final_solved_count": final_solved_count,
        "per_step_loss": per_step_loss,
        "per_step_accuracy": per_step_accuracy,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "per_step_halt_probability": jax.nn.sigmoid(jnp.mean(halt_logits, axis=1)),
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.int32),
    }
    metrics.update(_path_metrics(predictions, targets, loss_mask))
    metrics.update(
        {
            f"halt_selected_{key}": value
            for key, value in _path_metrics(predictions, targets, loss_mask).items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    metrics.update(
        {
            f"final_{key}": value
            for key, value in _path_metrics(final_predictions, targets, loss_mask).items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    for key in (
        "attention_gate_mean",
        "attention_gate_std",
        "attention_gate_low_saturation",
        "attention_gate_high_saturation",
        "attention_local_distance_scale",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    return loss, metrics


def build_train_step_runner(
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
):
    def train_step_with_weight(
        model: GridReasoningModel,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> dict[str, jax.Array]:
        def weighted_loss_and_metrics(model, batch, train, dropout_key):
            return loss_and_metrics(
                model,
                batch,
                train,
                dropout_key,
                halt_loss_weight,
                terminal_residual_weight,
            )

        grad_fn = nnx.value_and_grad(weighted_loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(model, batch, True, dropout_key)
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step_with_weight)


def build_trm_act_train_step_runner(halt_loss_weight: float = 0.5):
    def train_step(
        model: TinyRecursiveModel,
        optimizer: nnx.Optimizer,
        carry: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        def objective(model, carry, batch, train, dropout_key):
            return trm_act_loss_and_metrics(
                model,
                carry,
                batch,
                train,
                dropout_key,
                halt_loss_weight,
            )

        (_, (metrics, new_carry)), grads = nnx.value_and_grad(objective, has_aux=True)(
            model,
            carry,
            batch,
            True,
            dropout_key,
        )
        optimizer.update(model, grads)
        return metrics, new_carry

    return nnx.jit(train_step)


def build_trm_dense_unroll_train_step_runner(
    *,
    halt_loss_weight: float = 0.0,
):
    def train_step(
        model: TinyRecursiveModel,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> dict[str, jax.Array]:
        def objective(model, batch, train, dropout_key):
            return trm_dense_unroll_loss_and_metrics(
                model,
                batch,
                train,
                dropout_key,
                halt_loss_weight=halt_loss_weight,
            )

        (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(
            model,
            batch,
            True,
            dropout_key,
        )
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step)


def build_eval_step_runner(
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
):
    def eval_step_with_weight(
        model: GridReasoningModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = loss_and_metrics(
            model,
            batch,
            False,
            None,
            halt_loss_weight,
            terminal_residual_weight,
        )
        return metrics

    return nnx.jit(eval_step_with_weight)


def build_trm_eval_step_runner(halt_loss_weight: float = 0.5):
    def eval_step(
        model: TinyRecursiveModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = trm_eval_loss_and_metrics(
            model,
            batch,
            halt_loss_weight,
        )
        return metrics

    return nnx.jit(eval_step)


def save_checkpoint(
    checkpoint_dir: str,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    step: int,
    *,
    ema_model: GridReasoningModel | None = None,
) -> None:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    target_dir = Path(checkpoint_dir).resolve() / f"step_{step}"
    payload = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
        "uses_ema_model": ema_model is not None,
    }
    if ema_model is not None:
        payload["ema_model"] = nnx.state(ema_model)
    checkpointer.save(target_dir, payload, force=True)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    *,
    ema_model: GridReasoningModel | None = None,
) -> int:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restore_target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": 0,
        "uses_ema_model": False,
    }
    if ema_model is not None:
        restore_target["ema_model"] = nnx.state(ema_model)
    payload = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        args=ocp_args.PyTreeRestore(
            item=restore_target,
            restore_args=orbax_utils.restore_args_from_target(restore_target),
            partial_restore=True,
        ),
    )
    nnx.update(model, payload["model"])
    nnx.update(optimizer, payload["optimizer"])
    if ema_model is not None:
        if "ema_model" in payload:
            nnx.update(ema_model, payload["ema_model"])
        else:
            nnx.update(ema_model, payload["model"])
    return int(payload["step"])

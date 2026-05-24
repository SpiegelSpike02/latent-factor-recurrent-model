from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import optax.contrib

from lfrm.config import ExperimentConfig


def scale_by_adam_atan2(
    *,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Adam-style moments with atan2 normalization, matching URM's strong optimizer."""

    def init_fn(params):
        return optax.ScaleByAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree.map(jnp.zeros_like, params),
            nu=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        del params
        count_inc = optax.safe_increment(state.count)
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        mu_hat = optax.bias_correction(mu, b1, count_inc)
        nu_hat = optax.bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: jnp.arctan2(m, jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
        )
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scheduled_lr(
    *,
    peak_value: float,
    min_ratio: float,
    mid_ratio: float,
    mid_fraction: float,
    warmup_steps: int,
    optimizer_updates: int,
):
    if mid_ratio > 0.0 and 0.0 < mid_fraction < 1.0:
        decay_steps = max(optimizer_updates, warmup_steps + 1)
        decay_span = max(decay_steps - warmup_steps, 1)
        mid_value = peak_value * mid_ratio
        end_value = peak_value * min_ratio

        def schedule(count):
            count = jnp.asarray(count)
            warmup = peak_value * count / max(warmup_steps, 1)
            progress = jnp.clip(
                (count - warmup_steps) / decay_span,
                0.0,
                1.0,
            )
            fast_progress = jnp.clip(progress / mid_fraction, 0.0, 1.0)
            fast_cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * fast_progress))
            fast_value = mid_value + (peak_value - mid_value) * fast_cosine
            slow_progress = jnp.clip((progress - mid_fraction) / (1.0 - mid_fraction), 0.0, 1.0)
            slow_cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * slow_progress))
            slow_value = end_value + (mid_value - end_value) * slow_cosine
            decay_value = jnp.where(progress <= mid_fraction, fast_value, slow_value)
            return jnp.where(count <= warmup_steps, warmup, decay_value)

        return schedule

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=max(optimizer_updates, warmup_steps + 1),
        end_value=peak_value * min_ratio,
    )
    return schedule


def _is_puzzle_embedding_path(path: tuple[object, ...]) -> bool:
    return any(getattr(entry, "key", None) == "puzzle_embed" for entry in path)


def _path_keys(path: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str(getattr(entry, "key", entry)) for entry in path)


def _is_muon_matrix_path(path: tuple[object, ...], value) -> bool:
    if getattr(value, "ndim", None) != 2:
        return False
    keys = _path_keys(path)
    return "puzzle_embed" not in keys


def _muon_dimension_numbers(params):
    dim_nums = optax.contrib.MuonDimensionNumbers()
    return jax.tree_util.tree_map_with_path(
        lambda path, value: dim_nums if _is_muon_matrix_path(path, value) else None,
        params,
    )


def _optimizer_param_labels(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _value: "puzzle_embed" if _is_puzzle_embedding_path(path) else "default",
        params,
    )


def _uses_puzzle_embedding(config: ExperimentConfig) -> bool:
    if config.model.model_type == "trm":
        return config.model.trm_config.puzzle_embed_len > 0 or config.model.trm_config.puzzle_embed_ndim > 0
    if config.model.model_type == "urm":
        return config.model.urm_config.puzzle_embed_len > 0 or config.model.urm_config.puzzle_embed_ndim > 0
    return False


def uses_sparse_puzzle_embedding(config: ExperimentConfig) -> bool:
    return (
        config.model.model_type in ("trm", "urm")
        and _uses_puzzle_embedding(config)
        and config.optimizer.puzzle_embed_learning_rate > 0.0
        and config.train.train_mode == "act"
    )


def trainable_param_filter(config: ExperimentConfig):
    if uses_sparse_puzzle_embedding(config):
        from flax import nnx

        return nnx.All(nnx.Param, nnx.Not(nnx.PathContains("puzzle_embed")))
    from flax import nnx

    return nnx.Param


def _adam_atan2_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    transforms.append(scale_by_adam_atan2(b1=config.optimizer.beta1, b2=config.optimizer.beta2))
    if config.optimizer.weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(config.optimizer.weight_decay))
    transforms.extend((optax.scale_by_schedule(schedule), optax.scale(-1.0)))
    return optax.chain(*transforms)


def _muon_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if config.optimizer.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.optimizer.grad_clip_norm))
    transforms.append(
        optax.contrib.muon(
            learning_rate=schedule,
            beta=config.optimizer.beta2,
            weight_decay=config.optimizer.weight_decay,
            adam_b1=config.optimizer.beta1,
            adam_b2=config.optimizer.beta2,
            adam_weight_decay=0.0,
            muon_weight_dimension_numbers=_muon_dimension_numbers,
        )
    )
    return optax.chain(*transforms)


def _default_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
    if config.optimizer.optimizer_type == "muon":
        return _muon_optimizer(config, schedule)
    transforms: list[optax.GradientTransformation] = []
    if config.optimizer.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.optimizer.grad_clip_norm))
    if config.optimizer.optimizer_type == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=schedule,
                b1=config.optimizer.beta1,
                b2=config.optimizer.beta2,
                weight_decay=config.optimizer.weight_decay,
            )
        )
    elif config.optimizer.optimizer_type == "adam_atan2":
        transforms.append(_adam_atan2_optimizer(config, schedule))
    else:
        raise ValueError(f"Unsupported optimizer_type: {config.optimizer.optimizer_type}")
    return optax.chain(*transforms)


def _puzzle_embedding_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    transforms.append(optax.scale_by_sign())
    if config.optimizer.puzzle_embed_weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(config.optimizer.puzzle_embed_weight_decay))
    transforms.extend((optax.scale_by_schedule(schedule), optax.scale(-1.0)))
    return optax.chain(*transforms)


def build_optimizer(config: ExperimentConfig, model: object | None = None) -> optax.GradientTransformation:
    optimizer_updates = max(1, config.train.optimizer_updates)
    warmup_steps = max(1, config.optimizer.lr_warmup_steps)
    schedule = scheduled_lr(
        peak_value=config.optimizer.learning_rate,
        min_ratio=config.optimizer.lr_min_ratio,
        mid_ratio=config.optimizer.lr_mid_ratio,
        mid_fraction=config.optimizer.lr_mid_fraction,
        warmup_steps=warmup_steps,
        optimizer_updates=optimizer_updates,
    )
    default_optimizer = _default_optimizer(config, schedule)
    if (
        model is not None
        and _uses_puzzle_embedding(config)
        and not uses_sparse_puzzle_embedding(config)
        and config.optimizer.puzzle_embed_learning_rate > 0.0
    ):
        puzzle_schedule = scheduled_lr(
            peak_value=config.optimizer.puzzle_embed_learning_rate,
            min_ratio=config.optimizer.lr_min_ratio,
            mid_ratio=config.optimizer.lr_mid_ratio,
            mid_fraction=config.optimizer.lr_mid_fraction,
            warmup_steps=warmup_steps,
            optimizer_updates=optimizer_updates,
        )
        optimizer = optax.multi_transform(
            {
                "default": default_optimizer,
                "puzzle_embed": _puzzle_embedding_optimizer(config, puzzle_schedule),
            },
            _optimizer_param_labels,
        )
    else:
        optimizer = default_optimizer
    return optimizer

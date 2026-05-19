from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

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
    warmup_updates: int,
    optimizer_updates: int,
):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=warmup_updates,
        decay_steps=max(optimizer_updates, warmup_updates + 1),
        end_value=peak_value * min_ratio,
    )
    return lambda count: schedule(count + jnp.asarray(1, dtype=count.dtype))


def _is_puzzle_embedding_path(path: tuple[object, ...]) -> bool:
    return any(getattr(entry, "key", None) == "puzzle_embed" for entry in path)


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


def _adam_atan2_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    transforms.append(scale_by_adam_atan2(b1=config.optimizer.beta1, b2=config.optimizer.beta2))
    if config.optimizer.weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(config.optimizer.weight_decay))
    transforms.extend((optax.scale_by_schedule(schedule), optax.scale(-1.0)))
    return optax.chain(*transforms)


def _default_optimizer(config: ExperimentConfig, schedule) -> optax.GradientTransformation:
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
    warmup_updates = max(1, config.optimizer.warmup_updates)
    schedule = scheduled_lr(
        peak_value=config.optimizer.learning_rate,
        min_ratio=config.optimizer.lr_min_ratio,
        warmup_updates=warmup_updates,
        optimizer_updates=optimizer_updates,
    )
    default_optimizer = _default_optimizer(config, schedule)
    if (
        model is not None
        and _uses_puzzle_embedding(config)
        and config.optimizer.puzzle_embed_learning_rate > 0.0
    ):
        puzzle_schedule = scheduled_lr(
            peak_value=config.optimizer.puzzle_embed_learning_rate,
            min_ratio=config.optimizer.lr_min_ratio,
            warmup_updates=warmup_updates,
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
    if config.optimizer.flatten_optimizer:
        optimizer = optax.flatten(optimizer)
    if config.train.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer,
            every_k_schedule=config.train.gradient_accumulation_steps,
        )
    return optimizer

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from lfrm.models.recurrent.layers import (
    Array,
    CastedEmbedding,
    FullAttention,
    casted_linear_init,
    compute_dtype,
    maybe_cast,
    swiglu_intermediate_size,
    trunc_normal,
    unscaled_rms_norm,
)


class ConvSwiGLU(nnx.Module):
    """URM feed-forward block: SwiGLU, sequence depthwise conv, SiLU, down projection."""

    def __init__(self, config: ModelConfig, dtype: jnp.dtype, *, rngs: nnx.Rngs) -> None:
        urm = config.urm_config
        if urm.conv_kernel < 1:
            raise ValueError("URM conv_kernel must be positive")
        self.dtype = dtype
        hidden_size = config.d_model
        intermediate_size = swiglu_intermediate_size(hidden_size, urm.mlp_ratio, min_size=256)
        self.gate_up = nnx.Linear(
            hidden_size,
            2 * intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.depthwise = nnx.Conv(
            intermediate_size,
            intermediate_size,
            kernel_size=(urm.conv_kernel,),
            padding=[(urm.conv_kernel // 2, urm.conv_kernel // 2)],
            feature_group_count=intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=dtype,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, h: Array) -> Array:
        gate, up = jnp.split(self.gate_up(maybe_cast(h, self.dtype)), 2, axis=-1)
        x = jax.nn.silu(gate) * up
        x = self.depthwise(x)[:, : h.shape[1], :]
        x = jax.nn.silu(x)
        return self.down(x)


class URMBlock(nnx.Module):
    def __init__(self, config: ModelConfig, dtype: jnp.dtype, *, rngs: nnx.Rngs) -> None:
        self.dtype = dtype
        self.norm_eps = config.urm_config.rms_norm_eps
        self.attention = FullAttention(
            config.d_model,
            config.urm_config.num_heads,
            dtype,
            name="URM",
            rngs=rngs,
        )
        self.attention_norm = unscaled_rms_norm(config.d_model, self.norm_eps, dtype, rngs)
        self.conv_swiglu = ConvSwiGLU(config, dtype, rngs=rngs)
        self.mlp_norm = unscaled_rms_norm(config.d_model, self.norm_eps, dtype, rngs)

    def __call__(self, h: Array, rope_cos: Array, rope_sin: Array) -> Array:
        attn_output = self.attention(
            h,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        h = self.attention_norm(h + attn_output.astype(h.dtype))
        mlp_output = self.conv_swiglu(maybe_cast(h, self.dtype))
        return self.mlp_norm(h + mlp_output.astype(h.dtype))


class UnifiedReasoningModel(nnx.Module):
    """URM aligned to the official loops/H_cycles/L_cycles ACT formulation."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        if config.model_type != "urm":
            raise ValueError("UnifiedReasoningModel requires model_type='urm'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        urm = config.urm_config
        if min(urm.recurrent_steps, urm.h_cycles, urm.l_cycles, urm.l_layers) < 1:
            raise ValueError("URM recurrent_steps, h_cycles, l_cycles, and l_layers must be positive")
        if urm.num_heads < 1 or config.d_model % urm.num_heads != 0:
            raise ValueError("URM d_model must be divisible by num_heads")
        if urm.mlp_ratio < 1:
            raise ValueError("URM mlp_ratio must be at least 1")
        if urm.step_loss_weights is not None and len(urm.step_loss_weights) != urm.recurrent_steps:
            raise ValueError("URM step_loss_weights length must equal recurrent_steps")

        self.config = config
        self.runtime = runtime
        self.urm = urm
        self.recurrent_steps = int(urm.recurrent_steps)
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.puzzle_embed_len = max(0, int(urm.puzzle_embed_len))
        self.total_seq_len = config.seq_len + self.puzzle_embed_len

        self.token_embed = CastedEmbedding(
            int(config.input_vocab_size or config.vocab_size),
            config.d_model,
            self.dtype,
            init_std=1.0 / self.embed_scale,
            rngs=rngs,
        )
        self.has_puzzle_embed = self.puzzle_embed_len > 0
        if self.puzzle_embed_len > 0:
            puzzle_embed_ndim = self.puzzle_embed_len * config.d_model
            if urm.puzzle_embed_ndim > 0:
                puzzle_embed_ndim = urm.puzzle_embed_ndim
            if puzzle_embed_ndim != self.puzzle_embed_len * config.d_model:
                raise ValueError("URM puzzle_embed_ndim must equal puzzle_embed_len * d_model in this JAX port")
            self.puzzle_embed = CastedEmbedding(
                config.num_puzzle_identifiers,
                puzzle_embed_ndim,
                self.dtype,
                init_std=0.0,
                rngs=rngs,
            )

        self.blocks = nnx.List([URMBlock(config, self.dtype, rngs=rngs) for _ in range(urm.l_layers)])
        self.output_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.q_head = nnx.Linear(
            config.d_model,
            2,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(-5.0),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

        head_dim = config.d_model // urm.num_heads
        if head_dim % 2 != 0:
            raise ValueError("URM RoPE requires an even attention head dimension")
        positions = jnp.arange(self.total_seq_len, dtype=jnp.float32)
        inv_freq = 1.0 / (urm.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        freqs = jnp.einsum("n,d->nd", positions, inv_freq)
        rope = jnp.concatenate((freqs, freqs), axis=-1)
        self.rope_cos = nnx.data(jnp.cos(rope))
        self.rope_sin = nnx.data(jnp.sin(rope))

        self.init_hidden = nnx.data(
            trunc_normal(rngs.params(), (config.d_model,), std=1.0, dtype=self.dtype)
        )

    def _input_embeddings(
        self,
        tokens: Array,
        puzzle_identifiers: Array | None,
        puzzle_embeddings: Array | None = None,
    ) -> Array:
        token_emb = self.token_embed(tokens.astype(jnp.int32), train=True).astype(self.dtype) * self.embed_scale
        if not self.has_puzzle_embed:
            return token_emb
        if puzzle_embeddings is None:
            if puzzle_identifiers is None:
                puzzle_identifiers = jnp.zeros((tokens.shape[0],), dtype=jnp.int32)
            if not hasattr(self, "puzzle_embed"):
                raise ValueError("URM puzzle embedding table is not initialized")
            prefix = self.puzzle_embed(puzzle_identifiers, train=True)
        else:
            prefix = puzzle_embeddings
        prefix = prefix.reshape(tokens.shape[0], self.puzzle_embed_len, self.config.d_model)
        return jnp.concatenate((prefix, token_emb), axis=1)

    def _run_layers(self, hidden: Array, input_embeddings: Array) -> Array:
        hidden = hidden + input_embeddings
        for block in self.blocks:
            hidden = block(hidden, self.rope_cos, self.rope_sin)
        return hidden

    def _inner_update(self, hidden: Array, input_embeddings: Array, *, train: bool, dropout_key: Array) -> Array:
        def latent_update(h: Array) -> Array:
            for _ in range(self.urm.l_cycles):
                h = self._run_layers(h, input_embeddings)
            return h

        h = hidden
        for _ in range(self.urm.h_cycles - 1):
            h = jax.lax.stop_gradient(latent_update(h))

        h = latent_update(h)
        h = self.dropout(h, deterministic=not train, rngs=dropout_key)
        return h

    def _logits_and_halt(self, hidden: Array) -> tuple[Array, Array]:
        visible_hidden = hidden[:, self.puzzle_embed_len :, :]
        logits = self.output_head(maybe_cast(visible_hidden, self.dtype)).astype(jnp.float32)
        halt_hidden = hidden[:, 0, :] if self.puzzle_embed_len > 0 else visible_hidden[:, 0, :]
        q_logits = self.q_head(maybe_cast(halt_hidden, self.dtype)).astype(jnp.float32)
        return logits, q_logits[..., 0]

    def initial_carry(self, batch: dict[str, Array]) -> dict[str, Array]:
        batch_size = batch["inputs"].shape[0]
        hidden = jnp.broadcast_to(
            self.init_hidden[...][None, None, :],
            (batch_size, self.total_seq_len, self.config.d_model),
        )
        return {
            "hidden": hidden.astype(self.dtype),
            "steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "halted": jnp.ones((batch_size,), dtype=bool),
            "current_inputs": jnp.zeros_like(batch["inputs"]),
            "current_labels": jnp.zeros_like(batch["labels"]),
            "current_puzzle_identifiers": jnp.zeros_like(batch["puzzle_identifiers"]),
        }

    def _reset_hidden(self, batch_size: int) -> Array:
        hidden = jnp.broadcast_to(
            self.init_hidden[...][None, None, :],
            (batch_size, self.total_seq_len, self.config.d_model),
        )
        return hidden.astype(self.dtype)

    def forward_act_step(
        self,
        carry: dict[str, Array],
        batch: dict[str, Array],
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_embeddings: Array | None = None,
    ) -> tuple[dict[str, Array], Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        reset = carry["halted"]
        reset_broadcast = reset[:, None]
        reset_hidden = reset[:, None, None]
        inputs = jnp.where(reset_broadcast, batch["inputs"], carry["current_inputs"])
        labels = jnp.where(reset_broadcast, batch["labels"], carry["current_labels"])
        puzzle_ids = jnp.where(reset, batch["puzzle_identifiers"], carry["current_puzzle_identifiers"])
        hidden = jnp.where(reset_hidden, self._reset_hidden(inputs.shape[0]), carry["hidden"])
        steps = jnp.where(reset, 0, carry["steps"])

        input_embeddings = self._input_embeddings(inputs, puzzle_ids, puzzle_embeddings)
        hidden = self._inner_update(hidden, input_embeddings, train=train, dropout_key=dropout_key)
        logits, halt_logits = self._logits_and_halt(hidden)

        new_steps = steps + 1
        is_last_step = new_steps >= self.recurrent_steps
        if train:
            halted = is_last_step | (halt_logits > 0.0)
            if self.recurrent_steps > 1 and self.urm.halt_exploration_prob > 0.0:
                explore_key, min_step_key = jax.random.split(dropout_key)
                use_random_min = jax.random.uniform(explore_key, (inputs.shape[0],)) < self.urm.halt_exploration_prob
                random_min_steps = jax.random.randint(
                    min_step_key,
                    (inputs.shape[0],),
                    minval=2,
                    maxval=self.recurrent_steps + 1,
                )
                min_steps = jnp.where(use_random_min, random_min_steps, 1)
                halted = halted & (new_steps >= min_steps)
        else:
            halted = is_last_step

        new_carry = {
            "hidden": jax.lax.stop_gradient(hidden),
            "steps": jax.lax.stop_gradient(new_steps),
            "halted": jax.lax.stop_gradient(halted),
            "current_inputs": inputs,
            "current_labels": labels,
            "current_puzzle_identifiers": puzzle_ids,
        }
        diagnostics = {
            "halt_logits": halt_logits,
            "act_step": jnp.mean(new_steps.astype(jnp.float32)),
            "halted_rate": jnp.mean(halted.astype(jnp.float32)),
            "reset_rate": jnp.mean(reset.astype(jnp.float32)),
        }
        return new_carry, logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        hidden = self._reset_hidden(tokens.shape[0])
        input_embeddings = self._input_embeddings(tokens, puzzle_identifiers)
        dropout_keys = jax.random.split(dropout_key, self.recurrent_steps)

        def scan_step(h: Array, step_key: Array) -> tuple[Array, tuple[Array, Array, Array]]:
            prev_h = h if collect_diagnostics else None
            h = self._inner_update(h, input_embeddings, train=train, dropout_key=step_key)
            logits, halt_logits = self._logits_and_halt(h)
            if collect_diagnostics:
                assert prev_h is not None
                hidden_delta = jnp.mean(jnp.linalg.norm((h - prev_h).astype(jnp.float32), axis=-1))
            else:
                hidden_delta = jnp.asarray(0.0, dtype=jnp.float32)
            return h, (logits, halt_logits, hidden_delta)

        _hidden, (step_logits, halt_logits, hidden_delta) = jax.lax.scan(scan_step, hidden, dropout_keys)
        diagnostics = {
            "halt_logits": halt_logits,
            "hidden_delta_mean": hidden_delta,
            "unroll_steps": jnp.asarray(self.recurrent_steps, dtype=jnp.int32),
        }
        return step_logits, diagnostics

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
    ) -> Array:
        step_logits, _diagnostics = self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            puzzle_identifiers=puzzle_identifiers,
        )
        return step_logits[-1]

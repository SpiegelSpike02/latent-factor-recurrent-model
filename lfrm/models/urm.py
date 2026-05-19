from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, CastedEmbedding, casted_linear_init, compute_dtype, maybe_cast, trunc_normal, trunc_normal_init


def _rms_norm(x: Array, eps: float) -> Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


def _rotate_half(x: Array) -> Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def _swiglu_intermediate_size(hidden_size: int, expansion: int) -> int:
    rough = round(expansion * hidden_size * 2 / 3)
    return max(256, 256 * math.ceil(rough / 256))


class URMAttention(nnx.Module):
    """Official-style full attention with RoPE positional encoding."""

    def __init__(self, d_model: int, num_heads: int, dtype: jnp.dtype, *, rngs: nnx.Rngs) -> None:
        if d_model % num_heads != 0:
            raise ValueError("URM d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dtype = dtype

        self.qkv = nnx.Linear(
            d_model,
            3 * d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            d_model,
            d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, h: Array, rope_cos: Array, rope_sin: Array) -> Array:
        batch_size, seq_len, d_model = h.shape
        qkv = self.qkv(maybe_cast(h, self.dtype))
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        cos = rope_cos[None, None, :seq_len, :].astype(q.dtype)
        sin = rope_sin[None, None, :seq_len, :].astype(q.dtype)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        scores = jnp.einsum("bhnd,bhmd->bhnm", q, k, preferred_element_type=jnp.float32)
        scores = scores / math.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(h.dtype)
        output = jnp.einsum("bhnm,bhmd->bhnd", weights, v, preferred_element_type=jnp.float32)
        output = jnp.swapaxes(output, 1, 2).reshape(batch_size, seq_len, d_model)
        return self.out(output)


class ConvSwiGLU(nnx.Module):
    """URM feed-forward block: SwiGLU, sequence depthwise conv, SiLU, down projection."""

    def __init__(self, config: ModelConfig, dtype: jnp.dtype, *, rngs: nnx.Rngs) -> None:
        urm = config.urm_config
        if urm.conv_kernel < 1:
            raise ValueError("URM conv_kernel must be positive")
        self.dtype = dtype
        hidden_size = config.d_model
        intermediate_size = _swiglu_intermediate_size(hidden_size, urm.mlp_ratio)
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
            preferred_element_type=jnp.float32,
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
        self.attention = URMAttention(config.d_model, config.urm_config.num_heads, dtype, rngs=rngs)
        self.conv_swiglu = ConvSwiGLU(config, dtype, rngs=rngs)

    def __call__(self, h: Array, rope_cos: Array, rope_sin: Array) -> Array:
        attn_output = self.attention(maybe_cast(h, self.dtype), rope_cos, rope_sin)
        h = _rms_norm(h + attn_output.astype(h.dtype), self.norm_eps)
        mlp_output = self.conv_swiglu(maybe_cast(h, self.dtype))
        return _rms_norm(h + mlp_output.astype(h.dtype), self.norm_eps)


class UnifiedReasoningModel(nnx.Module):
    """URM aligned to the official loops/H_cycles/L_cycles ACT formulation."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        if config.model_type != "urm":
            raise ValueError("UnifiedReasoningModel requires model_type='urm'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        urm = config.urm_config
        if min(urm.recurrent_steps, urm.deep_recursion, urm.latent_recursion, urm.block_layers) < 1:
            raise ValueError("URM recurrent/deep/latent/block counts must be positive")
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
        self.puzzle_emb_len = max(0, int(urm.puzzle_emb_len))
        self.total_seq_len = config.seq_len + self.puzzle_emb_len

        self.token_embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=trunc_normal_init(1.0 / self.embed_scale),
            rngs=rngs,
        )
        self.has_puzzle_embed = self.puzzle_emb_len > 0
        if self.puzzle_emb_len > 0:
            puzzle_emb_ndim = self.puzzle_emb_len * config.d_model
            if urm.puzzle_emb_ndim > 0:
                puzzle_emb_ndim = urm.puzzle_emb_ndim
            if puzzle_emb_ndim != self.puzzle_emb_len * config.d_model:
                raise ValueError("URM puzzle_emb_ndim must equal puzzle_emb_len * d_model in this JAX port")
            self.puzzle_embed = CastedEmbedding(
                config.num_puzzle_identifiers,
                puzzle_emb_ndim,
                self.dtype,
                init_std=0.0,
                rngs=rngs,
            )

        self.blocks = nnx.List([URMBlock(config, self.dtype, rngs=rngs) for _ in range(urm.block_layers)])
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

        init_hidden = trunc_normal(rngs.params(), (config.d_model,), std=1.0, dtype=jnp.float32)
        self.init_hidden = nnx.Param(init_hidden)

    def _input_embeddings(self, tokens: Array, puzzle_identifiers: Array | None) -> Array:
        token_emb = self.token_embed(tokens.astype(jnp.int32)).astype(self.dtype) * self.embed_scale
        if not self.has_puzzle_embed:
            return token_emb
        if puzzle_identifiers is None:
            puzzle_identifiers = jnp.zeros((tokens.shape[0],), dtype=jnp.int32)
        if not hasattr(self, "puzzle_embed"):
            raise ValueError("URM puzzle embedding table is not initialized")
        prefix = self.puzzle_embed(puzzle_identifiers, train=True)
        prefix = prefix.reshape(tokens.shape[0], self.puzzle_emb_len, self.config.d_model)
        return jnp.concatenate((prefix, token_emb), axis=1)

    def _run_layers(self, hidden: Array, input_embeddings: Array) -> Array:
        hidden = hidden + input_embeddings
        for block in self.blocks:
            hidden = block(hidden, self.rope_cos, self.rope_sin)
        return hidden

    def _inner_update(self, hidden: Array, input_embeddings: Array, *, train: bool, dropout_key: Array) -> Array:
        h = hidden
        for _ in range(self.urm.deep_recursion - 1):
            for _ in range(self.urm.latent_recursion):
                h = self._run_layers(h, input_embeddings)
            h = jax.lax.stop_gradient(h)

        for _ in range(self.urm.latent_recursion):
            h = self._run_layers(h, input_embeddings)
        h = self.dropout(h, deterministic=not train, rngs=dropout_key)
        return h

    def _logits_and_halt(self, hidden: Array) -> tuple[Array, Array]:
        visible_hidden = hidden[:, self.puzzle_emb_len :, :]
        logits = self.output_head(maybe_cast(visible_hidden, self.dtype)).astype(jnp.float32)
        halt_hidden = hidden[:, 0, :] if self.puzzle_emb_len > 0 else visible_hidden[:, 0, :]
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
            "current_given_mask": jnp.zeros_like(batch["given_mask"]),
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
        del puzzle_embeddings
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        reset = carry["halted"]
        reset_broadcast = reset[:, None]
        reset_hidden = reset[:, None, None]
        inputs = jnp.where(reset_broadcast, batch["inputs"], carry["current_inputs"])
        labels = jnp.where(reset_broadcast, batch["labels"], carry["current_labels"])
        given_mask = jnp.where(reset_broadcast, batch["given_mask"], carry["current_given_mask"])
        puzzle_ids = jnp.where(reset, batch["puzzle_identifiers"], carry["current_puzzle_identifiers"])
        hidden = jnp.where(reset_hidden, self._reset_hidden(inputs.shape[0]), carry["hidden"])
        steps = jnp.where(reset, 0, carry["steps"])

        input_embeddings = self._input_embeddings(inputs, puzzle_ids)
        previous_hidden = hidden
        hidden = self._inner_update(hidden, input_embeddings, train=train, dropout_key=dropout_key)
        logits, halt_logits = self._logits_and_halt(hidden)

        new_steps = steps + 1
        halted = new_steps >= self.recurrent_steps
        if train and self.recurrent_steps > 1 and self.urm.halt_exploration_prob > 0.0:
            explore_key, min_step_key = jax.random.split(dropout_key)
            use_random_min = jax.random.uniform(explore_key, (inputs.shape[0],)) < self.urm.halt_exploration_prob
            random_min_steps = jax.random.randint(
                min_step_key,
                (inputs.shape[0],),
                minval=2,
                maxval=self.recurrent_steps + 1,
            )
            min_steps = jnp.where(use_random_min, random_min_steps, 1)
            halted = halted | ((halt_logits > 0.0) & (new_steps >= min_steps))

        new_carry = {
            "hidden": jax.lax.stop_gradient(hidden),
            "steps": new_steps,
            "halted": halted,
            "current_inputs": inputs,
            "current_labels": labels,
            "current_given_mask": given_mask,
            "current_puzzle_identifiers": puzzle_ids,
        }
        hidden_delta = jnp.mean(jnp.linalg.norm((hidden - previous_hidden).astype(jnp.float32), axis=-1))
        diagnostics = {
            "halt_logits": halt_logits,
            "act_step": jnp.mean(new_steps.astype(jnp.float32)),
            "halted_rate": jnp.mean(halted.astype(jnp.float32)),
            "reset_rate": jnp.mean(reset.astype(jnp.float32)),
            "hidden_delta_mean": hidden_delta,
        }
        return new_carry, logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        hidden = self._reset_hidden(tokens.shape[0])
        input_embeddings = self._input_embeddings(tokens, puzzle_identifiers)
        dropout_keys = jax.random.split(dropout_key, self.recurrent_steps)

        def scan_step(h: Array, step_key: Array) -> tuple[Array, tuple[Array, Array, Array]]:
            prev_h = h
            h = self._inner_update(h, input_embeddings, train=train, dropout_key=step_key)
            logits, halt_logits = self._logits_and_halt(h)
            hidden_delta = jnp.mean(jnp.linalg.norm((h - prev_h).astype(jnp.float32), axis=-1))
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

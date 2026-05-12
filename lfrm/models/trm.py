from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, CastedEmbedding, casted_linear_init, compute_dtype, maybe_cast, trunc_normal, trunc_normal_init


State = tuple[Array, Array]
Carry = dict[str, Array]


def _rms_norm(x: Array, eps: float) -> Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


def _swiglu_intermediate_size(hidden_size: int, expansion: float) -> int:
    raw = round(expansion * hidden_size * 2 / 3)
    return max(1, math.ceil(raw / 256) * 256)


class SwiGLU(nnx.Module):
    def __init__(self, hidden_size: int, expansion: float, dtype: jnp.dtype, *, rngs: nnx.Rngs) -> None:
        intermediate_size = _swiglu_intermediate_size(hidden_size, expansion)
        self.gate_up = nnx.Linear(
            hidden_size,
            2 * intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
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

    def __call__(self, x: Array) -> Array:
        gate, up = jnp.split(self.gate_up(x), 2, axis=-1)
        return self.down(jax.nn.silu(gate) * up)


class LocalConvSwiGLU(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        prefix_len: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        trm = config.trm_config
        if trm.local_mixing_kernel < 1 or trm.local_mixing_kernel % 2 == 0:
            raise ValueError("TRM local_mixing_kernel must be a positive odd integer")
        self.config = config
        self.prefix_len = prefix_len
        self.dtype = dtype
        self.depthwise = nnx.Conv(
            config.d_model,
            config.d_model,
            kernel_size=(trm.local_mixing_kernel, trm.local_mixing_kernel),
            padding="SAME",
            feature_group_count=config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=jnp.float32,
            rngs=rngs,
        )
        self.gate_up = nnx.Linear(
            config.d_model,
            2 * config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            config.d_model,
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        grid_tokens = hidden_states[:, self.prefix_len :, :]
        batch_size = grid_tokens.shape[0]
        grid = grid_tokens.reshape(
            batch_size,
            self.config.grid_height,
            self.config.grid_width,
            self.config.d_model,
        )
        local = self.depthwise(maybe_cast(grid, self.dtype))
        local = local.reshape(batch_size, self.config.seq_len, self.config.d_model)
        gate, up = jnp.split(self.gate_up(local), 2, axis=-1)
        mixed = self.down(jax.nn.silu(gate) * up)
        if self.prefix_len == 0:
            return mixed
        prefix = jnp.zeros_like(hidden_states[:, : self.prefix_len, :])
        return jnp.concatenate([prefix, mixed], axis=1)


class Attention(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        prefix_len: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        d_model = config.d_model
        trm = config.trm_config
        num_heads = trm.num_heads
        if d_model % num_heads != 0:
            raise ValueError("TRM d_model must be divisible by trm.num_heads")
        if trm.attention_type not in ("standard", "gated_dual"):
            raise ValueError("TRM attention_type must be one of: standard, gated_dual")
        self.config = config
        self.trm = trm
        self.prefix_len = prefix_len
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nnx.Linear(
            d_model,
            3 * d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        if trm.attention_type == "gated_dual":
            self.dual_gate = nnx.Linear(
                d_model,
                num_heads,
                dtype=dtype,
                param_dtype=jnp.float32,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            self.local_distance_logit = nnx.Param(jnp.zeros((num_heads,), dtype=jnp.float32))
            rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
            cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
            distance = jnp.abs(rows[:, None] - rows[None, :]) + jnp.abs(cols[:, None] - cols[None, :])
            distance = distance.astype(jnp.float32)
            if prefix_len > 0:
                padded = jnp.zeros((prefix_len + config.seq_len, prefix_len + config.seq_len), dtype=jnp.float32)
                distance = padded.at[prefix_len:, prefix_len:].set(distance)
            self.local_distance = nnx.data(distance)
        self.out = nnx.Linear(
            d_model,
            d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
        attention_bias: Array | None,
    ) -> tuple[Array, dict[str, Array]]:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        if rope_cos is not None and rope_sin is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32)
        scores = scores / math.sqrt(self.head_dim)
        if self.trm.attention_type == "gated_dual":
            global_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
            global_attended = jnp.einsum(
                "bhqk,bhkd->bhqd",
                global_weights,
                v,
                preferred_element_type=jnp.float32,
            )
            distance_scale = jax.nn.softplus(self.local_distance_logit)[None, :, None, None]
            local_scores = scores - distance_scale * self.local_distance[None, None, :, :]
            if attention_bias is not None:
                local_scores = local_scores + attention_bias[None, :, :, :].astype(jnp.float32)
            local_weights = jax.nn.softmax(local_scores.astype(jnp.float32), axis=-1).astype(x.dtype)
            local_attended = jnp.einsum(
                "bhqk,bhkd->bhqd",
                local_weights,
                v,
                preferred_element_type=jnp.float32,
            )
            gate = jax.nn.sigmoid(self.dual_gate(x)).astype(jnp.float32)
            diagnostics = {
                "attention_gate_mean": jnp.mean(gate),
                "attention_gate_std": jnp.std(gate),
                "attention_gate_low_saturation": jnp.mean((gate < 0.05).astype(jnp.float32)),
                "attention_gate_high_saturation": jnp.mean((gate > 0.95).astype(jnp.float32)),
                "attention_local_distance_scale": jnp.mean(jax.nn.softplus(self.local_distance_logit)),
            }
            gate = jnp.swapaxes(gate.astype(x.dtype), 1, 2)[..., None]
            attended = gate * global_attended + (1.0 - gate) * local_attended
            attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, d_model)
            return self.out(attended), diagnostics
        if attention_bias is not None:
            scores = scores + attention_bias[None, :, :, :].astype(jnp.float32)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v, preferred_element_type=jnp.float32)
        attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, d_model)
        return self.out(attended), {
            "attention_gate_mean": jnp.asarray(1.0, dtype=jnp.float32),
            "attention_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_low_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_high_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_local_distance_scale": jnp.asarray(0.0, dtype=jnp.float32),
        }


def _rotate_half(x: Array) -> Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-second, first], axis=-1)


def _apply_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    return (
        (q_f32 * cos + _rotate_half(q_f32) * sin).astype(q.dtype),
        (k_f32 * cos + _rotate_half(k_f32) * sin).astype(k.dtype),
    )


class TRMBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        sequence_length: int,
        prefix_len: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        trm = config.trm_config
        self.config = config
        self.trm = trm
        if trm.mlp_t:
            self.token_mlp = SwiGLU(sequence_length, trm.mlp_ratio, dtype, rngs=rngs)
        else:
            self.attention = Attention(config, prefix_len, dtype, rngs=rngs)
        if trm.local_mixing:
            self.local_mixing = LocalConvSwiGLU(config, prefix_len, dtype, rngs=rngs)
        self.channel_mlp = SwiGLU(config.d_model, trm.mlp_ratio, dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
        attention_bias: Array | None,
    ) -> tuple[Array, dict[str, Array]]:
        diagnostics = {
            "attention_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_low_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_high_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_local_distance_scale": jnp.asarray(0.0, dtype=jnp.float32),
        }
        if self.trm.mlp_t:
            mixed = jnp.swapaxes(hidden_states, 1, 2)
            mixed = _rms_norm(mixed + self.token_mlp(mixed), self.trm.rms_norm_eps)
            hidden_states = jnp.swapaxes(mixed, 1, 2)
        else:
            attention_out, diagnostics = self.attention(hidden_states, rope_cos, rope_sin, attention_bias)
            hidden_states = _rms_norm(
                hidden_states + attention_out,
                self.trm.rms_norm_eps,
            )
        if self.trm.local_mixing:
            hidden_states = _rms_norm(
                hidden_states + self.local_mixing(hidden_states),
                self.trm.rms_norm_eps,
            )
        hidden_states = _rms_norm(
            hidden_states + self.channel_mlp(hidden_states),
            self.trm.rms_norm_eps,
        )
        return hidden_states, diagnostics


class TinyRecursiveModel(nnx.Module):
    """Official-style Tiny Recursive Model baseline.

    The model keeps two recurrent token states, z_H and z_L. Each ACT step runs
    several H/L cycles with a shared low-level block, detaches the carry between
    ACT steps, and reads token logits plus a halt-quality head from z_H.
    """

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.model_type != "trm":
            raise ValueError("TinyRecursiveModel requires model_type='trm'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.num_steps < 1:
            raise ValueError("TRM num_steps must be at least 1")
        trm = config.trm_config
        if trm.l_layers < 1:
            raise ValueError("TRM l_layers must be at least 1")
        if trm.h_cycles < 1:
            raise ValueError("TRM h_cycles must be at least 1")
        if trm.l_cycles < 1:
            raise ValueError("TRM l_cycles must be at least 1")
        if trm.mlp_ratio < 1:
            raise ValueError("TRM mlp_ratio must be at least 1")
        if trm.attention_type not in ("standard", "gated_dual"):
            raise ValueError("TRM attention_type must be one of: standard, gated_dual")
        if trm.local_mixing_kernel < 1 or trm.local_mixing_kernel % 2 == 0:
            raise ValueError("TRM local_mixing_kernel must be a positive odd integer")
        if trm.puzzle_emb_ndim < 0:
            raise ValueError("TRM puzzle_emb_ndim must be non-negative")
        if trm.puzzle_emb_len < 0:
            raise ValueError("TRM puzzle_emb_len must be non-negative")
        if config.num_puzzle_identifiers < 1:
            raise ValueError("TRM num_puzzle_identifiers must be at least 1")
        if trm.pos_encodings not in ("none", "learned", "rope", "grid", "rel2d"):
            raise ValueError("TRM pos_encodings must be one of: none, learned, rope, grid, rel2d")
        if trm.pos_encodings == "rope" and config.d_model % trm.num_heads != 0:
            raise ValueError("TRM d_model must be divisible by trm.num_heads for RoPE")
        if trm.pos_encodings == "rope" and (config.d_model // trm.num_heads) % 2 != 0:
            raise ValueError("TRM RoPE head dimension must be even")
        if trm.step_loss_weights is not None:
            if len(trm.step_loss_weights) != config.num_steps:
                raise ValueError("TRM step_loss_weights length must equal num_steps")
            if any(weight < 0.0 for weight in trm.step_loss_weights):
                raise ValueError("TRM step_loss_weights must be non-negative")
            if sum(trm.step_loss_weights) <= 0.0:
                raise ValueError("TRM step_loss_weights must contain a positive weight")

        self.config = config
        self.runtime = runtime
        self.trm = trm
        dtype = compute_dtype(runtime.compute_dtype)
        self.dtype = dtype
        self.embed_scale = math.sqrt(config.d_model)
        puzzle_emb_ndim = trm.puzzle_emb_ndim
        if puzzle_emb_ndim == 0 and trm.puzzle_emb_len > 0:
            puzzle_emb_ndim = config.d_model
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.prefix_len = (
            math.ceil(puzzle_emb_ndim / config.d_model)
            if trm.puzzle_emb_len == 0 and puzzle_emb_ndim > 0
            else trm.puzzle_emb_len
        )
        if puzzle_emb_ndim == 0:
            self.prefix_len = 0
        if puzzle_emb_ndim > self.prefix_len * config.d_model:
            raise ValueError("TRM puzzle_emb_ndim must fit within puzzle_emb_len * d_model")
        self.total_seq_len = config.seq_len + self.prefix_len
        self.position_ids = jnp.arange(self.total_seq_len, dtype=jnp.int32)

        embed_init = trunc_normal_init(1.0 / self.embed_scale)
        self.token_embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        if self.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedEmbedding(
                config.num_puzzle_identifiers,
                self.puzzle_emb_ndim,
                dtype,
                init_std=0.0,
                rngs=rngs,
            )
        if trm.pos_encodings == "learned":
            self.position_embed = nnx.Embed(
                self.total_seq_len,
                config.d_model,
                dtype=dtype,
                param_dtype=jnp.float32,
                embedding_init=embed_init,
                rngs=rngs,
            )
        if trm.pos_encodings == "grid":
            box_height, box_width = self._box_shape(config.grid_height, config.grid_width)
            rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
            cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
            boxes = (rows // box_height) * (config.grid_width // box_width) + (cols // box_width)
            self.row_ids = nnx.data(rows)
            self.col_ids = nnx.data(cols)
            self.box_ids = nnx.data(boxes)
            self.row_embed = nnx.Embed(
                config.grid_height,
                config.d_model,
                dtype=dtype,
                param_dtype=jnp.float32,
                embedding_init=embed_init,
                rngs=rngs,
            )
            self.col_embed = nnx.Embed(
                config.grid_width,
                config.d_model,
                dtype=dtype,
                param_dtype=jnp.float32,
                embedding_init=embed_init,
                rngs=rngs,
            )
            self.box_embed = nnx.Embed(
                (config.grid_height // box_height) * (config.grid_width // box_width),
                config.d_model,
                dtype=dtype,
                param_dtype=jnp.float32,
                embedding_init=embed_init,
                rngs=rngs,
            )
        if trm.pos_encodings == "rel2d":
            rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
            cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
            rel_rows = rows[:, None] - rows[None, :] + config.grid_height - 1
            rel_cols = cols[:, None] - cols[None, :] + config.grid_width - 1
            self.rel2d_row_bias = nnx.Param(
                jnp.zeros((trm.num_heads, 2 * config.grid_height - 1), dtype=jnp.float32)
            )
            self.rel2d_col_bias = nnx.Param(
                jnp.zeros((trm.num_heads, 2 * config.grid_width - 1), dtype=jnp.float32)
            )
            if self.prefix_len > 0:
                padded_rows = jnp.full((self.total_seq_len, self.total_seq_len), -1, dtype=jnp.int32)
                padded_cols = jnp.full((self.total_seq_len, self.total_seq_len), -1, dtype=jnp.int32)
                rel_rows = padded_rows.at[self.prefix_len :, self.prefix_len :].set(rel_rows)
                rel_cols = padded_cols.at[self.prefix_len :, self.prefix_len :].set(rel_cols)
            self.rel2d_row_indices = nnx.data(rel_rows)
            self.rel2d_col_indices = nnx.data(rel_cols)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.blocks = nnx.List(
            [
                TRMBlock(config, self.total_seq_len, self.prefix_len, dtype, rngs=rngs)
                for _ in range(trm.l_layers)
            ]
        )
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.q_head = nnx.Linear(
            config.d_model,
            2,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(-5.0),
            rngs=rngs,
        )
        self.h_init = nnx.data(trunc_normal(rngs.params(), (config.d_model,), 1.0, dtype))
        self.l_init = nnx.data(trunc_normal(rngs.params(), (config.d_model,), 1.0, dtype))

        if trm.pos_encodings == "rope":
            head_dim = config.d_model // trm.num_heads
            inv_freq = 1.0 / (trm.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
            freqs = jnp.outer(jnp.arange(self.total_seq_len, dtype=jnp.float32), inv_freq)
            rope = jnp.concatenate([freqs, freqs], axis=-1)
            self.rope_cos = jnp.cos(rope)
            self.rope_sin = jnp.sin(rope)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def _attention_bias(self) -> Array | None:
        if self.trm.pos_encodings != "rel2d":
            return None
        row_valid = self.rel2d_row_indices >= 0
        col_valid = self.rel2d_col_indices >= 0
        row_indices = jnp.maximum(self.rel2d_row_indices, 0)
        col_indices = jnp.maximum(self.rel2d_col_indices, 0)
        row_bias = jnp.take(self.rel2d_row_bias[...], row_indices, axis=1)
        col_bias = jnp.take(self.rel2d_col_bias[...], col_indices, axis=1)
        rel2d_bias = row_bias + col_bias
        return jnp.where((row_valid & col_valid)[None, :, :], rel2d_bias, 0.0)

    @staticmethod
    def _box_shape(grid_height: int, grid_width: int) -> tuple[int, int]:
        box_height = int(math.sqrt(grid_height))
        box_width = int(math.sqrt(grid_width))
        if box_height < 1 or box_width < 1 or grid_height % box_height != 0 or grid_width % box_width != 0:
            raise ValueError("TRM grid position encoding requires factorable grid dimensions")
        return box_height, box_width

    def _condition_mask(self, tokens: Array) -> Array:
        return tokens != 1

    def _batch_puzzle_identifiers(self, batch: dict[str, Array]) -> Array:
        return batch["puzzle_identifiers"].astype(jnp.int32)

    def _input_embeddings(
        self,
        tokens: Array,
        puzzle_identifiers: Array | None,
        *,
        train: bool,
        dropout_key: Array | None,
        puzzle_embeddings: Array | None = None,
    ) -> Array:
        embedding = self.token_embed(tokens.astype(jnp.int32))
        if self.puzzle_emb_ndim > 0:
            if puzzle_embeddings is not None:
                prefix = puzzle_embeddings
            elif puzzle_identifiers is None:
                raise ValueError("TRM requires puzzle_identifiers when puzzle_emb_len > 0")
            else:
                prefix = self.puzzle_emb(puzzle_identifiers, train=train)
            pad_count = self.prefix_len * self.config.d_model - self.puzzle_emb_ndim
            if pad_count > 0:
                prefix = jnp.pad(prefix, ((0, 0), (0, pad_count)))
            prefix = prefix.reshape(tokens.shape[0], self.prefix_len, self.config.d_model)
            embedding = jnp.concatenate([prefix.astype(embedding.dtype), embedding], axis=1)
        if self.trm.pos_encodings == "learned":
            embedding = (embedding + self.position_embed(self.position_ids)[None, :, :]) * math.sqrt(0.5)
        elif self.trm.pos_encodings == "grid":
            grid_position = (
                self.row_embed(self.row_ids)
                + self.col_embed(self.col_ids)
                + self.box_embed(self.box_ids)
            ) * math.sqrt(1.0 / 3.0)
            if self.prefix_len > 0:
                prefix_position = jnp.zeros((self.prefix_len, self.config.d_model), dtype=grid_position.dtype)
                position = jnp.concatenate([prefix_position, grid_position], axis=0)
            else:
                position = grid_position
            embedding = (embedding + position[None, :, :]) * math.sqrt(0.5)
        embedding = self.embed_scale * embedding
        return self.dropout(embedding, deterministic=not train, rngs=dropout_key).astype(self.dtype)

    def _initial_state(self, batch_size: int) -> State:
        h_init = jnp.broadcast_to(self.h_init.astype(self.dtype), (batch_size, self.total_seq_len, self.config.d_model))
        l_init = jnp.broadcast_to(self.l_init.astype(self.dtype), (batch_size, self.total_seq_len, self.config.d_model))
        return h_init, l_init

    def initial_carry(self, batch: dict[str, Array]) -> Carry:
        batch_size = batch["inputs"].shape[0]
        z_h, z_l = self._initial_state(batch_size)
        puzzle_identifiers = self._batch_puzzle_identifiers(batch)
        return {
            "z_h": z_h,
            "z_l": z_l,
            "steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "halted": jnp.ones((batch_size,), dtype=bool),
            "current_inputs": jnp.zeros_like(batch["inputs"]),
            "current_labels": jnp.zeros_like(batch["labels"]),
            "current_given_mask": jnp.zeros_like(batch["given_mask"]),
            "current_puzzle_identifiers": jnp.zeros_like(puzzle_identifiers),
        }

    def _reset_carry_state(self, carry: Carry) -> State:
        z_h_init, z_l_init = self._initial_state(carry["steps"].shape[0])
        reset = carry["halted"][:, None, None]
        z_h = jnp.where(reset, z_h_init, carry["z_h"])
        z_l = jnp.where(reset, z_l_init, carry["z_l"])
        return z_h, z_l

    def _current_batch(self, carry: Carry, batch: dict[str, Array]) -> dict[str, Array]:
        reset = carry["halted"]
        return {
            "inputs": jnp.where(reset[:, None], batch["inputs"], carry["current_inputs"]),
            "labels": jnp.where(reset[:, None], batch["labels"], carry["current_labels"]),
            "given_mask": jnp.where(reset[:, None], batch["given_mask"], carry["current_given_mask"]),
            "puzzle_identifiers": jnp.where(
                reset,
                self._batch_puzzle_identifiers(batch),
                carry["current_puzzle_identifiers"],
            ),
        }

    def _empty_attention_diagnostics(self) -> dict[str, Array]:
        return {
            "attention_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_low_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_gate_high_saturation": jnp.asarray(0.0, dtype=jnp.float32),
            "attention_local_distance_scale": jnp.asarray(0.0, dtype=jnp.float32),
        }

    @staticmethod
    def _add_attention_diagnostics(lhs: dict[str, Array], rhs: dict[str, Array]) -> dict[str, Array]:
        return {key: lhs[key] + rhs[key] for key in lhs}

    @staticmethod
    def _scale_attention_diagnostics(diagnostics: dict[str, Array], normalizer: float | Array) -> dict[str, Array]:
        return {key: value / normalizer for key, value in diagnostics.items()}

    def _l_level(self, hidden_states: Array, input_injection: Array) -> tuple[Array, dict[str, Array]]:
        hidden_states = hidden_states + input_injection
        attention_bias = self._attention_bias()
        diagnostics = self._empty_attention_diagnostics()
        for block in self.blocks:
            hidden_states, block_diagnostics = block(
                maybe_cast(hidden_states, self.dtype),
                self.rope_cos,
                self.rope_sin,
                attention_bias,
            )
            diagnostics = self._add_attention_diagnostics(diagnostics, block_diagnostics)
        normalizer = jnp.maximum(jnp.asarray(len(self.blocks), dtype=jnp.float32), 1.0)
        return hidden_states, self._scale_attention_diagnostics(diagnostics, normalizer)

    def _h_cycle(self, state: State, input_embeddings: Array) -> tuple[State, dict[str, Array]]:
        z_h, z_l = state
        diagnostics = self._empty_attention_diagnostics()
        for _ in range(self.trm.l_cycles):
            z_l, l_diagnostics = self._l_level(z_l, z_h + input_embeddings)
            z_h, h_diagnostics = self._l_level(z_h, z_l)
            diagnostics = self._add_attention_diagnostics(diagnostics, l_diagnostics)
            diagnostics = self._add_attention_diagnostics(diagnostics, h_diagnostics)
        normalizer = jnp.asarray(2 * self.trm.l_cycles, dtype=jnp.float32)
        return (z_h, z_l), self._scale_attention_diagnostics(diagnostics, normalizer)

    def _act_step(self, state: State, input_embeddings: Array) -> tuple[State, dict[str, Array]]:
        z_h, z_l = state
        diagnostics = self._empty_attention_diagnostics()
        for _ in range(self.trm.h_cycles - 1):
            (z_h, z_l), cycle_diagnostics = self._h_cycle((z_h, z_l), input_embeddings)
            diagnostics = self._add_attention_diagnostics(diagnostics, cycle_diagnostics)
            z_h = jax.lax.stop_gradient(z_h)
            z_l = jax.lax.stop_gradient(z_l)
        (z_h, z_l), cycle_diagnostics = self._h_cycle((z_h, z_l), input_embeddings)
        diagnostics = self._add_attention_diagnostics(diagnostics, cycle_diagnostics)
        normalizer = jnp.asarray(self.trm.h_cycles, dtype=jnp.float32)
        return (z_h, z_l), self._scale_attention_diagnostics(diagnostics, normalizer)

    def _blank_mean(self, values: Array, condition_mask: Array) -> Array:
        mask = (~condition_mask).astype(jnp.float32)[..., None]
        trailing_size = math.prod(values.shape[2:]) if len(values.shape) > 2 else 1
        normalizer = jnp.maximum(jnp.sum(mask) * trailing_size, 1.0)
        return jnp.sum(values * mask) / normalizer

    def _quality_logits(self, z_h: Array) -> Array:
        q_logits = self.q_head(maybe_cast(z_h[:, 0], self.dtype)).astype(jnp.float32)
        return q_logits[..., 0], q_logits[..., 1]

    def _logits_from_state(self, z_h: Array) -> Array:
        return self.lm_head(maybe_cast(z_h, self.dtype)).astype(jnp.float32)

    def forward_act_step(
        self,
        carry: Carry,
        batch: dict[str, Array],
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_embeddings: Array | None = None,
    ) -> tuple[Carry, Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        dropout_key, exploration_key, min_step_key = jax.random.split(dropout_key, 3)
        current_batch = self._current_batch(carry, batch)
        inputs = current_batch["inputs"]
        condition_mask = self._condition_mask(inputs)
        input_embeddings = self._input_embeddings(
            inputs,
            current_batch["puzzle_identifiers"],
            train=train,
            dropout_key=dropout_key,
            puzzle_embeddings=puzzle_embeddings,
        )
        state = self._reset_carry_state(carry)
        z_h_prev = state[0][:, self.prefix_len :]
        z_l_prev = state[1][:, self.prefix_len :]
        next_state, attention_diagnostics = self._act_step(state, input_embeddings)
        z_h_next, z_l_next = next_state
        z_h_data = z_h_next[:, self.prefix_len :]
        z_l_data = z_l_next[:, self.prefix_len :]
        logits = self._logits_from_state(z_h_data)
        q_halt_logits, q_continue_logits = self._quality_logits(z_h_next)

        new_steps = jnp.where(carry["halted"], 0, carry["steps"]) + 1
        is_last_step = new_steps >= self.config.num_steps
        if train:
            if self.trm.no_act_continue:
                halted = jnp.logical_or(is_last_step, q_halt_logits > 0.0)
            else:
                halted = jnp.logical_or(is_last_step, q_halt_logits > q_continue_logits)
            if self.config.num_steps > 1 and self.trm.halt_exploration_prob > 0.0:
                explore = jax.random.uniform(exploration_key, q_halt_logits.shape) < self.trm.halt_exploration_prob
                min_steps = jax.random.randint(
                    min_step_key,
                    new_steps.shape,
                    minval=2,
                    maxval=self.config.num_steps + 1,
                    dtype=jnp.int32,
                )
                min_steps = jnp.where(explore, min_steps, 0)
                halted = jnp.logical_and(halted, new_steps >= min_steps)
        else:
            halted = is_last_step

        h_delta = jnp.linalg.norm((z_h_data - z_h_prev).astype(jnp.float32), axis=-1, keepdims=True)
        l_delta = jnp.linalg.norm((z_l_data - z_l_prev).astype(jnp.float32), axis=-1, keepdims=True)
        hidden_delta = self._blank_mean(h_delta, condition_mask)
        low_hidden_delta = self._blank_mean(l_delta, condition_mask)
        new_carry = {
            "z_h": jax.lax.stop_gradient(z_h_next),
            "z_l": jax.lax.stop_gradient(z_l_next),
            "steps": jax.lax.stop_gradient(new_steps),
            "halted": jax.lax.stop_gradient(halted),
            "current_inputs": current_batch["inputs"],
            "current_labels": current_batch["labels"],
            "current_given_mask": current_batch["given_mask"],
            "current_puzzle_identifiers": current_batch["puzzle_identifiers"],
        }
        diagnostics = {
            "hidden_delta_mean": hidden_delta,
            "h_hidden_delta_mean": hidden_delta,
            "l_hidden_delta_mean": low_hidden_delta,
            "quality_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "act_step": jnp.mean(new_steps.astype(jnp.float32)),
            "halted_rate": jnp.mean(halted.astype(jnp.float32)),
            "reset_rate": jnp.mean(carry["halted"].astype(jnp.float32)),
            **attention_diagnostics,
        }
        return new_carry, logits, diagnostics

    def _run_unroll(
        self,
        tokens: Array,
        puzzle_identifiers: Array | None,
        *,
        train: bool,
        dropout_key: Array | None,
        compute_terminal_residual: bool = True,
        include_layer_diagnostics: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        condition_mask = self._condition_mask(tokens)
        input_embeddings = self._input_embeddings(
            tokens,
            puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
        )
        state0 = self._initial_state(tokens.shape[0])

        def scan_step(state: State, _step_index):
            z_h_prev, z_l_prev = state
            next_state, attention_diagnostics = self._act_step(state, input_embeddings)
            z_h_next, z_l_next = next_state
            z_h_data = z_h_next[:, self.prefix_len :]
            z_h_prev_data = z_h_prev[:, self.prefix_len :]
            h_delta = jnp.linalg.norm((z_h_data - z_h_prev_data).astype(jnp.float32), axis=-1, keepdims=True)
            hidden_delta = self._blank_mean(h_delta, condition_mask)
            step_logits = self._logits_from_state(z_h_data)
            q_halt_logits, q_continue_logits = self._quality_logits(z_h_next)
            carry_state = jax.tree.map(jax.lax.stop_gradient, next_state)
            if include_layer_diagnostics:
                z_l_data = z_l_next[:, self.prefix_len :]
                z_l_prev_data = z_l_prev[:, self.prefix_len :]
                l_delta = jnp.linalg.norm((z_l_data - z_l_prev_data).astype(jnp.float32), axis=-1, keepdims=True)
                low_hidden_delta = self._blank_mean(l_delta, condition_mask)
                low_step_logits = jax.lax.stop_gradient(self._logits_from_state(z_l_data))
            else:
                low_hidden_delta = jnp.zeros_like(hidden_delta)
                low_step_logits = jnp.zeros_like(step_logits)
            return carry_state, (
                step_logits,
                low_step_logits,
                hidden_delta,
                low_hidden_delta,
                q_halt_logits,
                q_continue_logits,
                jnp.stack(
                    [
                        attention_diagnostics["attention_gate_mean"],
                        attention_diagnostics["attention_gate_std"],
                        attention_diagnostics["attention_gate_low_saturation"],
                        attention_diagnostics["attention_gate_high_saturation"],
                        attention_diagnostics["attention_local_distance_scale"],
                    ]
                ),
            )

        final_state, (
            step_logits,
            low_step_logits,
            step_hidden_delta,
            low_step_hidden_delta,
            q_halt_logits,
            q_continue_logits,
            step_attention_diagnostics,
        ) = jax.lax.scan(
            scan_step,
            state0,
            jnp.arange(self.config.num_steps),
        )
        diagnostics = {
            "hidden_delta_mean": step_hidden_delta,
            "quality_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "quality_target_solved": jnp.asarray(1.0, dtype=jnp.float32),
            "unroll_steps": jnp.asarray(self.config.num_steps, dtype=jnp.float32),
            "h": final_state[0][:, self.prefix_len :],
            "l": final_state[1][:, self.prefix_len :],
            "attention_gate_mean": jnp.mean(step_attention_diagnostics[:, 0]),
            "attention_gate_std": jnp.mean(step_attention_diagnostics[:, 1]),
            "attention_gate_low_saturation": jnp.mean(step_attention_diagnostics[:, 2]),
            "attention_gate_high_saturation": jnp.mean(step_attention_diagnostics[:, 3]),
            "attention_local_distance_scale": jnp.mean(step_attention_diagnostics[:, 4]),
        }
        if include_layer_diagnostics:
            diagnostics["h_hidden_delta_mean"] = step_hidden_delta
            diagnostics["l_hidden_delta_mean"] = low_step_hidden_delta
            diagnostics["l_logits"] = low_step_logits
        if compute_terminal_residual:
            final_logits = step_logits[-1]
            next_state, _ = self._act_step(final_state, input_embeddings)
            next_logits = self._logits_from_state(next_state[0][:, self.prefix_len :])
            probs = jax.nn.softmax(final_logits, axis=-1)
            next_probs = jax.nn.softmax(next_logits, axis=-1)
            prob_delta = next_probs - probs
            prob_residual_mse = self._blank_mean(jnp.square(prob_delta), condition_mask)
            diagnostics["terminal_belief_mse"] = prob_residual_mse
            diagnostics["terminal_belief_delta"] = jnp.sqrt(jnp.maximum(prob_residual_mse, 0.0))
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        compute_terminal_residual: bool = True,
        include_layer_diagnostics: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        return self._run_unroll(
            tokens,
            puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
            compute_terminal_residual=compute_terminal_residual,
            include_layer_diagnostics=include_layer_diagnostics,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        compute_terminal_residual: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
            compute_terminal_residual=compute_terminal_residual,
        )

    def forward_all_steps(
        self,
        tokens: Array,
        *,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_all_steps_with_diagnostics(
            tokens,
            puzzle_identifiers=puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
        )
        return logits

    def forward_final(
        self,
        tokens: Array,
        *,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_final_with_diagnostics(
            tokens,
            puzzle_identifiers=puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
        )
        return logits

    def __call__(
        self,
        tokens: Array,
        *,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        return self.forward_final(
            tokens,
            puzzle_identifiers=puzzle_identifiers,
            train=train,
            dropout_key=dropout_key,
        )[-1]

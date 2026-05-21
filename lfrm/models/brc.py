from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, casted_linear_init, compute_dtype, maybe_cast, trunc_normal_init
from .recurrent.layers import FullAttention, apply_rope, dot_product_attention, rms_norm as _shared_rms_norm


def _rms_norm(x: Array, eps: float = 1e-5) -> Array:
    return _shared_rms_norm(x, eps)


class BRCSolverBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        mlp_hidden_dim = max(hidden_dim, mlp_ratio * hidden_dim)
        self.attention = FullAttention(
            hidden_dim,
            num_heads,
            dtype,
            name="BRC",
            rngs=rngs,
        )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cross_q = nnx.Linear(
            hidden_dim,
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.cross_kv = nnx.Linear(
            hidden_dim,
            2 * hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.cross_out = nnx.Linear(
            hidden_dim,
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.film = nnx.Linear(
            hidden_dim,
            2 * hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.msg_in = nnx.Linear(
            hidden_dim,
            mlp_hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.msg_out = nnx.Linear(
            mlp_hidden_dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
    def __call__(
        self,
        h: Array,
        condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        attn = self.attention(h, rope_cos=rope_cos, rope_sin=rope_sin)
        h = _rms_norm(h.astype(jnp.float32) + attn.astype(jnp.float32), self.config.brc_config.rms_norm_eps).astype(self.dtype)
        batch_size, seq_len, hidden_dim = h.shape
        q = self.cross_q(maybe_cast(h, self.dtype))
        k, v = jnp.split(self.cross_kv(maybe_cast(condition, self.dtype)), 2, axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)
        cross = dot_product_attention(q, k, v).reshape(batch_size, seq_len, hidden_dim)
        h = _rms_norm(h.astype(jnp.float32) + self.cross_out(cross).astype(jnp.float32), self.config.brc_config.rms_norm_eps).astype(self.dtype)
        pooled_condition = jnp.mean(condition.astype(jnp.float32), axis=1)
        scale, shift = jnp.split(self.film(maybe_cast(pooled_condition, self.dtype)).astype(jnp.float32), 2, axis=-1)
        h_norm = _rms_norm(h.astype(jnp.float32), self.config.brc_config.rms_norm_eps).astype(jnp.float32)
        h = _rms_norm(
            h.astype(jnp.float32)
            + h_norm * jnp.tanh(scale)[:, None, :]
            + shift[:, None, :],
            self.config.brc_config.rms_norm_eps,
        ).astype(self.dtype)
        msg = self.msg_out(jax.nn.silu(self.msg_in(maybe_cast(h, self.dtype)))).astype(jnp.float32)
        return _rms_norm(h.astype(jnp.float32) + msg, self.config.brc_config.rms_norm_eps).astype(self.dtype)


class BRCModel(nnx.Module):
    """Belief recurrent controller for fixed-size grid reasoning tasks."""

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.model_type != "brc":
            raise ValueError("BRCModel requires model_type='brc'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.task_type == "sudoku" and config.vocab_size < 11:
            raise ValueError("BRC Sudoku expects vocab_size >= 11")
        brc = config.brc_config
        if brc.belief_steps < 1:
            raise ValueError("BRC belief_steps must be at least 1")
        if min(brc.h_cycles, brc.l_cycles, brc.l_layers) < 1:
            raise ValueError("BRC h_cycles, l_cycles, and l_layers must be at least 1")
        hidden_dim = int(brc.hidden_state_dim) if brc.hidden_state_dim > 0 else config.d_model
        if hidden_dim < 1:
            raise ValueError("BRC hidden_state_dim must be positive or 0 for d_model")
        if brc.num_heads < 1:
            raise ValueError("BRC num_heads must be at least 1")
        if hidden_dim % brc.num_heads != 0:
            raise ValueError("BRC hidden state dimension must be divisible by num_heads")
        if brc.mlp_ratio < 1:
            raise ValueError("BRC mlp_ratio must be at least 1")
        if brc.position_encoding not in ("rope", "learned", "none"):
            raise ValueError("BRC position_encoding must be 'rope', 'learned', or 'none'")
        if brc.position_encoding == "rope" and (hidden_dim // brc.num_heads) % 2 != 0:
            raise ValueError("BRC RoPE head dimension must be even")
        if brc.step_loss_schedule not in ("uniform", "linear"):
            raise ValueError("BRC step_loss_schedule must be 'uniform' or 'linear'")
        if not 0.0 <= brc.denoise_initial_prob <= 1.0:
            raise ValueError("BRC denoise_initial_prob must be in [0, 1]")
        if not 0.0 <= brc.denoise_trajectory_prob <= 1.0:
            raise ValueError("BRC denoise_trajectory_prob must be in [0, 1]")
        if not 0.0 <= brc.denoise_teacher_reveal_prob <= 1.0:
            raise ValueError("BRC denoise_teacher_reveal_prob must be in [0, 1]")
        if len(brc.denoise_mode_weights) != 3:
            raise ValueError("BRC denoise_mode_weights must contain three weights")
        if any(weight < 0.0 for weight in brc.denoise_mode_weights):
            raise ValueError("BRC denoise_mode_weights must be non-negative")
        if sum(brc.denoise_mode_weights) <= 0.0:
            raise ValueError("BRC denoise_mode_weights must contain a positive weight")
        if brc.fixed_point_entropy_weight < 0.0:
            raise ValueError("BRC fixed_point_entropy_weight must be non-negative")
        if min(brc.fixed_point_loss_weight, brc.context_weight_reg_weight) < 0.0:
            raise ValueError("BRC fixed-point loss weights must be non-negative")

        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.belief_steps = int(brc.belief_steps)
        self.h_cycles = int(brc.h_cycles)
        self.l_cycles = int(brc.l_cycles)
        self.hidden_dim = hidden_dim
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.belief_vocab_size = config.vocab_size
        self.box_height, self.box_width = self._box_shape(config.grid_height, config.grid_width)

        rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
        cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
        boxes = (rows // self.box_height) * (config.grid_width // self.box_width) + (cols // self.box_width)
        box_indices = self._build_box_indices(config.grid_height, config.grid_width, self.box_height, self.box_width)
        self.row_ids = nnx.data(rows)
        self.col_ids = nnx.data(cols)
        self.box_ids = nnx.data(boxes)
        self.box_indices = nnx.data(box_indices)
        self.num_boxes = int(box_indices.shape[0])

        embed_init = trunc_normal_init(1.0 / self.embed_scale)
        self.puzzle_embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.context_embed = nnx.Embed(2, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.draft_embed = nnx.Embed(
            max(10, config.vocab_size),
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        if brc.position_encoding == "learned":
            self.row_embed = nnx.Embed(config.grid_height, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.col_embed = nnx.Embed(config.grid_width, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.box_embed = nnx.Embed(self.num_boxes, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.time_embed = nnx.Embed(self.belief_steps, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        if brc.position_encoding == "rope":
            head_dim = self.hidden_dim // brc.num_heads
            inv_freq = 1.0 / (brc.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
            positions = jnp.arange(config.seq_len, dtype=jnp.float32)
            freqs = positions[:, None] * inv_freq[None, :]
            rope = jnp.concatenate((freqs, freqs), axis=-1)
            self.rope_cos = nnx.data(jnp.cos(rope))
            self.rope_sin = nnx.data(jnp.sin(rope))
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.input_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.readout_condition = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.readout_gate = nnx.Linear(
            self.hidden_dim * 4,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            bias_init=nnx.initializers.constant(-1.0),
            rngs=rngs,
        )
        readout_mlp_dim = max(self.hidden_dim, brc.mlp_ratio * self.hidden_dim)
        self.readout_fuse_in = nnx.Linear(
            self.hidden_dim * 4,
            readout_mlp_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.readout_fuse_out = nnx.Linear(
            readout_mlp_dim,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.solver_blocks = nnx.List(
            [
                BRCSolverBlock(
                    config,
                    self.hidden_dim,
                    brc.num_heads,
                    brc.mlp_ratio,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(brc.l_layers)
            ]
        )
        self.lm_head = nnx.Linear(
            self.hidden_dim,
            config.vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.step_gate = nnx.Linear(
            self.hidden_dim,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.energy_context_gate = nnx.Linear(
            config.d_model,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    @staticmethod
    def _box_shape(grid_height: int, grid_width: int) -> tuple[int, int]:
        box_height = int(math.sqrt(grid_height))
        while box_height > 1 and grid_height % box_height != 0:
            box_height -= 1
        box_width = int(math.sqrt(grid_width))
        while box_width > 1 and grid_width % box_width != 0:
            box_width -= 1
        return max(box_height, 1), max(box_width, 1)

    @staticmethod
    def _build_box_indices(grid_height: int, grid_width: int, box_height: int, box_width: int) -> Array:
        indices: list[list[int]] = []
        for box_row in range(grid_height // box_height):
            for box_col in range(grid_width // box_width):
                cells = []
                for row in range(box_row * box_height, (box_row + 1) * box_height):
                    for col in range(box_col * box_width, (box_col + 1) * box_width):
                        cells.append(row * grid_width + col)
                indices.append(cells)
        return jnp.asarray(indices, dtype=jnp.int32)

    def context_mask(self, tokens: Array) -> Array:
        return tokens != 0

    def _normalize_belief_logits(self, belief_logits: Array, tokens: Array) -> Array:
        del tokens
        return belief_logits.astype(jnp.float32)

    def initial_belief_logits(self, tokens: Array) -> Array:
        zeros = jnp.zeros((*tokens.shape, self.belief_vocab_size), dtype=jnp.float32)
        return self._normalize_belief_logits(zeros, tokens)

    def belief_logits_from_tokens(self, puzzle: Array, candidate: Array, mask: Array | None = None) -> Array:
        class_ids = jnp.clip(candidate, 0, self.config.vocab_size - 1)
        hard_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(class_ids.astype(jnp.int32), self.belief_vocab_size)
        if mask is not None:
            hard_logits = jnp.where(mask[..., None], hard_logits, 0.0)
        return self._normalize_belief_logits(hard_logits, puzzle)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _belief_embedding(self, tokens: Array, belief_logits: Array) -> Array:
        belief_probs = jax.nn.softmax(self._normalize_belief_logits(belief_logits, tokens), axis=-1)
        embedding_table = maybe_cast(self.draft_embed.embedding[: self.config.vocab_size], self.dtype)
        return jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(belief_probs, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )

    def _cell_embeddings(
        self,
        tokens: Array,
        belief_logits: Array,
        base_embeddings: Array,
        time_embedding: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        belief_embedding = self._belief_embedding(tokens, belief_logits)
        x = (
            base_embeddings
            + belief_embedding
            + time_embedding[None, None, :]
        ) * math.sqrt(1.0 / 5.0)
        return self.dropout(x, deterministic=not train, rngs=dropout_key).astype(self.dtype)

    def _belief_to_token_logits(self, belief_logits: Array, tokens: Array, step_index: Array) -> Array:
        total_steps = jnp.maximum(jnp.asarray(self.belief_steps - 1, dtype=jnp.float32), 1.0)
        progress = step_index.astype(jnp.float32) / total_steps
        sharpen = jnp.where(step_index >= jnp.maximum(self.belief_steps - 4, 0), 1.0 + 2.0 * progress, 1.0)
        belief_logits = self._normalize_belief_logits(belief_logits, tokens) * sharpen
        return belief_logits

    def _belief_update(self, tokens: Array, belief_logits: Array, delta_logits: Array, alpha: Array, step_index: Array) -> Array:
        del step_index
        delta = delta_logits.astype(jnp.float32)
        next_belief = belief_logits.astype(jnp.float32) + alpha.astype(jnp.float32) * delta
        return self._normalize_belief_logits(next_belief, tokens)

    def _fixed_point_stats(
        self,
        belief_logits: Array,
        next_belief: Array,
        context_features: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        prev_log_probs = jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1)
        next_log_probs = jax.nn.log_softmax(next_belief.astype(jnp.float32), axis=-1)
        prev_probs = jax.lax.stop_gradient(jnp.exp(prev_log_probs))
        next_probs = jnp.exp(next_log_probs)
        cell_kl = jnp.sum(prev_probs * (prev_log_probs - next_log_probs), axis=-1)
        cell_entropy = -jnp.sum(next_probs * next_log_probs, axis=-1)
        kl_delta = jnp.mean(cell_kl)
        entropy = jnp.mean(cell_entropy)
        confidence = jnp.mean(jnp.max(next_probs, axis=-1))
        cell_update_norm = jnp.linalg.norm((next_belief - belief_logits).astype(jnp.float32), axis=-1)
        update_norm = jnp.mean(cell_update_norm)
        context_weight = 2.0 * jax.nn.sigmoid(
            self.energy_context_gate(maybe_cast(context_features, self.dtype)).astype(jnp.float32)
        ).squeeze(-1)
        cell_energy = cell_update_norm + self.brc.fixed_point_entropy_weight * cell_entropy
        energy = jnp.sum(context_weight * cell_energy) / jnp.maximum(jnp.sum(context_weight), 1.0)
        context_weight_reg = jnp.mean(jnp.square(context_weight - 1.0))
        context_weight_mean = jnp.mean(context_weight)
        return energy, kl_delta, entropy, confidence, update_norm, context_weight_reg, context_weight_mean

    def _hidden_l_cycle(self, hidden_state: Array, cell_input: Array) -> Array:
        hidden_input = self.input_to_hidden(maybe_cast(cell_input, self.dtype)).astype(self.dtype)
        hidden = hidden_state.astype(self.dtype)
        for _ in range(self.l_cycles):
            hidden = _rms_norm(
                hidden.astype(jnp.float32) + hidden_input.astype(jnp.float32),
                self.brc.rms_norm_eps,
            ).astype(self.dtype)
            for block in self.solver_blocks:
                hidden = block(hidden, hidden_input, self.rope_cos, self.rope_sin)
        return hidden

    def _readout_fuse(self, hidden_state: Array, cell_input: Array) -> Array:
        hidden = _rms_norm(hidden_state.astype(jnp.float32), self.brc.rms_norm_eps).astype(self.dtype)
        condition = self.readout_condition(maybe_cast(cell_input, self.dtype)).astype(self.dtype)
        condition = _rms_norm(condition.astype(jnp.float32), self.brc.rms_norm_eps).astype(self.dtype)
        gate_input = jnp.concatenate(
            (
                hidden,
                condition,
                (hidden.astype(jnp.float32) * condition.astype(jnp.float32)).astype(self.dtype),
                (hidden.astype(jnp.float32) - condition.astype(jnp.float32)).astype(self.dtype),
            ),
            axis=-1,
        )
        gate = jax.nn.sigmoid(self.readout_gate(maybe_cast(gate_input, self.dtype)).astype(jnp.float32))
        update = self.readout_fuse_out(
            jax.nn.silu(self.readout_fuse_in(maybe_cast(gate_input, self.dtype)))
        ).astype(jnp.float32)
        fused = hidden.astype(jnp.float32) + gate * (condition.astype(jnp.float32) + update)
        return _rms_norm(fused, self.brc.rms_norm_eps).astype(self.dtype)

    def _h_cycle_step(
        self,
        tokens: Array,
        belief_logits: Array,
        hidden_state: Array,
        base_embeddings: Array,
        time_embedding: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array, Array, dict[str, Array]]:
        cell_input = self._cell_embeddings(
            tokens,
            belief_logits,
            base_embeddings,
            time_embedding,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = self._hidden_l_cycle(hidden_state, cell_input)
        read_state = self._readout_fuse(hidden, cell_input)
        alpha = jax.nn.sigmoid(self.step_gate(maybe_cast(read_state, self.dtype)).astype(jnp.float32))
        delta_logits = self.lm_head(maybe_cast(read_state, self.dtype))
        next_belief = self._belief_update(tokens, belief_logits, delta_logits, alpha, step_index)
        return next_belief, hidden, alpha, {
            "step_gate_mean": jnp.mean(alpha),
            "step_gate_std": jnp.std(alpha),
        }

    def _belief_step(
        self,
        tokens: Array,
        belief_logits: Array,
        hidden_state: Array,
        base_embeddings: Array,
        time_embedding: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array, Array, dict[str, Array]]:
        alpha = jnp.zeros((*tokens.shape, 1), dtype=jnp.float32)
        diagnostics = {
            "step_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "step_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
        }
        if dropout_key is None:
            h_dropout_keys = jax.random.split(jax.random.key(0), self.h_cycles)
        else:
            h_dropout_keys = jax.random.split(dropout_key, self.h_cycles)
        for h_index in range(self.h_cycles - 1):
            belief_logits, hidden_state, alpha, diagnostics = self._h_cycle_step(
                tokens,
                belief_logits,
                hidden_state,
                base_embeddings,
                time_embedding,
                step_index,
                train=train,
                dropout_key=h_dropout_keys[h_index],
            )
            belief_logits = jax.lax.stop_gradient(belief_logits)
            hidden_state = jax.lax.stop_gradient(hidden_state)
        belief_logits, hidden_state, alpha, diagnostics = self._h_cycle_step(
            tokens,
            belief_logits,
            hidden_state,
            base_embeddings,
            time_embedding,
            step_index,
            train=train,
            dropout_key=h_dropout_keys[self.h_cycles - 1],
        )
        return belief_logits, jax.lax.stop_gradient(hidden_state), alpha, diagnostics

    def initial_hidden_state(
        self,
        tokens: Array,
        belief_logits: Array,
        base_embeddings: Array,
        time_embedding: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        cell_input = self._cell_embeddings(
            tokens,
            belief_logits,
            base_embeddings,
            time_embedding,
            train=train,
            dropout_key=dropout_key,
        )
        return self.input_to_hidden(maybe_cast(cell_input, self.dtype)).astype(self.dtype)

    def context_memory(
        self,
        tokens: Array,
    ) -> tuple[Array, Array]:
        position_embeddings = self._position_embeddings()
        context = self.context_mask(tokens)
        base_embeddings = (
            self.puzzle_embed(tokens.astype(jnp.int32))
            + self.context_embed(context.astype(jnp.int32))
            + position_embeddings[None, :, :]
        )
        return base_embeddings, context

    def run_diffusion(
        self,
        tokens: Array,
        *,
        initial_belief: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_belief is None:
            initial_belief = self.initial_belief_logits(tokens)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key, time_embedding = scan_inputs
            belief_logits, hidden_state = carry
            next_belief, next_hidden, _alpha, block_diagnostics = self._belief_step(
                tokens,
                belief_logits,
                hidden_state,
                base_embeddings,
                time_embedding,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
            )
            next_carry = (jax.lax.stop_gradient(next_belief), next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._belief_to_token_logits(next_belief, tokens, step_index)
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
            ) = self._fixed_point_stats(
                belief_logits,
                next_belief,
                base_embeddings,
            )
            return next_carry, (
                logits,
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

        step_indices = jnp.arange(self.belief_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.belief_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.belief_steps)
        time_embeddings = self.time_embed(step_indices)
        initial_hidden = self.initial_hidden_state(
            tokens,
            initial_belief,
            base_embeddings,
            time_embeddings[0],
            train=train,
            dropout_key=step_dropout_keys[0],
        )
        initial_carry = (initial_belief.astype(jnp.float32), initial_hidden)
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys, time_embeddings),
        )
        belief_final, _hidden_final = final_carry
        (
            final_energy,
            final_kl_delta,
            final_entropy,
            final_confidence,
            final_update_norm,
            final_context_weight_reg,
            final_context_weight_mean,
        ) = self._fixed_point_stats(
            initial_belief.astype(jnp.float32),
            belief_final,
            base_embeddings,
        )
        if return_final_only:
            final_step = jnp.asarray(self.belief_steps - 1, dtype=jnp.int32)
            logits = self._belief_to_token_logits(belief_final, tokens, final_step)
            diagnostics = {
                "diffusion_filled_ratio": jnp.zeros((self.belief_steps,), dtype=jnp.float32),
                "step_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "step_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
                "denoise_energy": final_energy,
                "belief_kl_delta": final_kl_delta,
                "belief_entropy": final_entropy,
                "belief_confidence": final_confidence,
                "belief_update_norm": final_update_norm,
                "context_weight_reg": final_context_weight_reg,
                "context_weight_mean": final_context_weight_mean,
                "unroll_steps": jnp.asarray(self.belief_steps, dtype=jnp.float32),
                "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
                "belief_logits": belief_final,
            }
            return logits, diagnostics
        (
            step_logits,
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
        diagnostics = {
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
            "per_step_denoise_energy": energy,
            "per_step_belief_entropy": entropy,
            "unroll_steps": jnp.asarray(self.belief_steps, dtype=jnp.float32),
            "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
            "belief_logits": belief_final,
        }
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_belief: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.run_diffusion(
            tokens,
            initial_belief=initial_belief,
            train=train,
            dropout_key=dropout_key,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_belief: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_belief=initial_belief,
        )

    def forward_all_steps(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_all_steps_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits

    def forward_final(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_final_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits[-1]

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        return self.forward_final(tokens, train=train, dropout_key=dropout_key)

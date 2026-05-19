from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, casted_linear_init, compute_dtype, maybe_cast, trunc_normal, trunc_normal_init
from .recurrent.layers import rms_norm as _shared_rms_norm


def _rms_norm(x: Array, eps: float = 1e-5) -> Array:
    return _shared_rms_norm(x, eps)


class RelationTypedAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_relations: int,
        grid_height: int,
        grid_width: int,
        position_encoding: str,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError("BRC-Sudoku d_model must be divisible by num_heads")
        if position_encoding not in ("learned", "rel2d", "none"):
            raise ValueError("BRC-Sudoku position_encoding must be 'learned', 'rel2d', or 'none'")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = d_model // num_heads
        self.position_encoding = position_encoding
        self.dtype = dtype
        if position_encoding == "rel2d":
            seq_len = grid_height * grid_width
            rows = jnp.arange(seq_len, dtype=jnp.int32) // grid_width
            cols = jnp.arange(seq_len, dtype=jnp.int32) % grid_width
            rel_row = rows[None, :] - rows[:, None] + (grid_height - 1)
            rel_col = cols[None, :] - cols[:, None] + (grid_width - 1)
            rel2d_indices = rel_row * (2 * grid_width - 1) + rel_col
            self.rel2d_indices = nnx.data(rel2d_indices.astype(jnp.int32))
            self.rel2d_bias = nnx.Embed(
                (2 * grid_height - 1) * (2 * grid_width - 1),
                num_heads,
                dtype=dtype,
                param_dtype=jnp.float32,
                embedding_init=nnx.initializers.zeros,
                rngs=rngs,
            )
        self.qkv = nnx.Linear(
            d_model,
            3 * d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.relation_embed = nnx.Embed(
            num_relations,
            d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            embedding_init=trunc_normal_init(1.0 / math.sqrt(d_model)),
            rngs=rngs,
        )
        self.relation_bias = nnx.Param(jnp.zeros((num_relations, num_heads), dtype=jnp.float32))
        self.combine = nnx.Linear(
            num_relations * d_model,
            d_model,
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

    def __call__(self, h: Array, relation_masks: Array) -> Array:
        batch_size, seq_len, d_model = h.shape
        qkv = self.qkv(maybe_cast(h, self.dtype))
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        scores = jnp.einsum("bhnd,bhmd->bhnm", q, k, preferred_element_type=jnp.float32)
        scores = scores / math.sqrt(self.head_dim)
        if self.position_encoding == "rel2d":
            rel2d_bias = self.rel2d_bias(self.rel2d_indices).astype(jnp.float32)
            scores = scores + jnp.moveaxis(rel2d_bias, -1, 0)[None, :, :, :]
        relation_scores = scores[:, None, :, :, :] + self.relation_bias[...][None, :, :, None, None]
        relation_scores = jnp.where(relation_masks[None, :, None, :, :], relation_scores, -1.0e9)
        weights = jax.nn.softmax(relation_scores.astype(jnp.float32), axis=-1).astype(h.dtype)
        attended = jnp.einsum("brhnm,bhmd->brhnd", weights, v, preferred_element_type=jnp.float32)
        attended = jnp.swapaxes(attended, 2, 3).reshape(batch_size, self.num_relations, seq_len, d_model)
        relation_ids = jnp.arange(self.num_relations, dtype=jnp.int32)
        attended = attended + self.relation_embed(relation_ids)[None, :, None, :]
        attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, self.num_relations * d_model)
        return self.out(self.combine(attended))


class RelationTypedSolverBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_heads: int,
        mlp_ratio: int,
        latent_dim: int,
        num_relations: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        d_model = config.d_model
        hidden_dim = max(d_model, mlp_ratio * d_model)
        self.relation_attention = RelationTypedAttention(
            d_model,
            num_heads,
            num_relations,
            config.grid_height,
            config.grid_width,
            config.brc_config.position_encoding,
            dtype,
            rngs=rngs,
        )
        self.msg_in = nnx.Linear(
            2 * d_model,
            hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.msg_out = nnx.Linear(
            hidden_dim,
            d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.film = nnx.Linear(
            latent_dim,
            3 * d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, h: Array, cell_input: Array, z: Array, relation_masks: Array) -> tuple[Array, dict[str, Array]]:
        h = h + cell_input
        relation_msg = self.relation_attention(h, relation_masks)
        msg = jnp.concatenate([h, relation_msg], axis=-1)
        msg = self.msg_out(jax.nn.silu(self.msg_in(maybe_cast(msg, self.dtype)))).astype(jnp.float32)
        scale, shift, gate = jnp.split(self.film(maybe_cast(z, self.dtype)).astype(jnp.float32), 3, axis=-1)
        scale = 1.0 + 0.1 * jnp.tanh(scale)[:, None, :]
        shift = shift[:, None, :]
        gate = jax.nn.sigmoid(gate)[:, None, :]
        msg = scale * msg + shift
        h_next = _rms_norm(h.astype(jnp.float32) + gate * msg).astype(self.dtype)
        return h_next, {
            "brc_gate_mean": jnp.mean(gate),
            "brc_gate_std": jnp.std(gate),
        }


class RelationTypedVerifierBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_heads: int,
        mlp_ratio: int,
        num_relations: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        d_model = config.d_model
        hidden_dim = max(d_model, mlp_ratio * d_model)
        self.relation_attention = RelationTypedAttention(
            d_model,
            num_heads,
            num_relations,
            config.grid_height,
            config.grid_width,
            config.brc_config.position_encoding,
            dtype,
            rngs=rngs,
        )
        self.msg_in = nnx.Linear(
            2 * d_model,
            hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.msg_out = nnx.Linear(
            hidden_dim,
            d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, h: Array, relation_masks: Array) -> Array:
        relation_msg = self.relation_attention(h, relation_masks)
        msg = jnp.concatenate([h, relation_msg], axis=-1)
        msg = self.msg_out(jax.nn.silu(self.msg_in(maybe_cast(msg, self.dtype)))).astype(jnp.float32)
        return _rms_norm(h.astype(jnp.float32) + msg).astype(self.dtype)


class BRCSudokuModel(nnx.Module):
    """Sudoku-only BRC MVP.

    The model keeps Sudoku working memory in a soft belief state plus a recurrent
    spatial field. The small latent controller only FiLM/gates the update
    dynamics; it is intentionally not a slot cache or direct answer decoder.
    """

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.model_type != "brc_sudoku":
            raise ValueError("BRCSudokuModel requires model_type='brc_sudoku'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.vocab_size < 11:
            raise ValueError("BRC Sudoku expects vocab_size >= 11")
        brc = config.brc_config
        if brc.recurrent_steps < 1:
            raise ValueError("BRC recurrent_steps must be at least 1")
        if brc.block_layers < 1:
            raise ValueError("BRC block_layers must be at least 1")
        if brc.latent_dim < 1:
            raise ValueError("BRC latent_dim must be at least 1")
        if brc.num_heads < 1:
            raise ValueError("BRC num_heads must be at least 1")
        if config.d_model % brc.num_heads != 0:
            raise ValueError("BRC d_model must be divisible by num_heads")
        if brc.mlp_ratio < 1:
            raise ValueError("BRC mlp_ratio must be at least 1")
        if brc.position_encoding not in ("learned", "rel2d", "none"):
            raise ValueError("BRC position_encoding must be 'learned', 'rel2d', or 'none'")
        if brc.step_loss_weights is not None:
            if len(brc.step_loss_weights) != brc.recurrent_steps:
                raise ValueError("BRC step_loss_weights length must equal recurrent_steps")
            if any(weight < 0.0 for weight in brc.step_loss_weights):
                raise ValueError("BRC step_loss_weights must be non-negative")
            if sum(brc.step_loss_weights) <= 0.0:
                raise ValueError("BRC step_loss_weights must contain a positive weight")
        if brc.latent_fit_steps < 0:
            raise ValueError("BRC latent_fit_steps must be non-negative")
        if not 0.0 <= brc.denoise_initial_prob <= 1.0:
            raise ValueError("BRC denoise_initial_prob must be in [0, 1]")
        if not 0.0 <= brc.denoise_teacher_reveal_prob <= 1.0:
            raise ValueError("BRC denoise_teacher_reveal_prob must be in [0, 1]")
        if len(brc.denoise_mode_weights) != 4:
            raise ValueError("BRC denoise_mode_weights must contain four weights")
        if any(weight < 0.0 for weight in brc.denoise_mode_weights):
            raise ValueError("BRC denoise_mode_weights must be non-negative")
        if sum(brc.denoise_mode_weights) <= 0.0:
            raise ValueError("BRC denoise_mode_weights must contain a positive weight")
        if brc.verifier_layers < 1:
            raise ValueError("BRC verifier_layers must be at least 1")

        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.recurrent_steps = int(brc.recurrent_steps)
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.box_height, self.box_width = self._box_shape(config.grid_height, config.grid_width)

        rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
        cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
        boxes = (rows // self.box_height) * (config.grid_width // self.box_width) + (cols // self.box_width)
        box_indices = self._build_box_indices(config.grid_height, config.grid_width, self.box_height, self.box_width)
        self.row_ids = nnx.data(rows)
        self.col_ids = nnx.data(cols)
        self.box_ids = nnx.data(boxes)
        self.box_indices = nnx.data(box_indices)
        relation_masks = self._build_sudoku_relation_masks(rows, cols, boxes)
        self.relation_masks = nnx.data(relation_masks)
        self.num_relations = int(relation_masks.shape[0])
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
        self.given_embed = nnx.Embed(2, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.draft_embed = nnx.Embed(10, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        if brc.position_encoding == "learned":
            self.row_embed = nnx.Embed(config.grid_height, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.col_embed = nnx.Embed(config.grid_width, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.box_embed = nnx.Embed(self.num_boxes, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.time_embed = nnx.Embed(self.recurrent_steps, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

        self.latent_pool = nnx.Linear(
            config.d_model,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.latent_out = nnx.Linear(
            config.d_model,
            brc.latent_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.z_global = nnx.Param(trunc_normal(rngs.params(), (brc.latent_dim,), 1.0 / math.sqrt(brc.latent_dim)))
        self.latent_to_hidden = nnx.Linear(
            brc.latent_dim,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.state_init_to_hidden = nnx.Linear(
            config.d_model,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.h0 = nnx.Param(trunc_normal(rngs.params(), (config.d_model,), 1.0 / math.sqrt(config.d_model)))
        self.solver_blocks = nnx.List(
            [
                RelationTypedSolverBlock(
                    config,
                    brc.num_heads,
                    brc.mlp_ratio,
                    brc.latent_dim,
                    self.num_relations,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(brc.block_layers)
            ]
        )
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

        self.verifier_puzzle_embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.verifier_candidate_embed = nnx.Embed(
            config.vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.verifier_given_embed = nnx.Embed(2, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.verifier_blocks = nnx.List(
            [
                RelationTypedVerifierBlock(
                    config,
                    brc.num_heads,
                    brc.mlp_ratio,
                    self.num_relations,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(brc.verifier_layers)
            ]
        )
        self.verifier_hidden = nnx.Linear(
            config.d_model,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.verifier_head = nnx.Linear(
            config.d_model,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
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

    @staticmethod
    def _build_sudoku_relation_masks(rows: Array, cols: Array, boxes: Array) -> Array:
        eye = jnp.eye(rows.shape[0], dtype=bool)
        same_row = (rows[:, None] == rows[None, :]) & ~eye
        same_col = (cols[:, None] == cols[None, :]) & ~eye
        same_box = (boxes[:, None] == boxes[None, :]) & ~eye
        return jnp.stack([eye, same_row, same_col, same_box], axis=0)

    def condition_mask(self, tokens: Array) -> Array:
        return tokens != 1

    def draft_from_tokens(self, tokens: Array) -> Array:
        return jnp.where(tokens >= 2, tokens - 1, 0).astype(jnp.int32)

    def initial_draft(self, tokens: Array) -> Array:
        return jnp.where(self.condition_mask(tokens), self.draft_from_tokens(tokens), 0).astype(jnp.int32)

    def _clamp_belief_logits(self, belief_logits: Array, tokens: Array) -> Array:
        given = self.condition_mask(tokens)
        digit_ids = jnp.clip(tokens - 2, 0, 8)
        given_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(digit_ids.astype(jnp.int32), 9)
        return jnp.where(given[..., None], given_logits, belief_logits.astype(jnp.float32))

    def initial_belief_logits(self, tokens: Array) -> Array:
        zeros = jnp.zeros((*tokens.shape, 9), dtype=jnp.float32)
        return self._clamp_belief_logits(zeros, tokens)

    def belief_logits_from_tokens(self, puzzle: Array, candidate: Array, mask: Array | None = None) -> Array:
        digit_ids = jnp.clip(candidate - 2, 0, 8)
        hard_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(digit_ids.astype(jnp.int32), 9)
        if mask is not None:
            hard_logits = jnp.where(mask[..., None], hard_logits, 0.0)
        return self._clamp_belief_logits(hard_logits, puzzle)

    def belief_logits_from_draft(self, tokens: Array, draft: Array) -> Array:
        digit_ids = jnp.clip(draft - 1, 0, 8)
        hard_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(digit_ids.astype(jnp.int32), 9)
        hard_logits = jnp.where(draft[..., None] > 0, hard_logits, 0.0)
        return self._clamp_belief_logits(hard_logits, tokens)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _belief_embedding(self, tokens: Array, belief_logits: Array) -> Array:
        belief_probs = jax.nn.softmax(self._clamp_belief_logits(belief_logits, tokens), axis=-1)
        digit_embedding_table = maybe_cast(self.draft_embed.embedding[1:10], self.dtype)
        return jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(belief_probs, self.dtype),
            digit_embedding_table,
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

    def infer_latent(self, tokens: Array) -> Array:
        given = self.condition_mask(tokens).astype(jnp.int32)
        position_embeddings = self._position_embeddings()
        x = (
            self.puzzle_embed(tokens.astype(jnp.int32))
            + self.given_embed(given)
            + position_embeddings[None, :, :]
        ) * math.sqrt(1.0 / 3.0)
        pooled = jnp.mean(x.astype(jnp.float32), axis=1)
        latent = self.latent_out(jax.nn.silu(self.latent_pool(maybe_cast(pooled, self.dtype)))).astype(jnp.float32)
        return latent + self.z_global[...][None, :]

    def _clamp_logits(self, logits: Array, tokens: Array) -> Array:
        logits = logits.astype(jnp.float32)
        logits = logits.at[..., :2].set(-1.0e4)
        given = self.condition_mask(tokens)
        given_logits = jnp.full_like(logits, -1.0e4)
        given_logits = given_logits + 2.0e4 * jax.nn.one_hot(tokens.astype(jnp.int32), self.config.vocab_size)
        return jnp.where(given[..., None], given_logits, logits)

    def _belief_to_token_logits(self, belief_logits: Array, tokens: Array, step_index: Array) -> Array:
        total_steps = jnp.maximum(jnp.asarray(self.recurrent_steps - 1, dtype=jnp.float32), 1.0)
        progress = step_index.astype(jnp.float32) / total_steps
        sharpen = jnp.where(step_index >= jnp.maximum(self.recurrent_steps - 4, 0), 1.0 + 2.0 * progress, 1.0)
        digit_logits = self._clamp_belief_logits(belief_logits, tokens) * sharpen
        logits = jnp.full((*tokens.shape, self.config.vocab_size), -1.0e4, dtype=jnp.float32)
        return logits.at[..., 2:11].set(digit_logits)

    def _belief_update(self, tokens: Array, belief_logits: Array, raw_logits: Array, step_index: Array) -> Array:
        del step_index
        delta = raw_logits[..., 2:11].astype(jnp.float32)
        next_belief = 0.5 * belief_logits.astype(jnp.float32) + delta
        return self._clamp_belief_logits(next_belief, tokens)

    def _solver_update(self, h: Array, cell_input: Array, z: Array) -> tuple[Array, dict[str, Array]]:
        gate_mean = jnp.asarray(0.0, dtype=jnp.float32)
        gate_std = jnp.asarray(0.0, dtype=jnp.float32)
        for block in self.solver_blocks:
            h, block_diagnostics = block(h, cell_input, z, self.relation_masks)
            gate_mean = gate_mean + block_diagnostics["brc_gate_mean"]
            gate_std = gate_std + block_diagnostics["brc_gate_std"]
        normalizer = jnp.asarray(len(self.solver_blocks), dtype=jnp.float32)
        return h, {
            "brc_gate_mean": gate_mean / normalizer,
            "brc_gate_std": gate_std / normalizer,
        }

    def initial_recurrent_state(
        self,
        tokens: Array,
        z: Array,
        initial_belief: Array,
    ) -> tuple[Array, Array, Array]:
        position_embeddings = self._position_embeddings()
        given = self.condition_mask(tokens)
        base_embeddings = (
            self.puzzle_embed(tokens.astype(jnp.int32))
            + self.given_embed(given.astype(jnp.int32))
            + position_embeddings[None, :, :]
        )
        init_belief_embedding = self._belief_embedding(tokens, initial_belief)
        state_init_input = (base_embeddings + init_belief_embedding) * math.sqrt(1.0 / 4.0)
        latent_hidden = self.latent_to_hidden(maybe_cast(z, self.dtype)).astype(jnp.float32)
        h_bias = self.h0[...][None, None, :].astype(jnp.float32)
        h = self.state_init_to_hidden(maybe_cast(state_init_input, self.dtype)).astype(jnp.float32)
        h = _rms_norm(h + latent_hidden[:, None, :] + h_bias).astype(self.dtype)
        return h, base_embeddings, given

    def run_diffusion(
        self,
        tokens: Array,
        *,
        z: Array | None = None,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_raw_final_logits: bool = False,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if z is None:
            z = self.infer_latent(tokens)
        if initial_belief is None:
            if initial_draft is None:
                initial_belief = self.initial_belief_logits(tokens)
            else:
                initial_belief = self.belief_logits_from_draft(tokens, initial_draft)
        h, base_embeddings, given = self.initial_recurrent_state(tokens, z, initial_belief)
        unknown = (~given).astype(jnp.float32)
        unknown_normalizer = jnp.maximum(jnp.sum(unknown), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key, time_embedding = scan_inputs
            if return_raw_final_logits:
                h_prev, belief_logits, _raw_final_logits = carry
            else:
                h_prev, belief_logits = carry
            cell_input = self._cell_embeddings(
                tokens,
                belief_logits,
                base_embeddings,
                time_embedding,
                train=train,
                dropout_key=step_dropout_key,
            )
            h_next, block_diagnostics = self._solver_update(h_prev, cell_input, z)
            raw_logits = self.lm_head(maybe_cast(h_next, self.dtype))
            next_belief = self._belief_update(tokens, belief_logits, raw_logits, step_index)
            next_carry = (h_next, next_belief)
            if return_raw_final_logits:
                next_carry = (h_next, next_belief, raw_logits.astype(jnp.float32))
            if return_final_only:
                return next_carry, None
            logits = self._belief_to_token_logits(next_belief, tokens, step_index)
            hidden_delta = jnp.linalg.norm((h_next - h_prev).astype(jnp.float32), axis=-1)
            hidden_delta = jnp.sum(hidden_delta * unknown) / unknown_normalizer
            belief_probs = jax.nn.softmax(next_belief, axis=-1)
            confidence = jnp.max(belief_probs, axis=-1)
            filled_ratio = jnp.sum(confidence * unknown) / unknown_normalizer
            return next_carry, (
                logits,
                hidden_delta,
                filled_ratio,
                block_diagnostics["brc_gate_mean"],
                block_diagnostics["brc_gate_std"],
            )

        step_indices = jnp.arange(self.recurrent_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.recurrent_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.recurrent_steps)
        time_embeddings = self.time_embed(step_indices)
        initial_carry = (h, initial_belief.astype(jnp.float32))
        if return_raw_final_logits:
            raw_final0 = jnp.zeros((*tokens.shape, self.config.vocab_size), dtype=jnp.float32)
            initial_carry = (h, initial_belief.astype(jnp.float32), raw_final0)
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys, time_embeddings),
        )
        if return_raw_final_logits:
            h_final, belief_final, raw_final_logits = final_carry
        else:
            h_final, belief_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.recurrent_steps - 1, dtype=jnp.int32)
            logits = self._belief_to_token_logits(belief_final, tokens, final_step)
            diagnostics = {
                "hidden_delta_mean": jnp.zeros((self.recurrent_steps,), dtype=jnp.float32),
                "diffusion_filled_ratio": jnp.zeros((self.recurrent_steps,), dtype=jnp.float32),
                "brc_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "brc_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.recurrent_steps, dtype=jnp.float32),
                "z": z,
                "h": h_final,
                "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
                "belief_logits": belief_final,
            }
            if return_raw_final_logits:
                diagnostics["raw_final_logits"] = raw_final_logits
            return logits, diagnostics
        step_logits, hidden_delta, filled_ratio, gate_mean, gate_std = scan_outputs[:5]
        diagnostics = {
            "hidden_delta_mean": hidden_delta,
            "diffusion_filled_ratio": filled_ratio,
            "brc_gate_mean": jnp.mean(gate_mean),
            "brc_gate_std": jnp.mean(gate_std),
            "unroll_steps": jnp.asarray(self.recurrent_steps, dtype=jnp.float32),
            "z": z,
            "h": h_final,
            "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32) + 1,
            "belief_logits": belief_final,
        }
        if return_raw_final_logits:
            diagnostics["raw_final_logits"] = raw_final_logits
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        z_override: Array | None = None,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        compute_terminal_residual: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        del compute_terminal_residual
        return self.run_diffusion(
            tokens,
            z=z_override,
            initial_draft=initial_draft,
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
        z_override: Array | None = None,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        compute_terminal_residual: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            z_override=z_override,
            initial_draft=initial_draft,
            initial_belief=initial_belief,
            compute_terminal_residual=compute_terminal_residual,
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

    def _verifier_energy_from_candidate_embedding(self, puzzle: Array, candidate_embedding: Array) -> Array:
        given = self.condition_mask(puzzle).astype(jnp.int32)
        h = (
            self.verifier_puzzle_embed(puzzle.astype(jnp.int32))
            + candidate_embedding
            + self.verifier_given_embed(given)
            + self._position_embeddings()[None, :, :]
        ) * 0.5
        for block in self.verifier_blocks:
            h = block(h, self.relation_masks)
        pooled = jnp.mean(h.astype(jnp.float32), axis=1)
        hidden = jax.nn.silu(self.verifier_hidden(maybe_cast(pooled, self.dtype)))
        return self.verifier_head(hidden).astype(jnp.float32).squeeze(-1)

    def verifier_energy(self, puzzle: Array, candidate: Array) -> Array:
        candidate_embedding = self.verifier_candidate_embed(candidate.astype(jnp.int32))
        return self._verifier_energy_from_candidate_embedding(puzzle, candidate_embedding)

    def verifier_energy_from_probs(self, puzzle: Array, candidate_probs: Array) -> Array:
        embedding_table = maybe_cast(self.verifier_candidate_embed.embedding[...], self.dtype)
        candidate_embedding = jnp.einsum(
            "bnv,vd->bnd",
            maybe_cast(candidate_probs, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        return self._verifier_energy_from_candidate_embedding(puzzle, candidate_embedding)

    def refine_belief_with_verifier(
        self,
        puzzle: Array,
        belief_logits: Array,
        *,
        steps: int = 4,
        step_size: float = 0.05,
        prior_weight: float = 0.05,
        given_weight: float = 0.2,
        grad_clip_norm: float = 1.0,
    ) -> tuple[Array, dict[str, Array]]:
        belief0 = self._clamp_belief_logits(belief_logits, puzzle)
        unknown = (~self.condition_mask(puzzle)).astype(jnp.float32)[..., None]

        def candidate_probs_from_belief(belief: Array) -> Array:
            digit_probs = jax.nn.softmax(self._clamp_belief_logits(belief, puzzle), axis=-1)
            probs = jnp.zeros((*puzzle.shape, self.config.vocab_size), dtype=jnp.float32)
            return probs.at[..., 2:11].set(digit_probs)

        def objective(belief: Array) -> tuple[Array, dict[str, Array]]:
            clamped = self._clamp_belief_logits(belief, puzzle)
            candidate_probs = candidate_probs_from_belief(clamped)
            energy = jnp.mean(self.verifier_energy_from_probs(puzzle, candidate_probs))
            given_logits = self._belief_to_token_logits(clamped, puzzle, jnp.asarray(self.recurrent_steps - 1))
            given_mask = self.condition_mask(puzzle).astype(jnp.float32)
            log_probs = jax.nn.log_softmax(given_logits, axis=-1)
            target_log_prob = jnp.take_along_axis(log_probs, puzzle[..., None], axis=-1).squeeze(-1)
            given_loss = -jnp.sum(target_log_prob * given_mask) / jnp.maximum(jnp.sum(given_mask), 1.0)
            prior = jnp.sum(jnp.square((clamped - belief0) * unknown)) / jnp.maximum(jnp.sum(unknown) * 9.0, 1.0)
            loss = energy + given_weight * given_loss + prior_weight * prior
            return loss, {
                "belief_refine_energy": energy,
                "belief_refine_given_loss": given_loss,
                "belief_refine_prior_loss": prior,
            }

        def refine_step(belief: Array, _step_index: Array):
            (loss, components), grad = jax.value_and_grad(objective, has_aux=True)(belief)
            grad = grad * unknown
            grad_norm = jnp.linalg.norm(grad.astype(jnp.float32), axis=-1, keepdims=True)
            grad_scale = jnp.minimum(1.0, grad_clip_norm / (grad_norm + 1e-6))
            next_belief = self._clamp_belief_logits(belief - step_size * grad * grad_scale, puzzle)
            return next_belief, (
                loss,
                components["belief_refine_energy"],
                components["belief_refine_given_loss"],
                components["belief_refine_prior_loss"],
                jnp.mean(grad_norm),
            )

        belief, (loss, energy, given_loss, prior, grad_norm) = jax.lax.scan(
            refine_step,
            belief0,
            jnp.arange(steps, dtype=jnp.int32),
        )
        return belief, {
            "belief_refine_loss": jnp.mean(loss),
            "belief_refine_energy": jnp.mean(energy),
            "belief_refine_given_loss": jnp.mean(given_loss),
            "belief_refine_prior_loss": jnp.mean(prior),
            "belief_refine_grad_norm": jnp.mean(grad_norm),
        }

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        return self.forward_final(tokens, train=train, dropout_key=dropout_key)

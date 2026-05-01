# Latent Factor Recurrent Model Architecture

LFRM is the only active model path in this repository. It should be read as a
generalized Universal Transformer-style recurrent solver: a shared block is
applied repeatedly, but the recurrent state is no longer just an unstructured
hidden tensor.

## State

LFRM recurses over:

```text
S^t = (H^t, L^t, Q^t, Z^t)
Q^t = softmax(L^t / temperature)
```

`H` is the cell hidden canvas, `L` is the residual belief-logit canvas, `Q` is
the discrete belief canvas, and `Z` is the dynamic latent factor array. The
latent factors are pure slots with shape `(batch, slots, d_model)`: factor
identity is not bound to a digit or color.

Sudoku tokens remain compatible with the dataset format:

```text
blank = 1
digits = 2..10
belief_dim = vocab_size - 2
```

Given cells are clamped every recurrent step. This is input conditioning, not a
Sudoku rule.

## Update

Each recurrent step follows a Perceiver IO pattern and then performs belief
refinement:

1. A local grid mixer reads the generic 3x3 spatial neighborhood.
2. Shared cell-symbol micro-tokens expose per-channel evidence without using
   digit/color embeddings.
3. Latent factors read from those micro-tokens through symbol-equivariant
   multi-head cross-attention.
4. The latent array is processed by shared multi-head latent self-attention
   blocks.
5. Cell-symbol queries read back from latent factors, producing
   symbol-conditioned messages `g_{i,k}`.
6. A shared per-symbol scorer emits residual belief-logit updates.
7. A task-agnostic quality head scores the current recurrent step from pooled
   hidden state, latent factors, and belief statistics.

No digit/color embedding is used. The same scorer is applied to every symbol
channel, so permuting symbols in the input permutes output symbol logits in the
same way.

The current Sudoku research config uses `num_heads = 8`, `num_slots = 64`,
`d_model = 256`, and two latent processor layers. Smaller values are still
available through CLI overrides for smoke tests.

## Regularization And Metrics

The model reports:

- `q_loss`, the quality-head loss against per-step target token accuracy
- `q_selected_blank_ce_loss`, `q_selected_blank_cell_accuracy`, and
  `q_selected_solved_rate`
- `q_selected_step` and `oracle_step`
- `slot_consistency_loss`
- `slot_usage_entropy`
- `slot_diversity_loss`
- `terminal_belief_delta`, the RMS one-step terminal belief change
- `terminal_belief_mse`, the squared residual used by terminal-residual loss
- `belief_entropy` and `belief_confidence`

Slot consistency encourages latent factor assignment patterns to evolve
smoothly across recursive steps. Slot diversity discourages multiple slots from
attending to identical token patterns.

## Removed Baselines

The old Universal Transformer and Recurrent Transformer implementations were
removed after they served their purpose as recurrent-reasoning baselines. The
current code treats LFRM itself as the shared-block multi-step recurrent
reasoning framework, with the main research question shifted to the structure of
the recurrent state space: belief canvas plus dynamic latent factors.

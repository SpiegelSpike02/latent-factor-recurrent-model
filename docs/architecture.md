# Latent Factor Recurrent Model Architecture

LFRM is the only active model path in this repository. It should be read as a
generalized Universal Transformer-style recurrent solver: a shared block is
applied repeatedly, but the recurrent state is no longer just an unstructured
hidden tensor.

## State

LFRM recurses over:

```text
S_b^t = (H_b^t, L_b^t, Q_b^t, Z_b^t)
Q_b^t = softmax(L_b^t / temperature)
```

`b` indexes parallel hypotheses. `H` is the cell hidden canvas, `L` is the
residual belief-logit canvas, `Q` is the discrete belief canvas, and `Z` is the
dynamic latent factor array. The latent factors are slot-by-symbol tensors, so
symbol permutations act by permuting the belief and latent symbol axes together.

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
2. Latent factors read from cells through symbol-equivariant cross-attention.
3. The latent array is processed by a shared latent self-attention block.
4. Cells read back from latent factors through slots-to-cells cross-attention.
5. A shared per-symbol scorer emits residual belief-logit updates.

No digit/color embedding is used. The same scorer is applied to every symbol
channel, so permuting symbols in the input permutes output symbol logits in the
same way.

## Branches And Energy

LFRM maintains several branches in parallel. Training uses a branch-softmin CE
over blank cells so a good hypothesis can receive credit without hard-coded
search or backtracking. A learned energy head scores each branch; inference and
standard metrics use the lowest-energy branch.

The energy head sees `(Q, H, Z)` and is trained with a margin objective against
target and corrupted belief canvases. It is a learned verifier, not a Sudoku
checker.

## Regularization And Metrics

The model reports:

- `branch_min_ce` and `branch_mean_ce`
- `selected_branch_energy`
- `energy_margin_loss`
- `slot_consistency_loss`
- `slot_usage_entropy`
- `branch_diversity`
- `terminal_belief_delta`, the RMS one-step terminal belief change
- `terminal_belief_mse`, the squared residual used by terminal-residual loss
- `belief_entropy` and `belief_confidence`

Slot consistency encourages latent factors to keep a stable identity across
recursive steps. Slot usage entropy discourages all cells from collapsing onto a
small number of slots.

## Removed Baselines

The old Universal Transformer and Recurrent Transformer implementations were
removed after they served their purpose as recurrent-reasoning baselines. The
current code treats LFRM itself as the shared-block multi-step recurrent
reasoning framework, with the main research question shifted to the structure of
the recurrent state space: belief canvas plus dynamic latent factors.

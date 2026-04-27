# Recurrent Grid Reasoning Architecture

This repository now supports two recurrent model families for Sudoku:

- `universal_transformer`: a shared recurrent block with configurable
  communication and an optional damped transition.
- `recurrent_transformer`: a recurrent Transformer with configurable
  communication, standard pre-norm residual updates, and separate parameters
  for each recurrent step.

Both variants share the same data pipeline, step-wise supervision, Sudoku
validity loss, effective given logits, wandb logging, and checkpointing. This
keeps ablations focused on the architecture rather than the training harness.

## Data Signal

The dataset provides:

- `inputs`: Sudoku board tokens
- `labels`: solution tokens
- `given_mask`: whether each cell is originally given

Training supervises blank cells only:

```text
loss_mask = NOT given_mask
```

## Recurrent Grid Skeleton

Both model families operate over a recurrent grid state:

- fixed recurrent depth
- step embedding conditions every iteration without being written into the
  persistent state by itself
- optional inner compute depth inside each recurrent step

In shorthand:

```text
h_0 = embed(tokens) + row + col + box + cell-type embedding
for k in 1..T:
  h_k = recurrent_step(h_{k-1}, step_embed[k], ...)
logits = output_head(norm(h_T))
```

The `universal_transformer` family uses one shared recurrent step, while the
`recurrent_transformer` family uses step-specific blocks.

Training uses the same classification head at every recurrent step:

```text
logits_k = output_head(norm(h_k))
loss = sum_k w_k * CE(logits_k, labels)
```

The weights are uniform across recurrent depth, so every recurrent step is directly
encouraged to move toward a valid solution.

## Inner Compute Depth

Each outer recurrent step can contain a deeper shared compute loop:

```text
for k in 1..T:
  for i in 1..inner_steps:
    if reinject_input:
      h = h + input_embedding
    for j in 1..layers_per_step:
      h = block_j(h, step_embed[k])
    if i is before the final grad_inner_steps:
      h = stop_gradient(h)
  h_k = h
```

This is intentionally named as compute scheduling rather than as a high/low
hierarchy. It increases forward reasoning depth while limiting backward depth to
the final `grad_inner_steps` inside each outer step.

## Universal Transformer Block

When `model_type = "universal_transformer"`, the same block is reused at every
recurrent step. With damped updates enabled, this shared block uses a damped
candidate-state transition:

1. Step-Conditioned Communication
   The step embedding `e_k` is added to normalized sublayer inputs, not directly
   to `h_k`. This lets the step index modulate computation while preserving
   `||h_{k+1} - h_k||` as a meaningful convergence signal.

2. Typed Relation Propagation
   Messages propagate along:
   - row relations
   - column relations
   - box relations
   - optional lightweight global relations
   Relation matrices remove self edges before row-normalization, so propagation
   carries neighbor constraint information rather than duplicating the residual
   self path.
   The four relation streams are fused nonlinearly so that row/column/box
   constraints can interact instead of only adding linearly.

3. Light Read Scaling
   `u_k = h_k + rho_k * p_k`

4. Candidate Construction
   `z_k = u_k + FFN(LN(u_k))`

5. Damped State Update
   `h_{k+1} = h_k + alpha_k * (z_k - h_k)`

`rho_k` is a continuous read scaling term. It is not a read gate.

`alpha_k` is a damping / step-size coefficient. It is not a write gate.
It is conditioned on the step embedding, but there is no hand-written decay
schedule; any convergence pattern must be learned.

## Confidence Signal

Each recurrent step derives a token-wise uncertainty signal from the current
classifier entropy:

```text
entropy_k = H(softmax(output_head(norm(h_k))))
```

This entropy is fed into the `rho_k` / `alpha_k` predictor so that uncertain
cells can update more aggressively while already-settled cells can damp their
updates.

The entropy is used as an auxiliary signal only. It does not introduce a new
halting network or continue gate, and it is stop-gradient so the scale
predictor cannot exploit it as a shortcut.

## Sudoku-Specific Task Semantics

- The propagation module is residual: it emits an increment rather than
  overwriting the current state.
- `[model.clues].freeze_state = true` treats given cells as fixed internal clue states
  by suppressing their recurrent updates.
- `[model.clues].fix_outputs = true` treats given cells as fixed output clues by replacing
  their logits with the input token at loss/eval time.
- The global relation path starts weak so it complements, rather than overrides,
  row/column/box propagation.

## Why This Structure

For Sudoku, the important interactions live on a fixed constraint graph. Dense
all-to-all self-attention is therefore not the right primary propagation
mechanism. Typed relation propagation makes the information flow follow the real
row / column / box constraints while keeping the recurrent UT skeleton intact.

The same block interface can later host a different propagation module for ARC,
for example an object-centric or learned relation propagation layer.

## Recurrent Transformer Baseline

When `model_type = "recurrent_transformer"`, the model uses a sequence of
step-specific residual blocks. Each step receives both the current recurrent
state and the initial embedded input state:

```text
q_k = h_k + W_in([h_k; h_0])
u_k = q_k + Communication(LN(q_k) + e_k)
h_{k+1} = u_k + FFN_k(LN(u_k) + e_k)
```

The recurrent Transformer therefore performs iterative refinement, but unlike
the Universal Transformer it does not reuse the same block at every step:

```text
h_{k+1} = F_{theta_k}(h_k, h_0, e_k)
```

This makes it the conventional unshared-parameter recurrent comparison. It does
not expose `rho` or `alpha`, because those are damped-update concepts.
Evaluation still reports per-step loss, validity, and hidden-state delta so both
model families can be compared directly.

For comparison, the Universal Transformer family uses one shared recurrent
operator:

```text
x_k = h_k
u_k = x_k + Communication(LN(x_k) + e_k)
h_{k+1} = u_k + FFN(LN(u_k) + e_k)
```

or, with damped updates enabled, the corresponding `rho` / `alpha` transition.

## Ablation Switches

The main architectural and task-semantics switches are:

- `model_type = "universal_transformer" | "recurrent_transformer"`
- `communication_type = "relation" | "attention"`
- `[model.transition].type = "residual" | "damped"` selects the recurrent
  transition rule. The damped transition is only supported for
  `model_type = "universal_transformer"`.
- `[model.transition].hidden_dim` controls the `rho` / `alpha` scale predictor
  when the damped transition is enabled.
- `[model.attention].num_heads` controls dense attention heads and only affects
  `communication_type = "attention"`.
- `[model.relation].include_global` controls the optional global relation
  path and only affects `communication_type = "relation"`.
- `[model.clues].use_type_embedding = true | false`
- `[model.clues].fix_outputs = true | false`
- `[model.clues].freeze_state = true | false`
- `[model.compute].inner_steps` controls repeated compute inside each recurrent
  step.
- `[model.compute].layers_per_step` controls the block stack depth reused inside
  each inner step.
- `[model.compute].grad_inner_steps` keeps gradients only through the final
  inner compute steps.
- `[model.compute].reinject_input = true | false` controls whether the original
  input embedding is reintroduced inside each inner step.
- `validity_loss_weight = 0.0` disables the differentiable Sudoku legality loss
- `step_loss_weighting = "uniform" | "linear" | "final"` controls how much
  supervision each recurrent step receives; later-weighted modes reduce the
  chance that early unfinished reasoning steps dominate the reported loss
- `[train.ema].enabled = true | false` controls whether eval/checkpoint weights
  use an exponential moving average of the trained parameters. EMA does not
  change the raw training update; it is a stability lens for evaluation.

Communication and task-semantics switches are independent. For example,
`clues.freeze_state` can be used with either model family and either
communication type; it is a Sudoku clue-state handling choice, not a property of
one model family. Damped updates are specific to the `universal_transformer`
family.

# BRC Architecture

BRC means Belief-Controller Reasoner. The central split is:

```text
B = explicit output belief / working answer state
H = spatial hidden field / local computation state
z = small latent controller / update modulation
G = relation schema / communication graph
E = learned verifier or energy model
```

Anything that grows with the output size belongs in `B` or `H`. The latent `z`
must stay small and only control how the solver updates state. It should not
store a compressed answer, act as a slot cache, or directly decode output cells.

## Sudoku MVP

The current implemented path is `brc_sudoku`:

```text
C = puzzle givens
G = self, same_row, same_col, same_box
B_t = [batch, 81, 9] digit belief logits
P_t = softmax(B_t / tau_t) soft digit distribution
H_t = [batch, 81, d_model] recurrent spatial field
z = z_global + q(C)
E = independent relation-typed verifier
```

Each recurrent step embeds the puzzle, given flags, position/schema ids, time,
and `P_t`. A shared relation-typed solver block performs message passing over
the Sudoku schema and updates `H`; the token head emits a belief-logit delta,
and givens are clamped back into `B`.

`B_t` is the recurrent answer state and is updated additively as logits. `P_t`
is derived from `B_t` only when a probability distribution is needed, such as
for soft draft embeddings or soft verifier inputs. Given cells are clamped in
logit space with large finite logits for the clue digit and low finite logits
for the other digits.

The solver avoids early hard locking. `B` stays continuous through the recurrent
loop, with sharpening only near the final steps. Hard boards are decoded at the
end or supplied to the verifier for ranking.

## Controller

The controller latent is intentionally weak:

```text
allowed: FiLM scale/shift, recurrent gates, conservative latent fitting
disallowed: direct output logits, large key-value memory, slot-style answer cache
```

Sudoku has fixed rules, so the main inference state is `B`, not `z`. The code
keeps verifier-guided latent fitting optional and conservative through low inner
step counts, gradient clipping, and a prior penalty.

For the Sudoku MVP, verifier-guided latent fitting is not the main path.
`meta_outer_loss` is currently disabled in the default BRC config; z fitting is a
later ablation after the B/H recurrent belief solver and verifier ranking are
healthy.

## Verifier

The verifier has its own relation-typed recurrent core. It may share the
high-level schema but not the solver recurrent block, so it is less likely to
inherit the generator's exact failure modes.

It supports hard candidates for ranking and soft candidate distributions for
gradient-based belief refinement. Training negatives include random/corrupted
boards plus detached model early/final samples.

The verifier margin loss trains the verifier by enforcing:

```text
E(puzzle, true_solution) < E(puzzle, fake_candidate)
```

Model-generated fake candidates are detached for this loss. The margin loss is
not used as a generator objective, because backpropagating it into the generator
would push the generator to increase the energy of its own fake samples. If the
generator uses verifier signal, it should do so through a separate energy
minimization or belief refinement objective:

```text
minimize_B E(puzzle, softmax(B))
```

When belief refinement is enabled, verifier training should include soft model
beliefs in addition to hard boards; otherwise gradients through soft candidates
may be unreliable.

## Training

BRC-Sudoku uses:

- a single step-weighted solution CE schedule over recurrent steps
- mixed belief starts: full mask, partially masked true solution,
  self-conditioned model belief, and corrupted solution belief
- digit permutation augmentation
- verifier margin loss with hard negatives
- optional meta loss after conservative verifier-guided latent fitting

The main CE is computed on the original unknown cells. Givens are always visible
as conditions and are clamped during belief updates; they are used for
consistency checks rather than as the dominant supervised signal. This avoids
inflating training or eval metrics by rewarding clue copying.

`model.brc.step_loss_weights` controls the recurrent supervision schedule.
Weights are normalized internally and the default schedule increases toward late
steps, so early steps learn useful intermediate belief while late/final steps
carry the strongest final-solve pressure.

The denoising loss is answer-belief denoising, not clue denoising. Full puzzle
givens remain visible as conditions; the corrupted or masked object is the
answer belief draft. In particular, BRC-Sudoku does not mask out a subset of
givens and train the model to predict those clues.

Digit permutation augmentation is a default training condition, not a cosmetic
augmentation. Sudoku digits are symbols, so the model should not rely on fixed
semantic differences between labels like 1 and 9.

Sudoku datasets are assumed to contain unique-solution puzzles. If a dataset can
have multiple valid completions, exact match CE and full-board exact accuracy
must be interpreted carefully and supplemented with valid-solution metrics.

Core metrics are full-board solved rate, given consistency, invalid-board rate,
row/column/box conflict count, verifier ranking accuracy, and oracle-step
diagnostics. For multi-candidate inference, the key diagnostic is oracle top-k
accuracy versus verifier top-1 accuracy:

```text
oracle top-k high, verifier top-1 low -> generator can produce the solution, ranker is weak
oracle top-k low                     -> recurrent belief solver/generator is weak
```

Cell accuracy is treated as secondary because it can look high even when
full-board exact accuracy is poor.

## Generalization Target

Sudoku is the fixed-rule, fixed-size first case. The same state split is meant
to extend to:

```text
Maze:   local relation schema, variable-size B/H, weak z
ARC:    demo-conditioned z_episode, output belief B, verifier/reranking
```

The rule of thumb is:

```text
single fixed-rule instance -> refine B first
few-shot demonstrations    -> fit z_episode, then refine B
```

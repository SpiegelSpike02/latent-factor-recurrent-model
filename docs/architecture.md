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
B_t = [batch, 81, 9] soft digit belief logits
H_t = [batch, 81, d_model] recurrent spatial field
z = z_global + q(C)
E = independent relation-typed verifier
```

Each recurrent step embeds the puzzle, given flags, position/schema ids, time,
and the current soft belief. A shared relation-typed solver block performs
message passing over the Sudoku schema and updates `H`; the token head emits a
belief delta, and givens are clamped back into `B`.

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
step counts, gradient clipping, update clipping, and a prior penalty.

## Verifier

The verifier has its own relation-typed recurrent core. It may share the
high-level schema but not the solver recurrent block, so it is less likely to
inherit the generator's exact failure modes.

It supports hard candidates for ranking and soft candidate distributions for
gradient-based belief refinement. Training negatives include random/corrupted
boards plus model early/final samples.

## Training

BRC-Sudoku uses:

- a single step-weighted unknown-cell CE schedule over recurrent steps
- mixed belief starts: full mask, partially masked true solution,
  self-conditioned model belief, and corrupted solution belief
- digit permutation augmentation
- verifier margin loss with hard negatives
- optional meta loss after conservative verifier-guided latent fitting

Core metrics are full-board solved rate, given consistency, invalid-board rate,
row/column/box conflict count, verifier ranking accuracy, and oracle-step
diagnostics. Cell accuracy is treated as secondary because it can look high even
when full-board exact accuracy is poor.

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

# BDR Architecture

BDR means Belief Dynamics Reasoner. The name is meant to describe the current
model directly: the recurrent computation learns dynamics over an explicit
answer belief rather than routing through a separate controller state.

```text
z_t = centered answer logits, shape [batch, cells, symbols]
q_t = softmax(z_t), the probability view of the answer state
h_t = recurrent hidden grid field
C   = fixed puzzle/context tokens and given-mask information
F   = shared local/global recurrent solver
```

The persistent answer state is `z_t`; `q_t` is derived when the model needs a
probability distribution. Keeping logits as the stored coordinate avoids simplex
constraints during updates, while keeping `q_t` as the semantic view gives the
loss and diagnostics direct probabilistic meaning.

## Sudoku Path

The current implemented path is `bdr_sudoku`:

```text
C         = puzzle givens
z_0       = zero centered logits, equivalent to uniform q_0
q_t       = softmax(z_t)
h_t       = [batch, 81, hidden_dim] recurrent spatial field
read_t    = F(C, q_t, h_t)
z_{t+1}   = update(z_t, read_t)
```

Each commit step embeds the puzzle context and a compact feature view of the
current belief. The shared solver mixes information with global attention and a
local convolutional SwiGLU block. The read state is then converted into one of
two update rules.

Velocity mode learns a direct centered logit-space vector field:

```text
z_{t+1} = center(z_t + eta * v(read_t))
```

Energy mode learns a scalar energy over candidate distributions and follows the
negative energy gradient in the logit coordinate:

```text
E_t(q)     = energy(read_t, q)
g_t        = center(dE_t(q_t) / dq_t)
z_{t+1}    = center(z_t - eta * g_t)
q_{t+1}    = softmax(z_{t+1})
```

This makes energy mode closer to mirror descent or exponentiated-gradient
dynamics. It is not a proof of global convergence, because `read_t` is
recomputed after every step, but it gives the model a useful language for fixed
points, wrong attractors, and recovery directions.

## State Semantics

BDR does not treat logits and probabilities as interchangeable names for the
same thing. They play different roles:

- `z_t` is the stored dual coordinate used for additive updates.
- `q_t` is the probability view used for targets, confidence, energy candidates,
  and most diagnostics.
- `h_t` is the hidden computation field, not an answer cache.

The initial answer state is exactly uniform because `z_0` is all zeros. This is
kept as logits rather than as a mutable probability array because it makes both
velocity and energy updates unconstrained while preserving the same semantic
initial belief.

## Training

BDR-Sudoku uses step-carry training. Instead of unrolling all commit steps inside
one large training example, the training loop carries `z` and `h` across
optimizer updates and resets examples when a new puzzle is loaded or when the
early-stop diagnostic marks the current state stable.

The main loss is cross entropy on supervised cells. Sudoku givens are always
available as conditioning context; they are reported through consistency metrics
rather than used to inflate a clue-copying score. The current energy
configuration also supports:

- fixed-point update loss near the target state
- wrong-attractor rank, direction, and nonzero-update losses
- corrupted-recovery loss from synthetic corrupted states
- path-energy and update-size diagnostics

Digit permutation augmentation is a default training condition for Sudoku,
because digit labels are symbols rather than ordered semantic values.

## Metrics

The primary training diagnostics are query accuracy, query target probability,
exact accuracy, Sudoku conflict count, update RMS, distribution total-variation
movement, and energy-gradient RMS. Exact solving remains the strict metric; cell
accuracy and query probability are useful mainly for understanding whether the
belief dynamics are improving before complete boards are solved.

For energy mode, the wrong-attractor losses are especially important. If the
target energy is not below carried wrong states, the learned energy landscape is
not yet aligned with the desired fixed point even if CE improves.

## Generalization Target

Sudoku is the fixed-size first case. The same state split is intended to extend
to other grid tasks:

```text
Maze: output belief over path/non-path or cell labels, spatial hidden field
ARC:  demo-conditioned context plus explicit output-grid belief
```

The common rule is: keep the answer-sized object explicit, keep the hidden field
for computation, and make the learned dynamics accountable through probabilities
and update diagnostics.

# BDR Architecture

BDR means Belief Dynamics Reasoner. The name is meant to describe the current
model directly: the recurrent computation learns dynamics over an explicit
answer belief rather than routing through a separate controller state.

```text
q_t = explicit answer distribution, shape [batch, cells, symbols]
h_t = recurrent hidden grid field
C   = fixed puzzle/context tokens and given-mask information
F   = shared local/global recurrent solver
```

The persistent answer state is the probability distribution `q_t`. Updates may
still be parameterized as direct proposals, scalar energy gradients, or velocity
fields, but the carried belief remains an explicitly normalized distribution.

## Sudoku Path

The current implemented path is `bdr_sudoku`:

```text
C         = puzzle givens
q_0       = uniform answer distribution
h_t       = [batch, 81, hidden_dim] recurrent spatial field
read_t    = F(C, q_t, h_t)
q_{t+1}   = update(q_t, read_t)
```

Each commit step embeds the puzzle context and a compact feature view of the
current belief. The shared solver mixes information with global attention and a
local convolutional SwiGLU block. The read state is then converted into one of
three update rules.

Proposal mode directly emits the next belief:

```text
q_{t+1} = softmax(proposal(read_t))
```

Energy-gradient mode learns a scalar energy over candidate distributions and
follows the negative energy gradient:

```text
E_t(q)     = energy(read_t, q)
g_t        = center(dE_t(q_t) / dq_t)
q_{t+1}    = normalize(q_t - g_t)
```

Velocity mode learns a free centered vector field over the belief:

```text
v_t       = center(velocity(read_t))
q_{t+1}   = normalize(relu(q_t + v_t) + eps)
```

This keeps proposal as the direct baseline while preserving two higher-value
dynamical routes: scalar energy descent and learned velocity fields.

## State Semantics

BDR keeps the answer state and hidden computation separate:

- `q_t` is the explicit answer distribution used for targets, confidence,
  energy candidates, and most diagnostics.
- `h_t` is the hidden computation field, not an answer cache.

The initial answer state is exactly uniform. This makes every update rule start
from the same semantic belief.

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

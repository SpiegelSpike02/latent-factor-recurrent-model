# Recurrent Grid Reasoning

Recurrent Grid Reasoning is a JAX + `flax.nnx` research codebase for
recurrent reasoning on 2D grid and image-like tasks such as Sudoku, with
ARC-style extensions in mind.

The codebase currently provides two aligned base models:

- `universal_transformer`: a shared recurrent block with an optional damped
  transition using `rho` and `alpha`
- `recurrent_transformer`: step-specific pre-norm recurrent Transformer blocks
  that condition on both the current state and the initial input state

The communication operator is independently configurable:

- `relation`: typed row / column / box / optional global relation communication
- `attention`: dense global self-attention

This keeps model-family and communication ablations separate instead of binding
all Sudoku-specific choices to one model variant.

Common features include:

- shared recurrent block across reasoning depth
- learned 2D row / column / box positional embeddings
- cell-type embeddings for Sudoku clue/blank awareness
- optional effective logits that hard-fix given cells to their clues
- step-wise blank-cell supervision
- configurable inner compute depth with local gradient horizon
- differentiable Sudoku validity loss
- RMSNorm + SwiGLU
- configurable dropout with explicit JAX RNG handling under `nnx.jit`
- Orbax checkpoints

When `[model.transition].type = "damped"`, `universal_transformer`
additionally uses entropy-conditioned `rho` / `alpha` prediction for uncertain
vs settled cells.

A short architecture overview lives in [docs/architecture.md](docs/architecture.md).

## Quickstart

Install dependencies:

```bash
uv sync
```

Build an offline Sudoku dataset from local `train.csv` / `test.csv` files:

```bash
uv run rgr-build-sudoku --source-csv-dir path/to/sudoku_csvs --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

Train from the default Sudoku config:

```bash
uv run rgr-train --config configs/sudoku.toml
```

Train the recurrent Transformer config:

```bash
uv run rgr-train --config configs/sudoku_recurrent_transformer.toml
```

CLI flags still override config values:

```bash
uv run rgr-train --config configs/sudoku.toml --learning-rate 1e-4 --batch-size 16
```

For recurrent runs, `--step-loss-weighting linear` gives later reasoning steps
more influence while still supervising earlier steps. Use `final` to train only
against the last step, or `uniform` to restore the original equal weighting.

## Notes

- Sudoku data is built offline and sampled at runtime.
- The current builder writes a `given_mask.npy`; training derives blank-cell supervision by inverting it.

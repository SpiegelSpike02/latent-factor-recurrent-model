# Latent Factor Recurrent Model

Latent Factor Recurrent Model is a JAX + `flax.nnx` research codebase for
recurrent reasoning on 2D grid tasks. The repository now has one primary model
path:

- `lfrm`: a Perceiver IO-style multi-head latent bottleneck modified into a
  recurrent belief solver with dynamic latent factors and symbol-equivariant
  updates.

The older Universal Transformer and Recurrent Transformer implementations have
been removed. LFRM is the generalized shared-block recurrent reasoning path:
the same block is applied for multiple refinement steps, but the state being
refined is structured as cell hidden state, residual belief logits, a discrete
belief canvas, and latent factor slots.

LFRM intentionally does not consume Sudoku row/column/box relation matrices,
Sudoku unit matrices, hand-written validity losses, or task-specific solver
rules. Sudoku is currently the first smoke-test data format for a broader
task-agnostic grid solver.

A short architecture overview lives in [docs/architecture.md](docs/architecture.md).

## Source Layout

The implementation lives under the `lfrm` package:

- `lfrm.models`: LFRM and TRM models
- `lfrm.datasets`: generic grid dataset loading plus Sudoku/Maze dataset building
- `lfrm.training`: optimizer, loss, metrics, and checkpoint helpers
- `lfrm.scripts`: console-script entry points

## Quickstart

Install dependencies:

```bash
uv sync
```

Build an offline Sudoku dataset from local `train.csv` / `test.csv` files:

```bash
uv run lfrm-build-sudoku --source-csv-dir path/to/sudoku_csvs --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

Build an offline Maze dataset in the HRM/TRM format:

```bash
uv run lfrm-build-maze --output-dir data/maze-30x30-hard-1k
```

Train LFRM:

```bash
uv run lfrm-train --config configs/sudoku_lfrm.toml
```

CLI flags override config values:

```bash
uv run lfrm-train --config configs/sudoku_lfrm.toml --learning-rate 1e-4 --batch-size 16
```

Recurrent supervision is controlled by `dense_loss_weight`,
`final_loss_weight`, and `sequence_loss_weight`. Dense loss applies token-level
CE across all recurrent steps, final loss emphasizes the last step, and sequence
loss adds a set-style blank-token objective. `q_loss_weight` trains a
task-agnostic per-step quality head from target token accuracy, and evaluation
reports both the final step and the step selected by that head.

The package also applies project-level JAX defaults before JAX initializes:
Triton GEMM is enabled with `--xla_gpu_triton_gemm_any=true`, GPU preallocation
is set to 95% with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, preallocation is
explicitly enabled, and `TF_GPU_ALLOCATOR=cuda_malloc_async` selects CUDA async
allocation. Values already set in the shell are preserved.

## Notes

- Datasets are built offline with `train/` and `test/` splits and sampled at runtime.
- Sudoku derives blank-cell supervision from token `1`; Maze writes an explicit
  all-False `given_mask.npy` so every output token is supervised.
- Old UT/RT checkpoints and configs are not compatible with the current
  LFRM-only code path.

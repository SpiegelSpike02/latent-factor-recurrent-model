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

- `lfrm.models`: the LFRM model
- `lfrm.datasets`: generic grid dataset loading plus Sudoku dataset building
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

Train LFRM:

```bash
uv run lfrm-train --config configs/sudoku_lfrm.toml
```

CLI flags override config values:

```bash
uv run lfrm-train --config configs/sudoku_lfrm.toml --learning-rate 1e-4 --batch-size 16
```

For recurrent supervision, `--step-loss-weighting` can be `final`, `uniform`,
or `linear`. `uniform` and `linear` apply token-level CE across all recurrent
steps, which is useful for dense supervision and diagnosing refinement dynamics.
`--q-loss-weight` trains a task-agnostic per-step quality head from target
token accuracy, and evaluation reports both the final step and the step selected
by that head.

The package also applies project-level JAX defaults before JAX initializes:
Triton GEMM is enabled with `--xla_gpu_triton_gemm_any=true`, GPU preallocation
is set to 95% with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, preallocation is
explicitly enabled, and `TF_GPU_ALLOCATOR=cuda_malloc_async` selects CUDA async
allocation. Values already set in the shell are preserved.

## Notes

- Sudoku data is built offline and sampled at runtime.
- The builder writes `given_mask.npy`; training derives blank-cell supervision
  by inverting it.
- Old UT/RT checkpoints and configs are not compatible with the current
  LFRM-only code path.

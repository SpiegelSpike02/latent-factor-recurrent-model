# Recurrent Grid Reasoning

This is a JAX + `flax.nnx` research codebase for recurrent reasoning on 2D grid
tasks. The current Sudoku MVP is BRC-Sudoku, a belief-first recurrent solver
that keeps working memory out of the latent controller.

- `brc_sudoku`: the current Sudoku path. It uses digit belief logits, recurrent
  spatial hidden state, relation-typed attention, a small controller latent, and
  an independent verifier/energy model.
- `trm`: a Tiny Recursive Model baseline.

The BRC design separates state by capacity: size-growing information lives in
`B` (explicit output belief) and `H` (spatial hidden field), while `z` is a small
controller that only modulates update dynamics. Sudoku supplies a relation
schema `G` with `self`, `same_row`, `same_col`, and `same_box`; the learned
verifier `E` ranks hard candidates and can conservatively refine belief logits.

BRC-Sudoku does not use clue-dropout pseudo-labels, symbolic traces, DSL rules,
or a hand-written Sudoku checker in the loss. Training uses a single
step-weighted unknown-cell CE schedule with stronger late-step weights, mixed
answer-belief starts (`full_mask`/teacher/self-conditioned/corrupt), digit
permutation augmentation, and verifier hard negatives. Verifier margin training
uses detached model-generated fakes; generator-side verifier use is handled
separately through energy minimization or belief refinement.

A short architecture overview lives in [docs/architecture.md](docs/architecture.md).

## Source Layout

The implementation lives under the `lfrm` package:

- `lfrm.models`: BRC-Sudoku and TRM models
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
uv run lfrm-build-maze --output-dir data/maze-30x30-hard-1k-aug --aug
```

Train BRC-Sudoku:

```bash
uv run lfrm-train --config configs/sudoku_brc.toml
```

CLI flags override config values:

```bash
uv run lfrm-train --config configs/sudoku_brc.toml --learning-rate 1e-4 --batch-size 16
```

BRC-Sudoku and TRM dense-unroll recurrent supervision are controlled by
`model.brc.step_loss_weights` and `model.trm.step_loss_weights`: BRC uses one
relative CE weight per recurrent step, while TRM uses one relative CE weight per
rollout/output step. The weights are normalized internally. BRC-Sudoku additionally
reports given consistency, invalid-board rate, row/column/box conflict count,
verifier ranking accuracy, and belief/refinement diagnostics.

The package also applies project-level JAX defaults before JAX initializes:
Triton GEMM is enabled with `--xla_gpu_triton_gemm_any=true`, the XLA GPU
latency hiding scheduler is enabled, GPU preallocation is set to 95% with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, and preallocation is explicitly enabled.
Values already set in the shell are preserved. `TF_GPU_ALLOCATOR` is not set by
default because JAX documents `cuda_malloc_async` as a growing memory pool rather
than the fixed preallocation behavior expected for these training runs.

## Notes

- Datasets are built offline with `train/` and `test/` splits and sampled at runtime.
- Sudoku derives blank-cell supervision from token `1`; Maze writes an explicit
  `given_mask.npy` so known walls, start, and goal cells are treated as givens,
  while open cells are supervised as path/non-path decisions.
- Old Sudoku LFRM/Mini-GLIDER configs are not compatible with the current
  BRC-Sudoku path. The TRM Sudoku baseline config is still available.

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

Build ARC-AGI datasets in the official TRM/URM format. If the source JSON files
are not present locally, `lfrm-build-arc` downloads them first and caches them
under `data/raw/arc-agi/`. Set `HF_ENDPOINT=https://hf-mirror.com` to use a
HuggingFace mirror.

```bash
uv run lfrm-build-arc \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000

uv run lfrm-build-arc \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2 \
  --test-set-name2 evaluation \
  --num-aug 1000
```

Train BRC-Sudoku:

```bash
uv run lfrm-train --config configs/sudoku_brc.toml
```

CLI flags override config values:

```bash
uv run lfrm-train --config configs/sudoku_brc.toml --learning-rate 1e-4 --batch-size 16
```

BRC-Sudoku recurrent supervision is controlled by `model.brc.step_loss_weights`,
one relative CE weight per recurrent step. The weights are normalized internally.
TRM/URM official ACT configs do not use step-weighted dense supervision; dense
unroll remains a separate experimental path. BRC-Sudoku additionally reports
given consistency, invalid-board rate, row/column/box conflict count, verifier
ranking accuracy, and belief/refinement diagnostics.

The package also applies project-level JAX defaults before JAX initializes:
Triton GEMM is enabled with `--xla_gpu_triton_gemm_any=true`, the XLA GPU
latency hiding scheduler is enabled, fixed GPU preallocation is disabled with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, and
`TF_GPU_ALLOCATOR=cuda_malloc_async` selects CUDA's growing async memory pool.
For single-host multi-GPU runs, the documented NCCL `LL/LL128/SIMPLE` protocol
flags are also set. These defaults intentionally override stale shell values;
set `LFRM_RESPECT_EXTERNAL_JAX_ENV=true` to preserve externally supplied JAX
memory settings. Experimental XLA flags are not enabled by default because
unsupported flags are fatal at process startup; append local experiments with
`LFRM_EXTRA_XLA_FLAGS="--flag=value"` when testing a specific `jaxlib` build.
TRM/URM attention uses JAX SDPA with cuDNN fused attention by default. Override
with `LFRM_ATTENTION_IMPLEMENTATION=auto|cudnn|xla`; `auto` uses cuDNN for GPU
no-bias attention and keeps rel2d/bias attention on JAX's default lowering.
Training also captures a short JAX profiler window by default (`runtime.profile_start_step=20`,
`runtime.profile_steps=20`) into the run checkpoint directory under `profile/`.
When WandB is enabled, the TensorBoard profile directory is patched and uploaded
as a `jax-profile` artifact. Disable with `--no-profile-enabled` or
`profile_enabled = false` under `[runtime]`.

## Notes

- Datasets are built offline with `train/` and `test/` splits and sampled at runtime.
- Sudoku derives blank-cell supervision from token `1`; Maze writes an explicit
  `given_mask.npy` so known walls, start, and goal cells are treated as givens,
  while open cells are supervised as path/non-path decisions.
- Old Sudoku LFRM/Mini-GLIDER configs are not compatible with the current
  BRC-Sudoku path. The TRM Sudoku baseline config is still available.

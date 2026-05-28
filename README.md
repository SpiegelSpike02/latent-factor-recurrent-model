# Recurrent Grid Reasoning

This is a JAX + `flax.nnx` research codebase for recurrent reasoning on 2D grid
tasks. The current Sudoku MVP is BDR-Sudoku, a belief-first recurrent solver
that studies learned dynamics over an explicit answer belief.

- `bdr_sudoku`: the current Sudoku path. It uses digit belief logits, recurrent
  spatial hidden state, local convolutional mixing, global attention, and either
  velocity or energy-gradient belief updates.
- `trm`: a Tiny Recursive Model baseline.

BDR means Belief Dynamics Reasoner. The design separates explicit answer state
from recurrent computation. The persistent answer coordinate is centered logits
`z`; its softmax `q` is the per-cell probability view used by the loss and by
the recurrent solver. The separate hidden field `h` carries grid computation.
The current energy branch learns a scalar energy over candidate distributions
and updates `z` by a mirror-descent-like step; the velocity branch learns a free
logit-space update field.

BDR-Sudoku does not use clue-dropout pseudo-labels, symbolic traces, DSL rules,
or a hand-written Sudoku checker in the loss. Training uses step-carry
recurrence, supervised CE over target cells, digit permutation augmentation,
energy/path diagnostics, and optional fixed-point, wrong-attractor, and
corrupted-recovery auxiliary losses.

A short architecture overview lives in [docs/architecture.md](docs/architecture.md).

## Source Layout

The implementation lives under the `lfrm` package:

- `lfrm.models`: BDR-Sudoku and TRM models
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

Train BDR-Sudoku:

```bash
uv run lfrm-train --config configs/sudoku_bdr.toml
```

CLI flags override config values:

```bash
uv run lfrm-train --config configs/sudoku_bdr.toml --learning-rate 1e-4 --batch-size 16
```

BDR-Sudoku recurrent supervision is controlled by `model.bdr.step_loss_schedule`,
which selects how per-step CE terms are normalized across commit steps.
TRM/URM official ACT configs do not use step-weighted dense supervision; dense
unroll remains a separate experimental path. BDR-Sudoku additionally reports
given consistency, row/column/box conflict count, target probability, update
size, distribution movement, and energy-gradient diagnostics.

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
- Old Sudoku LFRM/Mini-GLIDER/BDR-prototype configs are not compatible with the current
  BDR-Sudoku path. The TRM Sudoku baseline config is still available.

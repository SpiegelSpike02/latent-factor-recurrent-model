from __future__ import annotations

import os


DEFAULT_JAX_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_GPU_ALLOCATOR": "cuda_malloc_async",
    "NCCL_LL128_BUFFSIZE": "-2",
    "NCCL_LL_BUFFSIZE": "-2",
    "NCCL_PROTO": "SIMPLE,LL,LL128",
}

DEFAULT_UNSET_ENV = (
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
)

DEFAULT_XLA_FLAGS = (
    "--xla_gpu_triton_gemm_any=true",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
)


def apply_jax_defaults() -> None:
    """Apply project-level JAX/XLA defaults before JAX initializes.

    These defaults intentionally overwrite the shell environment so training
    behavior is controlled by the repository, not by stale terminal exports.
    """
    for name, value in DEFAULT_JAX_ENV.items():
        os.environ[name] = value
    for name in DEFAULT_UNSET_ENV:
        os.environ.pop(name, None)
    os.environ["XLA_FLAGS"] = " ".join(DEFAULT_XLA_FLAGS)

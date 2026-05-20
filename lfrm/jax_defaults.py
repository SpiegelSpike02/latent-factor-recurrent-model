from __future__ import annotations

import os


DEFAULT_JAX_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_GPU_ALLOCATOR": "cuda_malloc_async",
}

DEFAULT_UNSET_ENV = (
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "NCCL_LL128_BUFFSIZE",
    "NCCL_LL_BUFFSIZE",
    "NCCL_PROTO",
)

DEFAULT_XLA_FLAGS = (
    "--xla_gpu_triton_gemm_any=true",
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

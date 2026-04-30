from __future__ import annotations

import os


DEFAULT_JAX_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "TF_GPU_ALLOCATOR": "cuda_malloc_async",
}

DEFAULT_XLA_FLAGS = (
    "--xla_gpu_triton_gemm_any=true",
)


def _append_missing_xla_flags(existing: str, defaults: tuple[str, ...]) -> str:
    current_flags = existing.split()
    current_names = {flag.split("=", maxsplit=1)[0] for flag in current_flags}
    missing_flags = [
        flag
        for flag in defaults
        if flag.split("=", maxsplit=1)[0] not in current_names
    ]
    return " ".join((*current_flags, *missing_flags))


def apply_jax_defaults() -> None:
    """Apply project-level JAX/XLA defaults before JAX initializes."""
    for name, value in DEFAULT_JAX_ENV.items():
        os.environ.setdefault(name, value)
    os.environ["XLA_FLAGS"] = _append_missing_xla_flags(
        os.environ.get("XLA_FLAGS", ""),
        DEFAULT_XLA_FLAGS,
    )

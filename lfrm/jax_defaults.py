from __future__ import annotations

import os


DEFAULT_JAX_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_GPU_ALLOCATOR": "cuda_malloc_async",
    "NCCL_LL128_BUFFSIZE": "-2",
    "NCCL_LL_BUFFSIZE": "-2",
    "NCCL_PROTO": "SIMPLE,LL,LL128",
}

RESPECT_EXTERNAL_ENV_FLAG = "LFRM_RESPECT_EXTERNAL_JAX_ENV"
DEFAULT_UNSET_ENV = (
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
)

DEFAULT_XLA_FLAGS = (
    "--xla_gpu_triton_gemm_any=true",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_async_collectives=true",
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
    respect_external_env = os.environ.get(RESPECT_EXTERNAL_ENV_FLAG, "").lower() in {"1", "true", "yes"}
    for name, value in DEFAULT_JAX_ENV.items():
        if respect_external_env:
            os.environ.setdefault(name, value)
        else:
            os.environ[name] = value
    if not respect_external_env:
        for name in DEFAULT_UNSET_ENV:
            os.environ.pop(name, None)
    os.environ["XLA_FLAGS"] = _append_missing_xla_flags(
        os.environ.get("XLA_FLAGS", ""),
        DEFAULT_XLA_FLAGS,
    )

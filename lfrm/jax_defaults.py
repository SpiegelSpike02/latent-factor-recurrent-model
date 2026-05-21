from __future__ import annotations

import os


DEFAULT_UNSET_ENV = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "TF_GPU_ALLOCATOR",
    "NCCL_LL128_BUFFSIZE",
    "NCCL_LL_BUFFSIZE",
    "NCCL_PROTO",
    "XLA_FLAGS",
    "LFRM_EXTRA_XLA_FLAGS",
    "LFRM_ATTENTION_IMPLEMENTATION",
)


def apply_jax_defaults() -> None:
    """Clear project-level JAX/XLA overrides before JAX initializes.

    The default policy is to let JAX/XLA choose attention, allocator, NCCL, and
    GPU lowering strategies automatically. We still clear stale terminal exports
    so a previous experiment does not silently force a non-default path.
    """
    for name in DEFAULT_UNSET_ENV:
        os.environ.pop(name, None)

from __future__ import annotations

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from lfrm.config import ExperimentConfig


def data_parallel_mesh(config: ExperimentConfig) -> Mesh | None:
    requested = config.runtime.data_parallel_devices
    if requested < 0:
        raise ValueError("data_parallel_devices must be non-negative")
    devices = jax.devices()
    if requested == 0:
        requested = len(devices)
    if requested <= 1:
        return None
    if requested > len(devices):
        raise ValueError(
            f"Requested data_parallel_devices={requested}, but only {len(devices)} JAX devices are visible"
        )
    return jax.make_mesh((requested,), ("data",), devices=devices[:requested])


def batch_sharding(mesh: Mesh | None) -> NamedSharding | None:
    if mesh is None:
        return None
    return NamedSharding(mesh, P("data"))


def replicated_sharding(mesh: Mesh | None) -> NamedSharding | None:
    if mesh is None:
        return None
    return NamedSharding(mesh, P())


def place_module_replicated(module, sharding: NamedSharding | None) -> None:
    if sharding is None:
        return
    nnx.update(module, jax.device_put(nnx.state(module), sharding))


def place_tree(tree, sharding: NamedSharding | None):
    if sharding is None:
        return tree
    return jax.device_put(tree, sharding)


def is_batch_sharded_device(device: jax.Device | NamedSharding) -> bool:
    return isinstance(device, NamedSharding) and device.spec == P("data")


def device_put_batch_sharded(batch: dict[str, np.ndarray], sharding: NamedSharding) -> dict[str, jax.Array]:
    num_devices = len(tuple(sharding.mesh.devices.flat))
    if num_devices == 0:
        raise ValueError("Cannot shard a batch over an empty mesh")

    def put_leaf(value: np.ndarray) -> jax.Array:
        if value.shape[0] % num_devices != 0:
            raise ValueError(
                f"Leading batch dimension {value.shape[0]} must be divisible by data devices={num_devices}"
            )
        return jax.device_put(value, sharding)

    return jax.tree.map(put_leaf, batch)

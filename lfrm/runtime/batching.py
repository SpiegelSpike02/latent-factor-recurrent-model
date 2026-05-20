from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import jax
import numpy as np
from jax.sharding import NamedSharding

from lfrm.config import ExperimentConfig
from lfrm.datasets import GridBatchSampler, sample_batch
from lfrm.runtime.sharding import device_put_batch_sharded, is_batch_sharded_device


def sample_device_batch(
    rng: np.random.Generator | GridBatchSampler,
    dataset,
    *,
    config: ExperimentConfig,
    split: str,
    device: jax.Device | NamedSharding,
) -> dict[str, jax.Array]:
    if isinstance(rng, GridBatchSampler):
        batch = rng.sample(batch_size=config.train.batch_size, seq_len=config.model.seq_len, split=split)
    else:
        batch = sample_batch(
            rng,
            dataset,
            batch_size=config.train.batch_size,
            seq_len=config.model.seq_len,
            split=split,
        )
    batch["inputs"] = np.asarray(batch["inputs"], dtype=np.int32)
    batch["labels"] = np.asarray(batch["labels"], dtype=np.int32)
    batch["puzzle_identifiers"] = np.asarray(batch["puzzle_identifiers"], dtype=np.int32)
    if is_batch_sharded_device(device):
        return device_put_batch_sharded(batch, device)
    return jax.device_put(batch, device=device)


class BatchPrefetcher:
    def __init__(self, sample_fn, *, depth: int = 4, workers: int = 2, refill: bool = True) -> None:
        if depth < 1:
            raise ValueError("Prefetch depth must be at least 1")
        if workers < 1:
            raise ValueError("Prefetch workers must be at least 1")
        self.sample_fn = sample_fn
        self.refill = refill
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.futures: list[Future] = []
        for _ in range(depth):
            self.futures.append(self.executor.submit(self.sample_fn))

    def next(self):
        future = self.futures.pop(0)
        batch = future.result()
        if self.refill:
            self.futures.append(self.executor.submit(self.sample_fn))
        return batch

    def close(self) -> None:
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


def eval_device_batch(
    dataset,
    *,
    config: ExperimentConfig,
    start: int,
    stop: int,
    device: jax.Device | NamedSharding,
    target_batch_size: int,
) -> dict[str, jax.Array]:
    if dataset.spec.seq_len != config.model.seq_len:
        raise ValueError(f"Requested seq_len={config.model.seq_len}, but dataset seq_len={dataset.spec.seq_len}")
    actual_batch_size = stop - start
    batch = {
        "inputs": np.asarray(dataset.eval_inputs[start:stop], dtype=np.int32),
        "labels": np.asarray(dataset.eval_labels[start:stop], dtype=np.int32),
        "given_mask": np.asarray(dataset.eval_given_mask[start:stop], dtype=bool),
        "puzzle_identifiers": np.asarray(dataset.eval_puzzle_identifiers[start:stop], dtype=np.int32),
    }
    example_mask = np.ones((actual_batch_size,), dtype=np.float32)
    if actual_batch_size < target_batch_size:
        pad_width = target_batch_size - actual_batch_size
        batch = {
            key: np.pad(value, ((0, pad_width), *[(0, 0)] * (value.ndim - 1)), mode="edge")
            for key, value in batch.items()
        }
        example_mask = np.pad(example_mask, (0, pad_width), constant_values=0.0)
    batch["example_mask"] = example_mask
    if is_batch_sharded_device(device):
        return device_put_batch_sharded(batch, device)
    return jax.device_put(batch, device=device)


def small_metric_items(
    metrics: dict[str, Any],
    *,
    max_elements: int = 4096,
    verbose: bool = True,
) -> dict[str, Any]:
    small: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in metrics.items():
        shape = getattr(value, "shape", ())
        size = int(np.prod(shape)) if shape else 1
        if size <= max_elements:
            small[key] = value
        else:
            skipped.append(f"{key}{tuple(shape)}")
    if skipped and verbose:
        print("[metrics] skipped large metric leaves:", ", ".join(skipped), flush=True)
    return small

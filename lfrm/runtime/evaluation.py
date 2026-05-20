from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import jax
import numpy as np

from lfrm.config import ExperimentConfig
from lfrm.runtime.batching import eval_device_batch, small_metric_items
from lfrm.runtime.sharding import batch_sharding, data_parallel_mesh


def evaluate(eval_step_fn, model, dataset, *, config: ExperimentConfig) -> dict[str, Any]:
    mesh = data_parallel_mesh(config)
    sharded_device = batch_sharding(mesh)
    primary_device = jax.devices()[0]
    device = sharded_device or primary_device
    reduced: dict[str, Any] | None = None
    total = dataset.eval_inputs.shape[0]
    if total == 0:
        raise ValueError("Eval split is empty")
    batch_size = config.eval.batch_size or config.train.microbatch_size
    if batch_size <= 0:
        raise ValueError("eval batch_size must be at least 1 when set")
    total_weight = 0.0
    num_batches = (total + batch_size - 1) // batch_size

    print(
        f"[eval] running {num_batches} batches "
        f"x batch_size={batch_size} "
        f"({total} examples)",
        flush=True,
    )
    eval_ranges = [
        (start, min(start + batch_size, total))
        for start in range(0, total, batch_size)
    ]

    def make_eval_batch(batch_index: int):
        start, stop = eval_ranges[batch_index]
        return stop - start, eval_device_batch(
            dataset,
            config=config,
            start=start,
            stop=stop,
            device=device,
            target_batch_size=batch_size,
        )

    eval_executor = ThreadPoolExecutor(max_workers=config.runtime.prefetch_workers)
    eval_futures: list[Future] = []
    next_eval_index = 0

    def submit_eval_batch() -> None:
        nonlocal next_eval_index
        if next_eval_index < num_batches:
            eval_futures.append(eval_executor.submit(make_eval_batch, next_eval_index))
            next_eval_index += 1

    for _ in range(min(config.runtime.prefetch_depth, num_batches)):
        submit_eval_batch()
    try:
        for batch_index in range(1, num_batches + 1):
            future = eval_futures.pop(0)
            weight_int, batch = future.result()
            submit_eval_batch()
            metrics = jax.device_get(small_metric_items(eval_step_fn(model, batch)))
            weight = float(weight_int)
            if reduced is None:
                reduced = {
                    key: np.zeros(np.asarray(value).shape, dtype=np.float64)
                    for key, value in metrics.items()
                }
            for key, value in metrics.items():
                value_array = np.asarray(value, dtype=np.float64)
                if key == "count" or key.endswith("_count"):
                    reduced[key] += value_array
                else:
                    reduced[key] += value_array * weight
            total_weight += weight
            if batch_index == 1 or batch_index == num_batches or batch_index % 10 == 0:
                print(f"[eval] batch {batch_index}/{num_batches}", flush=True)
    finally:
        for future in eval_futures:
            future.cancel()
        eval_executor.shutdown(wait=False, cancel_futures=True)
    if reduced is None:
        raise ValueError("No eval batches were produced")
    scale = 1.0 / total_weight
    averaged = {
        key: value if key == "count" or key.endswith("_count") else value * scale
        for key, value in reduced.items()
    }
    return {
        key: float(value) if np.ndim(value) == 0 else value.astype(float).tolist()
        for key, value in averaged.items()
    }

# ruff: noqa: D103, T201

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import dask
import dask.array as da
import numpy as np
import psutil

from spatialdata._core.operations.rasterize import rasterize
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def largest_chunk_bytes(array: da.Array) -> int:
    return int(np.prod([max(axis) for axis in array.chunks]) * array.dtype.itemsize)


def main() -> None:
    args = parse_args()
    tile_size = args.tile_size or None
    source = da.ones(
        (3, args.size, args.size),
        chunks=(3, 256, 256),
        dtype=np.float32,
    )
    image = Image2DModel.parse(
        source,
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )
    result = rasterize(
        image,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[args.size, args.size],
        target_coordinate_system="global",
        target_width=args.size,
        tile_size=tile_size,
    )

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.002):
            peak_rss = max(peak_rss, process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    durations = []
    checksum = 0.0
    try:
        for _ in range(args.repeats):
            with tempfile.TemporaryDirectory() as directory:
                target = np.memmap(
                    Path(directory) / "output.dat",
                    mode="w+",
                    dtype=result.dtype,
                    shape=result.shape,
                )
                start = time.perf_counter()
                da.store(
                    result.data,
                    target,
                    lock=False,
                    compute=True,
                    scheduler="single-threaded",
                )
                target.flush()
                durations.append(time.perf_counter() - start)
                checksum = float(target[0, 0, 0] + target[-1, -1, -1])
                del target
                gc.collect()
    finally:
        stop.set()
        monitor.join()

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "dask": dask.__version__,
                "numpy": np.__version__,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "input_chunks": [list(axis) for axis in source.chunks],
                "tile_size": tile_size,
                "output_chunks": [list(axis) for axis in result.data.chunks],
                "largest_output_chunk_mib": largest_chunk_bytes(result.data) / 2**20,
                "graph_tasks": len(result.data.__dask_graph__()),
                "runs_seconds": durations,
                "median_seconds": statistics.median(durations),
                "baseline_rss_mib": baseline_rss / 2**20,
                "peak_rss_mib": peak_rss / 2**20,
                "peak_rss_delta_mib": (peak_rss - baseline_rss) / 2**20,
                "checksum": checksum,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

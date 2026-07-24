# ruff: noqa: D103, T201

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import dask
import datashader as ds
import geopandas
import numpy as np
import shapely
from geopandas import GeoDataFrame
from shapely import box

from spatialdata import rasterize
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def run_once(data: GeoDataFrame, *, canvas_size: int, tile_size: int) -> dict[str, object]:
    original = ds.Canvas.polygons
    rows = []

    def counted(canvas: ds.Canvas, source: GeoDataFrame, *args: object, **kwargs: object) -> object:
        rows.append(len(source))
        return original(canvas, source, *args, **kwargs)

    ds.Canvas.polygons = counted
    try:
        start = time.perf_counter()
        result = rasterize(
            data,
            axes=("x", "y"),
            min_coordinate=[0, 0],
            max_coordinate=[10, 10],
            target_coordinate_system="global",
            target_width=canvas_size,
            tile_size=tile_size,
            return_regions_as_labels=True,
        )
        computed = result.data.compute(scheduler="single-threaded")
        duration = time.perf_counter() - start
    finally:
        ds.Canvas.polygons = original

    return {
        "seconds": duration,
        "polygon_calls": len(rows),
        "total_rows_sent": sum(rows),
        "max_rows_sent": max(rows),
        "checksum": int(np.asarray(computed).sum()),
    }


def main() -> None:
    args = parse_args()
    step = 10.0 / args.side
    shapes = [
        box(x * step, y * step, x * step + step * 0.4, y * step + step * 0.4)
        for y in range(args.side)
        for x in range(args.side)
    ]
    data = ShapesModel.parse(
        GeoDataFrame(geometry=shapes),
        transformations={"global": Identity()},
    )

    for _ in range(args.warmup_runs):
        run_once(data, canvas_size=args.canvas_size, tile_size=args.tile_size)
    runs = [run_once(data, canvas_size=args.canvas_size, tile_size=args.tile_size) for _ in range(args.repeats)]
    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "dask": dask.__version__,
                    "datashader": ds.__version__,
                    "geopandas": geopandas.__version__,
                    "numpy": np.__version__,
                    "shapely": shapely.__version__,
                    "dask_scheduler": "single-threaded",
                },
                "workload": {
                    "shapes": len(shapes),
                    "canvas_size": args.canvas_size,
                    "tile_size": args.tile_size,
                    "tiles": (args.canvas_size // args.tile_size) ** 2,
                    "warmup_runs": args.warmup_runs,
                    "measured_runs": args.repeats,
                },
                "median_seconds": statistics.median(run["seconds"] for run in runs),
                "runs": runs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Cell-only volumes: every cell minus all interscellar volumes, written slab by slab.

A voxel keeps its cell label unless it lies in any interscellar volume, or, when a nuclei
segmentation is given, in any nucleus. Only the union of the volumes matters here, so the
result is the same for both pipelines:

- absolute stores: the label array '0'.
- adaptive stores: the 'interscellar_meshes' label grid. Overlapping pairs share voxels
  there, but the union is still exact. When the store was written with --preview none
  there is no grid, and the union is rebuilt from the lossless 'pair_volumes' archive.

A pair's intracellular territories lie inside its own two cells and its corridor outside
all cells, so the volume voxels removed from a cell are exactly those of the pairs that
cell belongs to. The volume is processed in Z-slabs, one output chunk per slab, so slabs
can be written by parallel workers without collisions and memory stays at a few slabs
per worker regardless of dataset size.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

from ..utils.exclude_nuclei import _create_array, _find_label_array, _zarr_gzip_dataset_kwargs

_ZARR_MAJOR = int(str(zarr.__version__).split(".")[0])

CELL_KEY_PREFERENCE = ["labels", "0"]
INTERSCELLAR_KEY_PREFERENCE = ["interscellar_meshes", "0", "labels"]
OUTPUT_KEY = "0"

# Adaptive pair archive layout (see PairVolumeArchive in compute_interscellar_volumes_3d_adaptive).
_ARCHIVE_GROUP = "pair_volumes"
_FOOTPRINT_KEYS = ("territory_a", "corridor", "territory_b")
# Archive reads closer than this many indices apart are merged into one, so pairs whose
# data share a chunk do not decompress it repeatedly.
_ARCHIVE_READ_GAP = 262144

_WORKER: Dict[str, Any] = {}


def _check_readable(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    if os.path.exists(os.path.join(path, "zarr.json")) and _ZARR_MAJOR < 3:
        raise RuntimeError(
            f"The {what} at {path} is a Zarr v3 store, but this environment has "
            f"zarr {zarr.__version__}, which reads only v2. Run from an environment with "
            f"zarr>=3 (which requires Python >=3.11); the store was most likely written by one."
        )


def _spatial_shape(node: Any) -> Tuple[int, int, int]:
    return tuple(int(v) for v in node.shape[-3:])


def _read_slab(node: Any, z0: int, z1: int) -> np.ndarray:
    if node.ndim == 5:
        return np.asarray(node[0, 0, z0:z1])
    return np.asarray(node[z0:z1])


def _open_node(path: str, key: str) -> Any:
    node = zarr.open(path, mode="r")
    for part in [p for p in key.split("/") if p]:
        node = node[part]
    return node


def _label_counts(slab: np.ndarray) -> Dict[int, int]:
    values = slab[slab > 0]
    if values.size == 0:
        return {}
    labels, counts = np.unique(values, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def _default_output_path(interscellar_zarr: str, nuclei_excluded: bool = False) -> str:
    directory = os.path.dirname(os.path.abspath(interscellar_zarr))
    base = os.path.basename(os.path.normpath(interscellar_zarr))
    if base.endswith(".zarr"):
        base = base[:-5]
    marker = "_interscellar_volumes"
    if marker in base:
        head, _, tail = base.rpartition(marker)
        base = f"{head}_cell_only_volumes{tail}"
    else:
        base = f"{base}_cell_only_volumes"
    if nuclei_excluded:
        base = f"{base}_nuclei_excluded"
    return os.path.join(directory, f"{base}.zarr")


def _resolve_interscellar(
    path: str, key: Optional[str]
) -> Tuple[Any, str, str, Tuple[int, int, int]]:
    """Returns (root, source, key, grid shape); source is 'grid' or 'archive'."""
    root = zarr.open(path, mode="r")
    has_archive = _ARCHIVE_GROUP in root and "pair_id" in root[_ARCHIVE_GROUP]
    has_grid = any(k in root for k in INTERSCELLAR_KEY_PREFERENCE)
    if key == _ARCHIVE_GROUP or (key is None and has_archive and not has_grid):
        if not has_archive:
            raise ValueError(f"No '{_ARCHIVE_GROUP}' pair archive in {path}")
        shape = root.attrs.get("volume_shape_zyx")
        if shape is None:
            raise ValueError(
                f"{path} holds only the pair archive (written with --preview none) and no "
                f"'volume_shape_zyx' attribute, so the grid its pairs are indexed in is "
                f"unknown. Re-run the adaptive pipeline with --preview both."
            )
        return root, "archive", _ARCHIVE_GROUP, tuple(int(v) for v in shape)
    root, node, key_used = _find_label_array(path, key, INTERSCELLAR_KEY_PREFERENCE)
    return root, "grid", key_used, _spatial_shape(node)


# Workers. Each opens the stores once from their paths and then handles one Z-slab per
# task, writing it straight into the output chunk that covers it.

def _init_worker(config: Dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER["cells"] = _open_node(config["cell_path"], config["cell_key"])
    _WORKER["nuclei"] = (
        _open_node(config["nuclei_path"], config["nuclei_key"]) if config["nuclei_path"] else None
    )
    _WORKER["out"] = zarr.open(config["output_path"], mode="r+")[OUTPUT_KEY]
    _WORKER["dtype"] = np.dtype(config["dtype"])
    _WORKER["source"] = config["source"]
    if config["source"] == "grid":
        _WORKER["volumes"] = _open_node(config["interscellar_path"], config["interscellar_key"])
        return
    group = zarr.open(config["interscellar_path"], mode="r")[_ARCHIVE_GROUP]
    stated = group.attrs.get("committed_rows")
    n_rows = int(group["pair_id"].shape[0]) if stated is None else int(stated)
    _WORKER["group"] = group
    _WORKER["origins"] = np.asarray(group["origin_zyx"][:n_rows]).astype(np.int64)
    _WORKER["shapes"] = np.asarray(group["shape_zyx"][:n_rows]).astype(np.int64)
    _WORKER["indptr"] = {
        name: np.asarray(group[f"{name}_indptr"][: n_rows + 1]).astype(np.int64)
        for name in _FOOTPRINT_KEYS
    }


def _read_merged(array: Any, ranges: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Reads each [lo, hi) of a 1D array, merging nearby ranges into single reads."""
    order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
    out: List[Optional[np.ndarray]] = [None] * len(ranges)
    i = 0
    while i < len(order):
        lo, hi = ranges[order[i]]
        j = i + 1
        while j < len(order) and ranges[order[j]][0] - hi <= _ARCHIVE_READ_GAP:
            hi = max(hi, ranges[order[j]][1])
            j += 1
        block = np.asarray(array[lo:hi])
        for k in order[i:j]:
            a, b = ranges[k]
            out[k] = block[a - lo:b - lo]
        i = j
    return out


def _archive_union_slab(z0: int, z1: int, plane: Tuple[int, int]) -> np.ndarray:
    union = np.zeros((z1 - z0,) + tuple(plane), dtype=bool)
    origins, shapes = _WORKER["origins"], _WORKER["shapes"]
    rows = np.flatnonzero((origins[:, 0] < z1) & (origins[:, 0] + shapes[:, 0] > z0))
    if rows.size == 0:
        return union
    for name in _FOOTPRINT_KEYS:
        indptr = _WORKER["indptr"][name]
        ranges = [(int(indptr[r]), int(indptr[r + 1])) for r in rows]
        wanted = [k for k, (lo, hi) in enumerate(ranges) if hi > lo]
        if not wanted:
            continue
        chunks = _read_merged(_WORKER["group"][name], [ranges[k] for k in wanted])
        for k, deltas in zip(wanted, chunks):
            row = int(rows[k])
            flat = np.cumsum(deltas, dtype=np.uint64)
            z, y, x = np.unravel_index(flat, tuple(int(v) for v in shapes[row]))
            z = z + (origins[row, 0] - z0)
            keep = (z >= 0) & (z < z1 - z0)
            union[z[keep], y[keep] + origins[row, 1], x[keep] + origins[row, 2]] = True
    return union


def _process_slab(z0: int, z1: int) -> Dict[str, Any]:
    cells = _read_slab(_WORKER["cells"], z0, z1)
    in_cell = cells > 0
    if _WORKER["source"] == "grid":
        in_volume = _read_slab(_WORKER["volumes"], z0, z1) > 0
    else:
        in_volume = _archive_union_slab(z0, z1, cells.shape[1:])
    drop = in_volume
    removed_interscellar = int(np.count_nonzero(in_cell & in_volume))
    removed_nuclei_only = 0
    if _WORKER["nuclei"] is not None:
        in_nucleus = _read_slab(_WORKER["nuclei"], z0, z1) > 0
        removed_nuclei_only = int(np.count_nonzero(in_cell & in_nucleus & ~in_volume))
        drop = in_volume | in_nucleus
    result = np.where(drop, 0, cells).astype(_WORKER["dtype"], copy=False)
    out = _WORKER["out"]
    if out.ndim == 5:
        out[0, 0, z0:z1] = result
    else:
        out[z0:z1] = result
    return {
        "z1": z1,
        "before": _label_counts(cells),
        "after": _label_counts(result),
        "removed_interscellar": removed_interscellar,
        "removed_nuclei_only": removed_nuclei_only,
    }


def _process_slab_task(bounds: Tuple[int, int]) -> Dict[str, Any]:
    return _process_slab(*bounds)


def compute_cell_only_volumes(
    cell_segmentation_zarr: str,
    interscellar_zarr: str,
    output_zarr_path: Optional[str] = None,
    cell_key: Optional[str] = None,
    interscellar_key: Optional[str] = None,
    slab: int = 16,
    nuclei_segmentation_zarr: Optional[str] = None,
    nuclei_key: Optional[str] = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Writes the cell-only volumes zarr and returns a summary.

    The summary holds 'output_zarr', 'source' ('grid' or 'archive'), 'voxel_size_um'
    (None when the interscellar store does not record it) and per-cell voxel counts
    before ('cell_voxels_before') and after ('cell_voxels_after') the subtraction.
    """
    _check_readable(cell_segmentation_zarr, "cell segmentation zarr")
    _check_readable(interscellar_zarr, "interscellar zarr")
    if nuclei_segmentation_zarr:
        _check_readable(nuclei_segmentation_zarr, "nuclei segmentation zarr")
    output_zarr_path = output_zarr_path or _default_output_path(
        interscellar_zarr, nuclei_excluded=bool(nuclei_segmentation_zarr)
    )

    _, cell_node, cell_key_used = _find_label_array(
        cell_segmentation_zarr, cell_key, CELL_KEY_PREFERENCE
    )
    isc_root, source, isc_key_used, isc_shape = _resolve_interscellar(
        interscellar_zarr, interscellar_key
    )
    shape = _spatial_shape(cell_node)
    if isc_shape != shape:
        raise ValueError(
            f"Shape mismatch: cells {shape} (key '{cell_key_used}') vs interscellar "
            f"{isc_shape} (key '{isc_key_used}'). Both must be on the same grid."
        )
    nuc_key_used = None
    if nuclei_segmentation_zarr:
        _, nuc_node, nuc_key_used = _find_label_array(nuclei_segmentation_zarr, nuclei_key)
        if _spatial_shape(nuc_node) != shape:
            raise ValueError(
                f"Shape mismatch: cells {shape} vs nuclei {_spatial_shape(nuc_node)} "
                f"(key '{nuc_key_used}'). Both must be on the same grid."
            )
    dtype = cell_node.dtype if np.issubdtype(cell_node.dtype, np.integer) else np.dtype(np.uint32)
    slab = max(1, int(slab))
    n_jobs = max(1, int(n_jobs))
    with_nuclei = nuc_key_used is not None

    print("Cell-only volumes: cells minus the union of all interscellar volumes"
          + (" and all nuclei" if with_nuclei else ""))
    print(f"  Cells:        {cell_segmentation_zarr} (key '{cell_key_used}', {cell_node.dtype})")
    if source == "archive":
        print(f"  Interscellar: {interscellar_zarr} (union rebuilt from the "
              f"'{_ARCHIVE_GROUP}' pair archive)")
    else:
        print(f"  Interscellar: {interscellar_zarr} (key '{isc_key_used}')")
    if with_nuclei:
        print(f"  Nuclei:       {nuclei_segmentation_zarr} (key '{nuc_key_used}')")
    print(f"  Output:       {output_zarr_path}")
    print(f"  Grid (Z, Y, X): {shape}; {slab} slices per slab")

    write_5d = cell_node.ndim == 5
    out_shape = ((1, 1) + shape) if write_5d else shape
    # One output chunk per slab along Z, so parallel slab writes never share a chunk.
    spatial_chunks = (slab, min(256, shape[1]), min(256, shape[2]))
    chunks = ((1, 1) + spatial_chunks) if write_5d else spatial_chunks
    out_root = zarr.open(output_zarr_path, mode="w")
    _create_array(
        out_root, OUTPUT_KEY, out_shape, dtype, chunks, 0,
        **_zarr_gzip_dataset_kwargs(level=6, copy_compressors_from=cell_node),
    )

    config = {
        "cell_path": cell_segmentation_zarr,
        "cell_key": cell_key_used,
        "interscellar_path": interscellar_zarr,
        "interscellar_key": isc_key_used,
        "source": source,
        "nuclei_path": nuclei_segmentation_zarr if with_nuclei else None,
        "nuclei_key": nuc_key_used,
        "output_path": output_zarr_path,
        "dtype": dtype.str,
    }
    bounds = [(z0, min(z0 + slab, shape[0])) for z0 in range(0, shape[0], slab)]
    plane = shape[1] * shape[2] * slab
    per_worker = plane * (2 * dtype.itemsize + 4 + (1 if with_nuclei else 0) + 3)
    print(f"  Workers: {n_jobs}; about {per_worker * n_jobs / 1e9:.2f} GB resident "
          f"({per_worker / 1e9:.2f} GB each). Lower --n-jobs or --slab if that is too much.")

    before: Dict[int, int] = {}
    after: Dict[int, int] = {}
    removed_interscellar = 0
    removed_nuclei_only = 0
    started = time.time()

    def _collect(part: Dict[str, Any], done: int) -> None:
        nonlocal removed_interscellar, removed_nuclei_only
        for totals, counts in ((before, part["before"]), (after, part["after"])):
            for label, count in counts.items():
                totals[label] = totals.get(label, 0) + count
        removed_interscellar += part["removed_interscellar"]
        removed_nuclei_only += part["removed_nuclei_only"]
        print(f"  slab {done}/{len(bounds)}  ({time.time() - started:.0f}s)", flush=True)

    if n_jobs == 1:
        _init_worker(config)
        try:
            for done, (z0, z1) in enumerate(bounds, start=1):
                _collect(_process_slab(z0, z1), done)
        finally:
            _WORKER.clear()
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker, initargs=(config,)
        ) as pool:
            for done, part in enumerate(pool.map(_process_slab_task, bounds), start=1):
                _collect(part, done)

    voxel_size = isc_root.attrs.get("voxel_size_um")
    out_root.attrs.update({
        "description": (
            "Cell-only volumes (original cell segmentation minus interscellar volumes"
            + (" and nuclei)" if with_nuclei else ")")
        ),
        "subtraction": (
            "cells minus union of all interscellar volumes (interscellar > 0)"
            + (" and all nuclei (nuclei > 0)" if with_nuclei else "")
        ),
        "cell_segmentation_zarr": os.path.abspath(cell_segmentation_zarr),
        "cell_segmentation_key": cell_key_used,
        "source_interscellar_zarr": os.path.abspath(interscellar_zarr),
        "source_interscellar_key": isc_key_used,
        "original_cells": len(before),
        "remaining_cells": len(after),
        "original_cell_voxels": int(sum(before.values())),
        "remaining_cell_voxels": int(sum(after.values())),
        "cell_voxels_removed_by_interscellar": removed_interscellar,
    })
    if with_nuclei:
        out_root.attrs["nuclei_segmentation_zarr"] = os.path.abspath(nuclei_segmentation_zarr)
        out_root.attrs["nuclei_segmentation_key"] = nuc_key_used
        out_root.attrs["cell_voxels_removed_by_nuclei_only"] = removed_nuclei_only
    if voxel_size is not None:
        out_root.attrs["voxel_size_um"] = list(voxel_size)

    removed = sum(before.values()) - sum(after.values())
    print(f"Wrote {output_zarr_path}")
    print(f"  Cells: {len(before):,} -> {len(after):,} "
          f"({len(before) - len(after):,} lie entirely inside interscellar volumes"
          + (" or nuclei)" if with_nuclei else ")"))
    if before:
        total = sum(before.values())
        print(f"  Cell voxels removed: {removed:,} of {total:,} ({removed / total * 100:.1f}%)")
        print(f"    in interscellar volumes: {removed_interscellar:,} "
              f"({removed_interscellar / total * 100:.1f}%)")
        if with_nuclei:
            print(f"    in nuclei, outside volumes: {removed_nuclei_only:,} "
                  f"({removed_nuclei_only / total * 100:.1f}%)")
    return {
        "output_zarr": output_zarr_path,
        "source": source,
        "voxel_size_um": None if voxel_size is None else tuple(voxel_size),
        "cell_voxels_before": before,
        "cell_voxels_after": after,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build cell-only volumes: the cell segmentation with every voxel that lies in "
            "any interscellar volume set to 0, and optionally every voxel in a nucleus. "
            "Works with absolute stores ('0') and adaptive stores ('interscellar_meshes', "
            "or the 'pair_volumes' archive when written with --preview none). Processed in "
            "Z-slabs, so memory stays small. Output labels are cell IDs, readable by "
            "feature-extract-3d with --object-id-column cell_id."
        )
    )
    parser.add_argument("--cell-segmentation-zarr", required=True,
                        help="Cell segmentation zarr (cell IDs as labels).")
    parser.add_argument("--interscellar-zarr", required=True,
                        help="Interscellar volumes zarr (adaptive or absolute).")
    parser.add_argument(
        "--nuclei-segmentation-zarr", default=None,
        help="Optional nuclei segmentation zarr; every voxel inside a nucleus is also removed.",
    )
    parser.add_argument(
        "--output-zarr", default=None,
        help="Output path. Defaults to the interscellar zarr's name with "
             "'_interscellar_volumes' replaced by '_cell_only_volumes' (plus "
             "'_nuclei_excluded' when nuclei are given), in the same folder.",
    )
    parser.add_argument("--cell-key", default=None,
                        help="Dataset key in the cell zarr (default: auto-detect).")
    parser.add_argument(
        "--interscellar-key", default=None,
        help="Dataset key in the interscellar zarr (default: auto-detect). Pass "
             "'pair_volumes' to rebuild the union from an adaptive store's pair archive "
             "even when its label grid is present.",
    )
    parser.add_argument("--nuclei-key", default=None,
                        help="Dataset key in the nuclei zarr (default: auto-detect).")
    parser.add_argument("--slab", type=int, default=16,
                        help="Z slices processed at a time (default 16).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Worker processes, each handling one Z-slab at a time (default 1).")
    args = parser.parse_args(argv)

    try:
        compute_cell_only_volumes(
            cell_segmentation_zarr=args.cell_segmentation_zarr,
            interscellar_zarr=args.interscellar_zarr,
            output_zarr_path=args.output_zarr,
            cell_key=args.cell_key,
            interscellar_key=args.interscellar_key,
            slab=args.slab,
            nuclei_segmentation_zarr=args.nuclei_segmentation_zarr,
            nuclei_key=args.nuclei_key,
            n_jobs=args.n_jobs,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import zarr

DEFAULT_VOXEL_SIZE_UM = (0.56, 0.28, 0.28)

# Written by compute_interscellar_volumes_3d_adaptive (3D) and ..._absolute (5D OME).
# Both label voxels by pair_id, so either can be scored; only the layout differs.
_PAIR_LABEL_KEYS = ("interscellar_meshes", "0", "labels")

_UNIT_TO_UM = {
    "": 1.0, "micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0,
    "nanometer": 1e-3, "nm": 1e-3,
    "millimeter": 1e3, "mm": 1e3,
    "centimeter": 1e4, "cm": 1e4,
    "meter": 1e6, "m": 1e6,
}

PAIR_COLUMNS = [
    'pair_id', 'cell_a_id', 'cell_b_id',
    'n_voxels', 'interscellar_volume_um3', 'volumes_csv_volume_um3',
    'contested_voxels', 'contested_fraction',
    'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
    'min_distance_um', 'max_distance_um', 'max_attainable_weight',
]
SCORE_COLUMNS = [
    'biomarker', 'n_spots', 'score_sum', 'score_mean', 'score_per_um3',
    'median_dist_um', 'decay_power',
]


# Zarr access and OME-NGFF grid metadata

_ZARR_MAJOR = int(str(zarr.__version__).split(".")[0])


def _on_disk_zarr_format(path: str) -> Optional[int]:
    """Format version of a store on disk, from its metadata files alone."""
    if os.path.exists(os.path.join(path, "zarr.json")):
        return 3
    if os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(
        os.path.join(path, ".zarray")
    ):
        return 2
    return None


def _open_store(path: str, what: str) -> Any:
    """Open a store read-only, naming the v3-store / v2-library mismatch outright.

    zarr-python 2 cannot read a v3 store and reports it as an empty path rather than
    as an unsupported format, which sends people hunting for a missing file. The
    pipeline writes whichever format the installed zarr produces, so a volume built
    under zarr 3 is unreadable from a zarr 2 environment.
    """
    try:
        return zarr.open(path, mode="r")
    except Exception as exc:
        if _on_disk_zarr_format(path) == 3 and _ZARR_MAJOR < 3:
            raise RuntimeError(
                f"The {what} at {path} is a Zarr v3 store, but this environment has "
                f"zarr {zarr.__version__}, which reads only v2. Run from an "
                f"environment with zarr>=3 (which requires Python >=3.11); the store "
                f"was most likely written by one."
            ) from exc
        raise


def _node_by_key(root: Any, key: str) -> Any:
    if key == "<root>":
        return root
    node = root
    for part in key.split("/"):
        node = node[part]
    return node


def _spatial_shape_zyx(arr: Any) -> Tuple[int, int, int]:
    shape = tuple(int(v) for v in arr.shape)
    if len(shape) == 5:
        return shape[2], shape[3], shape[4]
    if len(shape) == 4:
        return shape[1], shape[2], shape[3]
    if len(shape) == 3:
        return shape
    raise ValueError(f"Expected a 3D-5D array, got shape={shape}")


def _chunk_zyx(arr: Any, default: int = 64) -> Tuple[int, int, int]:
    shape = _spatial_shape_zyx(arr)
    chunks = getattr(arr, "chunks", None)
    if not chunks:
        return tuple(min(default, s) for s in shape)
    return tuple(max(1, min(int(c), s)) for c, s in zip(tuple(chunks)[-3:], shape))


def _read_block(arr: Any, bounds: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bounds
    ndim = len(arr.shape)
    if ndim == 5:
        return np.asarray(arr[0, 0, z0:z1, y0:y1, x0:x1])
    if ndim == 4:
        return np.asarray(arr[0, z0:z1, y0:y1, x0:x1])
    if ndim == 3:
        return np.asarray(arr[z0:z1, y0:y1, x0:x1])
    raise ValueError(f"Expected a 3D-5D array, got ndim={ndim}")


def _parse_ngff_grid(root: Any):
    attrs = getattr(root, "attrs", None)
    multiscales = attrs.get("multiscales") if attrs is not None else None
    if not multiscales:
        return None, None

    entry = multiscales[0]
    axes = entry.get("axes")
    datasets = entry.get("datasets") or []
    if not axes or not datasets:
        return None, None

    names, units = [], []
    for axis in axes:
        if isinstance(axis, Mapping):
            names.append(str(axis.get("name", "")).lower())
            units.append(str(axis.get("unit", "")).lower())
        else:
            names.append(str(axis).lower())
            units.append("")

    spatial = [i for i, n in enumerate(names) if n in ("z", "y", "x")]
    if [names[i] for i in spatial] != ["z", "y", "x"]:
        raise ValueError(
            f"OME-NGFF spatial axes are {[names[i] for i in spatial]}, but this script "
            f"reads z, y, x order. Re-save the store with axes in z, y, x."
        )

    dataset = datasets[0]

    def _scale_of(container: Any):
        for transform in container.get("coordinateTransformations", []) or []:
            if transform.get("type") == "scale":
                return [float(v) for v in transform.get("scale", [])]
        return None

    scale = _scale_of(dataset)
    if scale is None or len(scale) != len(names):
        return dataset.get("path"), None

    # A scale on the multiscales entry composes with the dataset's own.
    outer = _scale_of(entry)
    if outer is not None and len(outer) == len(scale):
        scale = [a * b for a, b in zip(scale, outer)]

    voxel = tuple(scale[i] * _UNIT_TO_UM.get(units[i], 1.0) for i in spatial)
    return dataset.get("path"), voxel


def _resolve_array(root: Any, explicit: Optional[str], preferred: Sequence[str], what: str):
    if explicit is not None:
        node = _node_by_key(root, explicit)
        if not hasattr(node, "shape"):
            raise ValueError(f"'{explicit}' is not an array in the {what} zarr")
        return explicit, node

    if hasattr(root, "shape"):
        return "<root>", root

    ngff_key, _ = _parse_ngff_grid(root)
    if ngff_key is not None and ngff_key in root and hasattr(root[ngff_key], "shape"):
        return ngff_key, root[ngff_key]

    for key in preferred:
        if key not in root:
            continue
        node = root[key]
        if hasattr(node, "shape"):
            return key, node
        if "0" in node and hasattr(node["0"], "shape"):   # OME pyramid: root[key][0]
            return f"{key}/0", node["0"]

    candidates = [
        key for key in (sorted(root.keys()) if hasattr(root, "keys") else [])
        if hasattr(root[key], "shape") and len(root[key].shape) >= 3
    ]
    if len(candidates) == 1:
        return candidates[0], root[candidates[0]]
    if not candidates:
        raise ValueError(f"No 3D+ array found in the {what} zarr.")
    raise ValueError(f"The {what} zarr holds several arrays {candidates}; name one explicitly.")


def _validate_grid(label_shape, label_voxel, other_shape, other_voxel, name: str) -> None:
    if other_shape != label_shape:
        raise ValueError(
            f"Spot zarr '{name}' has shape {other_shape} but the interscellar zarr is "
            f"{label_shape}. Both must be on the same voxel grid."
        )
    if label_voxel is None or other_voxel is None:
        return
    if not np.allclose(label_voxel, other_voxel, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"Spot zarr '{name}' declares voxel size {other_voxel} um but the "
            f"interscellar zarr declares {label_voxel} um. Both must match."
        )


# 3D block iteration

def _choose_block(chunk_zyx, shape_zyx, budget_bytes: int, bytes_per_voxel: int):
    block = [min(c, s) for c, s in zip(chunk_zyx, shape_zyx)]
    budget_voxels = max(1, budget_bytes // max(1, bytes_per_voxel))

    for axis in (2, 1, 0):
        while block[axis] < shape_zyx[axis]:
            trial = list(block)
            trial[axis] = min(block[axis] + chunk_zyx[axis], shape_zyx[axis])
            if trial[0] * trial[1] * trial[2] > budget_voxels:
                break
            block = trial
    return tuple(block)


def _iter_blocks(shape_zyx, block_zyx) -> List[Tuple[int, int, int, int, int, int]]:
    z_size, y_size, x_size = shape_zyx
    bz, by, bx = block_zyx
    return [
        (z0, min(z0 + bz, z_size), y0, min(y0 + by, y_size), x0, min(x0 + bx, x_size))
        for z0 in range(0, z_size, bz)
        for y0 in range(0, y_size, by)
        for x0 in range(0, x_size, bx)
    ]


def _group_by_label(pair_ids: np.ndarray):
    order = np.argsort(pair_ids, kind="stable")
    uniq, starts = np.unique(pair_ids[order], return_index=True)
    return order, uniq, starts


class _PairAccumulator:

    def __init__(self, capacity: int, fields: Sequence[Tuple[str, Any, Any]]):
        self._fills = {name: fill for name, _, fill in fields}
        self._fields = {name: np.full(max(capacity, 1), fill, dtype=dtype)
                        for name, dtype, fill in fields}

    def __getitem__(self, name: str) -> np.ndarray:
        return self._fields[name]

    def ensure(self, max_id: int) -> None:
        needed = int(max_id) + 1
        for name, array in self._fields.items():
            if array.size >= needed:
                continue
            # Double rather than fit exactly, so a run of rising IDs reallocates a
            # logarithmic number of times instead of once per block.
            grown = np.full(max(needed, array.size * 2), self._fills[name], dtype=array.dtype)
            grown[:array.size] = array
            self._fields[name] = grown


# Workers. Each opens the stores itself from a path, so no array crosses a
# process boundary; the centroid table is memory-mapped and therefore shared.

_WORKER: Dict[str, Any] = {}


def _init_worker(
    interscellar_zarr: str,
    label_key: str,
    overlap_key: Optional[str],
    spot_specs: Sequence[Tuple[str, str, str]],
    centroid_path: Optional[str],
    voxel_size_um: Tuple[float, float, float],
    want_coords: bool,
) -> None:
    root = _open_store(interscellar_zarr, "interscellar zarr")
    _WORKER['labels'] = _node_by_key(root, label_key)
    _WORKER['overlap'] = None if overlap_key is None else _node_by_key(root, overlap_key)
    _WORKER['spots'] = [
        _node_by_key(_open_store(path, f"'{name}' spot zarr"), key)
        for name, path, key in spot_specs
    ]
    # mmap: the OS page cache shares one copy across workers, so a centroid table for
    # millions of pairs is not duplicated n_jobs times.
    _WORKER['centroids'] = (
        None if centroid_path is None else np.load(centroid_path, mmap_mode="r")
    )
    _WORKER['voxel'] = voxel_size_um
    _WORKER['want_coords'] = want_coords


def _pass1_block(bounds):
    labels = _read_block(_WORKER['labels'], bounds)
    occupied = labels > 0
    if not occupied.any():
        return None

    pair_ids = labels[occupied].astype(np.int64, copy=False)
    zz, yy, xx = np.nonzero(occupied)
    order, uniq, starts = _group_by_label(pair_ids)
    z0, _, y0, _, x0, _ = bounds

    return (
        uniq,
        np.diff(np.append(starts, pair_ids.size)),
        np.add.reduceat((zz[order] + z0).astype(np.float64), starts),
        np.add.reduceat((yy[order] + y0).astype(np.float64), starts),
        np.add.reduceat((xx[order] + x0).astype(np.float64), starts),
    )


def _pass2_block(bounds):
    labels = _read_block(_WORKER['labels'], bounds)
    occupied = labels > 0
    if not occupied.any():
        return None

    vz, vy, vx = _WORKER['voxel']
    z0, _, y0, _, x0, _ = bounds

    pair_ids = labels[occupied].astype(np.int64, copy=False)
    zz, yy, xx = np.nonzero(occupied)
    centre = np.asarray(_WORKER['centroids'][pair_ids], dtype=np.float64)
    distances = np.sqrt(
        (vz * ((zz + z0) - centre[:, 0])) ** 2
        + (vy * ((yy + y0) - centre[:, 1])) ** 2
        + (vx * ((xx + x0) - centre[:, 2])) ** 2
    )
    del centre

    order, uniq, starts = _group_by_label(pair_ids)
    ordered = distances[order]
    r_max = np.maximum.reduceat(ordered, starts)
    r_min = np.minimum.reduceat(ordered, starts)
    del ordered

    contested = None
    if _WORKER['overlap'] is not None:
        shared = _read_block(_WORKER['overlap'], bounds)[occupied] > 1
        contested = np.add.reduceat(shared[order].astype(np.int64), starts)

    spots = []
    for index, spot_array in enumerate(_WORKER['spots']):
        # 1 nonzero voxel == 1 spot. Spots outside every volume are ignored.
        inside = _read_block(spot_array, bounds)[occupied] != 0
        if not inside.any():
            continue
        coords = None
        if _WORKER['want_coords']:
            coords = np.stack([
                (zz[inside] + z0).astype(np.int32),
                (yy[inside] + y0).astype(np.int32),
                (xx[inside] + x0).astype(np.int32),
            ], axis=1)
        spots.append((
            index,
            pair_ids[inside].astype(np.int32),
            distances[inside].astype(np.float32),
            coords,
        ))

    return uniq, r_max.astype(np.float32), r_min.astype(np.float32), contested, spots


# Pass drivers

def _run_blocks(task, blocks, n_jobs, init_args, label, verbose=True):
    total = len(blocks)
    step = max(1, total // 20)

    def _tick(done: int) -> None:
        if verbose and (done % step == 0 or done == total):
            print(f"  {label}: {done}/{total} blocks", end="\r", flush=True)

    if n_jobs <= 1:
        _init_worker(*init_args)
        for done, bounds in enumerate(blocks, start=1):
            yield task(bounds)
            _tick(done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker, initargs=init_args
        ) as pool:
            futures = [pool.submit(task, bounds) for bounds in blocks]
            for done, future in enumerate(as_completed(futures), start=1):
                yield future.result()
                _tick(done)
    if verbose:
        print(f"  {label}: {total}/{total} blocks")


def _accumulate_geometry(blocks, n_jobs, init_args, capacity, verbose=True):
    """Pass 1: voxel count and centroid of every interscellar volume."""
    acc = _PairAccumulator(capacity, [
        ('counts', np.int64, 0), ('sum_z', np.float64, 0.0),
        ('sum_y', np.float64, 0.0), ('sum_x', np.float64, 0.0),
    ])

    for result in _run_blocks(_pass1_block, blocks, n_jobs, init_args, "centroids", verbose):
        if result is None:
            continue
        uniq, counts, sum_z, sum_y, sum_x = result
        acc.ensure(int(uniq[-1]))
        # uniq is unique within a block, so buffered fancy-index += is safe here and
        # is substantially faster than np.add.at.
        acc['counts'][uniq] += counts
        acc['sum_z'][uniq] += sum_z
        acc['sum_y'][uniq] += sum_y
        acc['sum_x'][uniq] += sum_x

    return acc


def _accumulate_scores(blocks, n_jobs, init_args, capacity, n_biomarkers, verbose=True):
    """Pass 2: R and d_min per pair, contested voxels, and every spot's distance."""
    acc = _PairAccumulator(capacity, [
        ('r_max', np.float32, 0.0), ('r_min', np.float32, np.inf),
        ('contested', np.int64, 0),
    ])
    collected = [{'ids': [], 'dist': [], 'coords': []} for _ in range(n_biomarkers)]

    for result in _run_blocks(_pass2_block, blocks, n_jobs, init_args, "scores", verbose):
        if result is None:
            continue
        uniq, r_max, r_min, contested, spots = result
        acc.ensure(int(uniq[-1]))
        acc['r_max'][uniq] = np.maximum(acc['r_max'][uniq], r_max)
        acc['r_min'][uniq] = np.minimum(acc['r_min'][uniq], r_min)
        if contested is not None:
            acc['contested'][uniq] += contested
        for index, ids, dist, coords in spots:
            collected[index]['ids'].append(ids)
            collected[index]['dist'].append(dist)
            if coords is not None:
                collected[index]['coords'].append(coords)

    def _join(chunks, dtype, width=None):
        if not chunks:
            return np.empty((0, width) if width else 0, dtype=dtype)
        return np.concatenate(chunks)

    spots_per_biomarker = [
        (
            _join(entry['ids'], np.int32),
            _join(entry['dist'], np.float32),
            _join(entry['coords'], np.int32, width=3) if entry['coords'] else None,
        )
        for entry in collected
    ]
    return acc, spots_per_biomarker


def score_from_distance(d_p, r_max, power: float = 1.0):
    """w_p = (1 - d_p / R) ** power.
    R == 0 means the volume is a single voxel (its own centroid).
    """
    positive = r_max > 0
    ramp = np.where(positive, 1.0 - d_p / np.where(positive, r_max, 1.0), 1.0)
    ramp = np.clip(ramp, 0.0, 1.0)
    return ramp if power == 1.0 else ramp ** power


# Naming and inputs

def _zarr_stem(path: str) -> str:
    basename = os.path.basename(path.rstrip(os.sep))
    stem = basename[:-5] if basename.endswith(".zarr") else os.path.splitext(basename)[0]
    if stem.endswith("_interscellar_volumes"):
        stem = stem[: -len("_interscellar_volumes")]
    return stem or "interscellar"


def _sanitize_biomarker(name: str) -> str:
    cleaned = "_".join(str(name).split())
    if not cleaned:
        raise ValueError("Biomarker name must be a non-empty string.")
    return cleaned


def _default_output_csv(interscellar_zarr: str) -> str:
    directory = os.path.dirname(os.path.abspath(interscellar_zarr.rstrip(os.sep))) or "."
    return os.path.join(directory, f"{_zarr_stem(interscellar_zarr)}_interscellar_scores.csv")


def _atomic_write_csv(frame: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _normalize_spot_inputs(spot_zarrs, biomarker, spot_keys):
    if isinstance(spot_zarrs, Mapping):
        entries = [(str(k), str(v)) for k, v in spot_zarrs.items()]
    else:
        items = [spot_zarrs] if isinstance(spot_zarrs, str) else list(spot_zarrs)
        entries = []
        for item in items:
            text = str(item)
            # Only split on '=' when the whole string is not itself a real path.
            if "=" in text and not os.path.exists(text):
                name, _, path = text.partition("=")
                entries.append((name, path))
            else:
                entries.append(("", text))

    if not entries:
        raise ValueError("At least one spot zarr must be provided.")

    named = []
    for name, path in entries:
        if not name:
            if biomarker and len(entries) == 1:
                name = biomarker
            elif biomarker:
                raise ValueError(
                    "--biomarker names a single spot zarr; with several, use NAME=PATH."
                )
            else:
                name = _zarr_stem(path)
        clean = _sanitize_biomarker(name)
        key = None if spot_keys is None else spot_keys.get(clean, spot_keys.get(name))
        named.append((clean, os.path.abspath(str(path).rstrip(os.sep)), key))

    names = [n for n, _, _ in named]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate biomarker names: {duplicates}")
    return named


def _load_volumes_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "pair_id" not in frame.columns:
        raise ValueError(f"{path} has no 'pair_id' column. Found: {sorted(frame.columns)}")

    frame = frame.rename(columns={'cell_id_a': 'cell_a_id', 'cell_id_b': 'cell_b_id'})
    keep = ['pair_id', 'cell_a_id', 'cell_b_id', 'total_interscellar_volume_um3']
    frame = frame[[c for c in keep if c in frame.columns]].copy()

    frame['pair_id'] = pd.to_numeric(frame['pair_id'], errors="coerce")
    dropped = int(frame['pair_id'].isna().sum())
    if dropped:
        print(f"  Warning: {dropped} rows in {path} have a non-numeric pair_id; skipped")
    frame = frame.dropna(subset=['pair_id'])
    frame['pair_id'] = frame['pair_id'].astype(np.int64)

    duplicated = frame['pair_id'].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, 'pair_id'].unique()[:10].tolist()
        raise ValueError(
            f"{path} has {int(duplicated.sum())} duplicate pair_id rows "
            f"(examples: {examples}). Each pair must appear once."
        )
    if (frame['pair_id'] <= 0).any():
        raise ValueError(
            f"{path} has {int((frame['pair_id'] <= 0).sum())} rows with pair_id <= 0. "
            f"Zero is the background label in the mesh zarr, so pair IDs must be positive."
        )
    return frame


# Public API

def calculate_interscellar_scores_3d(
    interscellar_zarr: str,
    spot_zarrs: Any,
    biomarker: Optional[str] = None,
    volumes_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
    n_jobs: int = 1,
    decay_power: float = 1.0,
    interscellar_key: Optional[str] = None,
    spot_keys: Optional[Mapping[str, str]] = None,
    block_mb: int = 128,
    per_point_csv: Optional[str] = None,
    overlap_qc: bool = True,
    verbose: bool = True,
) -> str:
    interscellar_zarr = os.path.abspath(interscellar_zarr.rstrip(os.sep))
    if not os.path.exists(interscellar_zarr):
        raise FileNotFoundError(f"Interscellar zarr not found: {interscellar_zarr}")
    if not np.isfinite(decay_power) or decay_power <= 0:
        raise ValueError(f"decay_power must be a positive number, got {decay_power}")
    if block_mb <= 0:
        raise ValueError(f"block_mb must be positive, got {block_mb}")
    n_jobs = max(1, int(n_jobs))

    spots_in = _normalize_spot_inputs(spot_zarrs, biomarker, spot_keys)
    for name, path, _ in spots_in:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Spot zarr for '{name}' not found: {path}")

    output_csv = os.path.abspath(output_csv or _default_output_csv(interscellar_zarr))

    label_root = _open_store(interscellar_zarr, "interscellar zarr")
    label_key, labels = _resolve_array(
        label_root, interscellar_key, _PAIR_LABEL_KEYS, "interscellar"
    )
    if not np.issubdtype(np.dtype(labels.dtype), np.integer):
        raise ValueError(
            f"Interscellar array '{label_key}' has dtype {labels.dtype}; pair labels "
            f"must be an integer type."
        )
    shape_zyx = _spatial_shape_zyx(labels)
    _, label_voxel = _parse_ngff_grid(label_root)
    print(f"Loading interscellar zarr: {interscellar_zarr}")
    print(f"  Pair-label array '{label_key}' {labels.dtype} shape (Z, Y, X): {shape_zyx}")

    overlap_key = None
    if overlap_qc and hasattr(label_root, "keys") and "overlap_count" in label_root:
        overlap_key = "overlap_count"
        print("  Found 'overlap_count'; reporting contested voxels per pair")

    spot_specs: List[Tuple[str, str, str]] = []
    spot_itemsizes: List[int] = []
    for name, path, key in spots_in:
        root = _open_store(path, f"'{name}' spot zarr")
        resolved, array = _resolve_array(root, key, ("spots", "0", "labels"), f"'{name}' spot")
        dtype = np.dtype(array.dtype)
        if not (np.issubdtype(dtype, np.integer) or dtype == np.bool_):
            print(
                f"  Warning: spot array for '{name}' has dtype {dtype}; every nonzero "
                f"voxel counts as exactly one spot regardless of its value"
            )
        _, spot_voxel = _parse_ngff_grid(root)
        _validate_grid(shape_zyx, label_voxel, _spatial_shape_zyx(array), spot_voxel, name)
        spot_specs.append((name, path, resolved))
        spot_itemsizes.append(dtype.itemsize)
        print(f"  Spot '{name}': '{resolved}' {dtype} from {path}")

    if voxel_size_um is not None:
        vz, vy, vx = (float(v) for v in voxel_size_um)
        source = "argument"
    elif label_voxel is not None:
        vz, vy, vx = label_voxel
        source = "OME-NGFF metadata"
    else:
        stored = label_root.attrs.get("voxel_size_um") if hasattr(label_root, "attrs") else None
        try:
            vz, vy, vx = (float(v) for v in stored)
            source = "voxel_size_um attribute"
        except (TypeError, ValueError):
            vz, vy, vx = DEFAULT_VOXEL_SIZE_UM
            source = "package default"
            print(f"  Warning: no voxel size found in the zarr; using {DEFAULT_VOXEL_SIZE_UM}")
    if min(vz, vy, vx) <= 0:
        raise ValueError(f"Voxel size must be positive in every axis, got {(vz, vy, vx)}")
    voxel_volume_um3 = float(vz * vy * vx)
    print(f"  Voxel size (Z, Y, X) um: {(vz, vy, vx)} from {source}")

    volumes_frame = None
    capacity = 1024
    if volumes_csv:
        volumes_frame = _load_volumes_csv(volumes_csv)
        if len(volumes_frame):
            capacity = int(volumes_frame['pair_id'].max()) + 1
        print(f"Loaded {len(volumes_frame)} pairs from {volumes_csv}")

    chunk = _chunk_zyx(labels)
    per_voxel = np.dtype(labels.dtype).itemsize + sum(spot_itemsizes)
    if overlap_key is not None:
        per_voxel += np.dtype(_node_by_key(label_root, overlap_key).dtype).itemsize
    block = _choose_block(chunk, shape_zyx, block_mb * 1024 * 1024, per_voxel)
    blocks = _iter_blocks(shape_zyx, block)
    print(
        f"Traversing {len(blocks)} blocks of {block} (chunk {chunk}) with "
        f"{n_jobs} worker(s), ~{block_mb} MiB of array per block"
    )

    geometry = _accumulate_geometry(
        blocks, n_jobs,
        (interscellar_zarr, label_key, None, (), None, (vz, vy, vx), False),
        capacity, verbose,
    )
    counts = geometry['counts']
    present = counts > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError(
            f"No labeled voxels found in '{label_key}'. Check that "
            f"{interscellar_zarr} is an interscellar volumes zarr."
        )

    # Spot pair IDs are carried as int32 to halve the per-spot memory; a label above
    # that range would wrap silently, so refuse it outright rather than mis-assign.
    if counts.size - 1 > np.iinfo(np.int32).max:
        raise ValueError(
            f"Largest pair_id in '{label_key}' is {counts.size - 1}, beyond the int32 "
            f"range this script indexes spots with."
        )

    centroids = np.zeros((counts.size, 3), dtype=np.float32)
    centroids[present, 0] = geometry['sum_z'][present] / counts[present]
    centroids[present, 1] = geometry['sum_y'][present] / counts[present]
    centroids[present, 2] = geometry['sum_x'][present] / counts[present]
    del geometry
    print(f"  {n_present} interscellar volumes have voxels in this zarr")

    want_coords = per_point_csv is not None
    handle, centroid_path = tempfile.mkstemp(prefix="isc_centroids_", suffix=".npy")
    os.close(handle)
    try:
        np.save(centroid_path, centroids)
        scores, spot_data = _accumulate_scores(
            blocks, n_jobs,
            (interscellar_zarr, label_key, overlap_key, tuple(spot_specs),
             centroid_path, (vz, vy, vx), want_coords),
            counts.size, len(spot_specs), verbose,
        )
    finally:
        if os.path.exists(centroid_path):
            os.unlink(centroid_path)

    r_max = scores['r_max'].astype(np.float64)
    r_min = np.where(np.isfinite(scores['r_min']), scores['r_min'], np.nan).astype(np.float64)
    n_slots = counts.size

    if volumes_frame is not None:
        table = volumes_frame.copy()
    else:
        table = pd.DataFrame({'pair_id': np.nonzero(present)[0].astype(np.int64)})

    pair_ids = table['pair_id'].to_numpy()
    in_range = (pair_ids >= 0) & (pair_ids < n_slots)
    gather = np.where(in_range, pair_ids, 0)

    def _pick(source: np.ndarray, missing: Any) -> np.ndarray:
        return np.where(in_range, source[gather], missing)

    table['n_voxels'] = _pick(counts, 0).astype(np.int64)
    has_voxels = table['n_voxels'].to_numpy() > 0
    table['interscellar_volume_um3'] = table['n_voxels'] * voxel_volume_um3
    if 'total_interscellar_volume_um3' in table.columns:
        table = table.rename(columns={'total_interscellar_volume_um3': 'volumes_csv_volume_um3'})
    else:
        table['volumes_csv_volume_um3'] = np.nan

    if overlap_key is not None:
        contested = _pick(scores['contested'], 0).astype(np.int64)
        table['contested_voxels'] = contested
        table['contested_fraction'] = np.where(
            has_voxels, contested / np.maximum(table['n_voxels'].to_numpy(), 1), np.nan
        )
    else:
        table['contested_voxels'] = -1
        table['contested_fraction'] = np.nan

    for axis, index in (('z', 0), ('y', 1), ('x', 2)):
        scaled = centroids[:, index].astype(np.float64) * (vz, vy, vx)[index]
        table[f'centroid_{axis}_um'] = np.where(has_voxels, _pick(scaled, np.nan), np.nan)

    table['min_distance_um'] = np.where(has_voxels, _pick(r_min, np.nan), np.nan)
    table['max_distance_um'] = np.where(has_voxels, _pick(r_max, np.nan), np.nan)
    # A non-convex volume can centroid into background, putting w = 1 out of reach for
    # every spot. This is the ceiling the volume's own shape imposes on any score.
    table['max_attainable_weight'] = np.where(
        has_voxels,
        score_from_distance(
            table['min_distance_um'].to_numpy(),
            table['max_distance_um'].to_numpy(),
            decay_power,
        ),
        np.nan,
    )

    volume_um3 = table['interscellar_volume_um3'].to_numpy()
    per_biomarker, point_frames = [], []

    for (name, _, _), (spot_ids, spot_distances, spot_coords) in zip(spot_specs, spot_data):
        ids = spot_ids.astype(np.int64, copy=False)
        weights = score_from_distance(
            spot_distances.astype(np.float64), r_max[ids], decay_power
        )
        n_spots = np.bincount(ids, minlength=n_slots)
        score_sum = np.bincount(ids, weights=weights, minlength=n_slots)

        median = np.full(n_slots, np.nan, dtype=np.float64)
        if ids.size:
            grouped = pd.Series(spot_distances.astype(np.float64)).groupby(ids).median()
            median[grouped.index.to_numpy()] = grouped.to_numpy()

        rows = table.copy()
        rows['biomarker'] = name
        rows['n_spots'] = _pick(n_spots, 0).astype(np.int64)
        sums = _pick(score_sum, 0.0)
        counted = rows['n_spots'].to_numpy()
        rows['score_sum'] = sums
        rows['score_mean'] = np.where(counted > 0, sums / np.maximum(counted, 1), np.nan)
        rows['score_per_um3'] = np.where(
            volume_um3 > 0, sums / np.where(volume_um3 > 0, volume_um3, 1.0), np.nan
        )
        rows['median_dist_um'] = _pick(median, np.nan)
        rows['decay_power'] = decay_power
        per_biomarker.append(rows)
        print(f"  {name}: {int(ids.size)} spots inside an interscellar volume")

        if want_coords and ids.size and spot_coords is not None:
            point_frames.append(pd.DataFrame({
                'pair_id': spot_ids,
                'biomarker': name,
                'z': spot_coords[:, 0],
                'y': spot_coords[:, 1],
                'x': spot_coords[:, 2],
                'dist_um': spot_distances,
                'weight': weights,
            }))

    long_table = pd.concat(per_biomarker, ignore_index=True)
    ordered = [c for c in PAIR_COLUMNS + SCORE_COLUMNS if c in long_table.columns]
    ordered += [c for c in long_table.columns if c not in ordered]
    long_table = long_table[ordered].sort_values(
        ['biomarker', 'pair_id']
    ).reset_index(drop=True)

    _atomic_write_csv(long_table, output_csv)

    orphans = sorted(set(np.nonzero(present)[0].tolist())
                     - set(table.loc[has_voxels, 'pair_id'].tolist()))
    if orphans:
        print(
            f"  Warning: {len(orphans)} pair IDs in the zarr are absent from the pair "
            f"roster and were not scored (examples: {orphans[:10]})"
        )
    empty = int((~has_voxels).sum())
    if empty:
        print(f"  Note: {empty} pairs have no voxels in the zarr and scored 0")
    if overlap_key is not None:
        shared_pairs = int((table['contested_voxels'] > 0).sum())
        if shared_pairs:
            mean_fraction = float(
                np.nanmean(table.loc[table['contested_voxels'] > 0, 'contested_fraction'])
            )
            print(
                f"  Note: {shared_pairs} pairs share voxels with another pair "
                f"(mean {100 * mean_fraction:.1f}% of the footprint). Scores describe "
                f"the footprint each pair owns in the mesh zarr."
            )
    unreachable = int((table['max_attainable_weight'] < 0.99).sum())
    if unreachable:
        print(
            f"  Note: {unreachable} of {int(has_voxels.sum())} volumes cannot reach "
            f"weight 1.0 -- their centroid lies outside the volume (curved shells)"
        )

    if point_frames:
        _atomic_write_csv(
            pd.concat(point_frames, ignore_index=True).sort_values(
                ['biomarker', 'pair_id', 'dist_um']
            ),
            os.path.abspath(per_point_csv),
        )
        print(f"Wrote per-point values to: {os.path.abspath(per_point_csv)}")

    print(
        f"Wrote {len(long_table)} rows ({len(table)} pairs x "
        f"{len(spot_specs)} biomarkers) to: {output_csv}"
    )
    return output_csv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interscellar score for point-form biomarkers. Detects spot-mask voxels "
            "inside each interscellar volume and weights each by w = (1 - d/R)^power, "
            "where d is its distance from the volume's centroid and R the volume's "
            "extent from that centroid. Emits a long-format CSV, one row per pair_id "
            "and biomarker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interscellar-zarr", required=True,
        help="Interscellar volumes zarr from step 2; voxels labeled by pair_id.",
    )
    parser.add_argument(
        "--spot-zarr", required=True, action="append", metavar="[NAME=]PATH",
        help=(
            "Biomarker spot mask on the same voxel grid; 1 nonzero voxel = 1 spot. "
            "Repeat for several biomarkers, naming each as NAME=PATH."
        ),
    )
    parser.add_argument(
        "--biomarker", default=None,
        help="Name for a single --spot-zarr given without a NAME= prefix.",
    )
    parser.add_argument(
        "--volumes-csv", default=None,
        help=(
            "Per-pair volumes CSV from step 2. Supplies cell IDs and the full pair "
            "roster so pairs with no spots still get a row."
        ),
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Output CSV. Defaults to <zarr_dir>/<stem>_interscellar_scores.csv",
    )
    parser.add_argument(
        "--per-point-csv", default=None,
        help="Optional per-spot dump: pair_id, biomarker, z, y, x, dist_um, weight.",
    )
    parser.add_argument(
        "--decay-power", type=float, default=1.0,
        help="Exponent on the ramp: 1.0 is linear, >1 concentrates weight at the centroid.",
    )
    parser.add_argument(
        "--voxel-size-um", nargs=3, type=float, default=None, metavar=("Z", "Y", "X"),
        help=(
            "Voxel size in micrometers. Defaults to the zarr's OME-NGFF scale, then its "
            f"voxel_size_um attribute, then {DEFAULT_VOXEL_SIZE_UM}."
        ),
    )
    parser.add_argument(
        "--interscellar-key", default=None,
        help="Array key inside the interscellar zarr (auto-detected if omitted).",
    )
    parser.add_argument(
        "--spot-key", default=None, action="append", metavar="NAME=KEY",
        help="Array key inside a named spot zarr (auto-detected if omitted).",
    )
    parser.add_argument(
        "--block-mb", type=int, default=128,
        help="Array bytes per 3D block. Peak per worker is a few times this.",
    )
    parser.add_argument(
        "--no-overlap-qc", action="store_true",
        help="Skip reading 'overlap_count'; saves one array read per block.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker processes. Blocks are distributed across them.",
    )
    args = parser.parse_args(argv)

    spot_keys = None
    if args.spot_key:
        spot_keys = {}
        for item in args.spot_key:
            name, sep, key = item.partition("=")
            if not sep:
                parser.error(f"--spot-key must be NAME=KEY, got '{item}'")
            spot_keys[_sanitize_biomarker(name)] = key

    try:
        calculate_interscellar_scores_3d(
            interscellar_zarr=args.interscellar_zarr,
            spot_zarrs=args.spot_zarr,
            biomarker=args.biomarker,
            volumes_csv=args.volumes_csv,
            output_csv=args.output_csv,
            voxel_size_um=tuple(args.voxel_size_um) if args.voxel_size_um else None,
            n_jobs=args.n_jobs,
            decay_power=args.decay_power,
            interscellar_key=args.interscellar_key,
            spot_keys=spot_keys,
            block_mb=args.block_mb,
            per_point_csv=args.per_point_csv,
            overlap_qc=not args.no_overlap_qc,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

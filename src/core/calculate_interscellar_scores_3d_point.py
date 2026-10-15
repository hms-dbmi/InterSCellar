from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import distance_transform_edt

DEFAULT_VOXEL_SIZE_UM = (0.56, 0.28, 0.28)

_ARCHIVE_GROUP = "pair_volumes"
_FOOTPRINT_KEYS = ("territory_a", "corridor", "territory_b")
_INTERFACE_KEYS = ("interface_a", "interface_b")
_COMPONENT_KEYS = _FOOTPRINT_KEYS + _INTERFACE_KEYS
_INTERFACE_KIND_NAME = {1: "direct", 2: "near"}

_UNIT_TO_UM = {
    "": 1.0, "micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0,
    "nanometer": 1e-3, "nm": 1e-3,
    "millimeter": 1e3, "mm": 1e3,
    "centimeter": 1e4, "cm": 1e4,
    "meter": 1e6, "m": 1e6,
}

PAIR_COLUMNS = [
    'pair_id', 'cell_a_id', 'cell_b_id', 'interface_kind',
    'n_voxels', 'interscellar_volume_um3', 'volumes_csv_volume_um3',
    'territory_a_voxels', 'corridor_voxels', 'territory_b_voxels',
    'interface_voxels', 'interface_a_voxels', 'interface_b_voxels',
    'shared_voxels', 'shared_fraction',
    'interface_dist_max_um', 'interface_dist_mean_um',
    'voxels_beyond_reference', 'fraction_beyond_reference',
    'min_surface_separation_um',
    'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
    'reference_distance_um', 'decay_power',
]
SCORE_COLUMNS = [
    'biomarker', 'n_spots', 'spots_beyond_reference',
    'score_sum', 'score_mean', 'score_per_um3',
    'mean_dist_um', 'median_dist_um',
]

# Zarr access and OME-NGFF grid metadata

_ZARR_MAJOR = int(str(zarr.__version__).split(".")[0])


def _on_disk_zarr_format(path: str) -> Optional[int]:
    if os.path.exists(os.path.join(path, "zarr.json")):
        return 3
    if os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(
        os.path.join(path, ".zarray")
    ):
        return 2
    return None


def _open_store(path: str, what: str) -> Any:
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


# Lossless pair archive

def _archive_missing_message(path: str) -> str:
    return (
        f"{path} has no '{_ARCHIVE_GROUP}' group, so it cannot be scored.\n"
        f"  The interscellar score is measured from each pair's contact or facing "
        f"interface, and it counts a shared voxel once for every pair that claims it. "
        f"Both need the lossless per-pair archive: a dense pair-label array keeps only "
        f"the highest pair_id per voxel and stores no interface at all.\n"
        f"  Rebuild the volumes with compute_interscellar_volumes_3d_adaptive.py "
        f"(--out-zarr writes '{_ARCHIVE_GROUP}' alongside the preview arrays). Stores "
        f"from compute_interscellar_volumes_3d_absolute.py cannot be scored this way."
    )


def _open_archive(interscellar_zarr: str):
    root = _open_store(interscellar_zarr, "interscellar zarr")
    if not hasattr(root, "keys") or _ARCHIVE_GROUP not in root:
        raise ValueError(_archive_missing_message(interscellar_zarr))
    group = root[_ARCHIVE_GROUP]
    if "pair_id" not in group:
        raise ValueError(
            f"{interscellar_zarr}/{_ARCHIVE_GROUP} has no 'pair_id' array; the archive is "
            f"incomplete. Rebuild it with compute_interscellar_volumes_3d_adaptive.py."
        )
    return root, group


def _committed_rows(group: Any) -> int:
    on_disk = int(group['pair_id'].shape[0])
    stated = group.attrs.get("committed_rows")
    if stated is None:
        return on_disk
    return max(0, min(int(stated), on_disk))


def _read_rows(group: Any, name: str, n_rows: int, missing: Any = None):
    if name not in group:
        return missing
    return np.asarray(group[name][:n_rows])


def _read_spans(group: Any, name: str, n_rows: int) -> np.ndarray:
    key = f"{name}_indptr"
    if key not in group or name not in group:
        return np.zeros((n_rows, 2), dtype=np.int64)
    indptr = np.asarray(group[key][:n_rows + 1]).astype(np.int64, copy=False)
    if indptr.size < n_rows + 1:
        raise ValueError(
            f"'{key}' holds {indptr.size} offsets but the archive commits {n_rows} rows "
            f"(it needs {n_rows + 1}). The store was written by an interrupted run."
        )
    limit = int(group[name].shape[0])
    if int(indptr.max(initial=0)) > limit:
        raise ValueError(
            f"'{key}' points past the end of '{name}' ({int(indptr.max())} > {limit}); "
            f"the archive is truncated. Rerun the volume build for this store."
        )
    return np.stack([indptr[:-1], indptr[1:]], axis=1)


def _decode_component(indices: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(int(shape[0]) * int(shape[1]) * int(shape[2]), dtype=bool)
    if indices.size:
        mask[np.cumsum(indices, dtype=np.uint64)] = True
    return mask.reshape(tuple(int(v) for v in shape))


def _read_archive_table(group: Any, path: str) -> Dict[str, Any]:
    n_rows = _committed_rows(group)
    if n_rows == 0:
        raise ValueError(
            f"{path}/{_ARCHIVE_GROUP} commits 0 pairs. Either every pair was rejected as "
            f"unbridged (check the rejected-pairs CSV) or the volume build never ran."
        )

    pair_ids = np.asarray(group['pair_id'][:n_rows]).astype(np.int64, copy=False)
    duplicated = pd.Index(pair_ids).duplicated()
    if duplicated.any():
        examples = np.unique(pair_ids[duplicated])[:10].tolist()
        raise ValueError(
            f"{path}/{_ARCHIVE_GROUP} holds {int(duplicated.sum())} duplicate pair_id "
            f"rows (examples: {examples}). Each pair must be archived once; this store "
            f"was most likely resumed into after its commit counter was lost."
        )

    origins = np.asarray(group['origin_zyx'][:n_rows]).astype(np.int64, copy=False)
    shapes = np.asarray(group['shape_zyx'][:n_rows]).astype(np.int64, copy=False)
    if (shapes <= 0).any():
        bad = pair_ids[(shapes <= 0).any(axis=1)][:10].tolist()
        raise ValueError(f"Archived crop shapes are non-positive for pairs {bad}.")

    table = {
        'n_rows': n_rows,
        'pair_id': pair_ids,
        'origin_zyx': origins,
        'shape_zyx': shapes,
        'spans': {name: _read_spans(group, name, n_rows) for name in _COMPONENT_KEYS},
    }
    for name in (
        'cell_a_id', 'cell_b_id', 'interface_kind', 'n_voxels', 'n_territory_a',
        'n_corridor', 'n_territory_b', 'n_interface_a', 'n_interface_b',
        'centroid_zyx_um', 'min_surface_separation_um', 'volume_um3', 'shared_voxels',
    ):
        table[name] = _read_rows(group, name, n_rows)
    return table

# Voxel grid agreement between the pair archive and the spot masks

def _declared_volume_shape(root: Any, group: Any):
    for key in ("interscellar_meshes", "overlap_count"):
        if hasattr(root, "keys") and key in root and hasattr(root[key], "shape"):
            return _spatial_shape_zyx(root[key]), f"the '{key}' preview array"
    for holder, label in ((root, "store"), (group, _ARCHIVE_GROUP)):
        attrs = getattr(holder, "attrs", None)
        stated = attrs.get("volume_shape_zyx") if attrs is not None else None
        if stated is not None and len(list(stated)) == 3:
            return tuple(int(v) for v in stated), f"the {label} 'volume_shape_zyx' attribute"
    return None, None


def _check_grid(
    shape_zyx: Tuple[int, int, int],
    origins: np.ndarray,
    shapes: np.ndarray,
    pair_ids: np.ndarray,
    source: str,
) -> None:
    upper = origins + shapes
    outside = (origins < 0).any(axis=1) | (upper > np.asarray(shape_zyx)).any(axis=1)
    if not outside.any():
        return
    examples = [
        f"pair {int(pair_ids[row])}: "
        f"{tuple(int(v) for v in origins[row])}..{tuple(int(v) for v in upper[row])}"
        for row in np.flatnonzero(outside)[:5]
    ]
    raise ValueError(
        f"{int(outside.sum())} of {len(pair_ids)} archived pair crops reach outside the "
        f"{shape_zyx} grid taken from {source}:\n    " + "\n    ".join(examples) + "\n"
        f"  The pair volumes and the spot masks were built on different crops of the "
        f"segmentation. Re-export the spot masks on the same grid as the volumes, or "
        f"rebuild the volumes on the grid the spot masks use."
    )


def _resolve_grid(
    spots_in: Sequence[Tuple[str, str, Optional[str]]],
    declared_shape: Optional[Tuple[int, int, int]],
    declared_source: Optional[str],
    label_voxel: Optional[Tuple[float, float, float]],
):
    specs: List[Tuple[str, str, str]] = []
    shapes: Dict[str, Tuple[int, int, int]] = {}
    voxels: Dict[str, Any] = {}

    for name, path, key in spots_in:
        root = _open_store(path, f"'{name}' spot zarr")
        resolved, array = _resolve_array(root, key, ("spots", "0", "labels"), f"'{name}' spot")
        dtype = np.dtype(array.dtype)
        if not (np.issubdtype(dtype, np.integer) or dtype == np.bool_):
            print(
                f"  Warning: spot array for '{name}' has dtype {dtype}; every nonzero "
                f"voxel counts as exactly one spot regardless of its value"
            )
        shapes[name] = _spatial_shape_zyx(array)
        _, voxels[name] = _parse_ngff_grid(root)
        specs.append((name, path, resolved))
        print(f"  Spot '{name}': '{resolved}' {dtype} shape {shapes[name]} from {path}")

    distinct = sorted(set(shapes.values()))
    if len(distinct) > 1:
        detail = "; ".join(f"'{n}' is {s}" for n, s in sorted(shapes.items()))
        raise ValueError(
            f"The spot masks are not all on one voxel grid: {detail}. Every spot mask "
            f"must cover the same region of the segmentation as the pair volumes."
        )

    spot_shape = distinct[0]
    if declared_shape is not None:
        if spot_shape != declared_shape:
            raise ValueError(
                f"The spot masks are {spot_shape} but the interscellar volumes were built "
                f"on a {declared_shape} grid, according to {declared_source}. Both must be "
                f"the same shape in (Z, Y, X); a transposed or differently cropped spot "
                f"mask would silently score the wrong voxels."
            )
        grid, source = declared_shape, declared_source
    else:
        grid, source = spot_shape, "the spot masks"
        print(
            f"  Note: the volume store does not record its full-volume shape (no preview "
            f"arrays, no 'volume_shape_zyx' attribute); adopting {grid} from the spot masks "
            f"and checking every pair crop against it"
        )

    if label_voxel is not None:
        for name, spot_voxel in voxels.items():
            if spot_voxel is None:
                continue
            if not np.allclose(label_voxel, spot_voxel, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"Spot zarr '{name}' declares voxel size {spot_voxel} um but the "
                    f"interscellar zarr declares {label_voxel} um. Both must match."
                )
    return specs, grid, source


_WORKER: Dict[str, Any] = {}


def _init_worker(
    interscellar_zarr: str,
    spot_specs: Sequence[Tuple[str, str, str]],
    voxel_size_um: Tuple[float, float, float],
    reference_distance_um: float,
    want_coords: bool,
) -> None:
    root = _open_store(interscellar_zarr, "interscellar zarr")
    _WORKER['group'] = root[_ARCHIVE_GROUP]
    _WORKER['spots'] = [
        _node_by_key(_open_store(path, f"'{name}' spot zarr"), key)
        for name, path, key in spot_specs
    ]
    _WORKER['voxel'] = tuple(float(v) for v in voxel_size_um)
    _WORKER['reference'] = float(reference_distance_um)
    _WORKER['want_coords'] = want_coords


def _score_pair(row: int, origin, shape, spans):
    group = _WORKER['group']
    shape = tuple(int(v) for v in shape)
    origin = tuple(int(v) for v in origin)

    masks = {}
    for name, (lo, hi) in zip(_COMPONENT_KEYS, spans):
        indices = (
            np.asarray(group[name][int(lo):int(hi)]) if hi > lo
            else np.zeros(0, dtype=np.uint32)
        )
        masks[name] = _decode_component(indices, shape)

    footprint = masks['territory_a'] | masks['corridor'] | masks['territory_b']
    interface = masks['interface_a'] | masks['interface_b']
    n_footprint = int(footprint.sum())
    n_interface = int(interface.sum())

    empty = (row, n_footprint, n_interface, np.nan, np.nan, 0, [])
    if n_footprint == 0 or n_interface == 0:
        return empty

    distance = distance_transform_edt(~interface, sampling=_WORKER['voxel']).astype(np.float32)

    footprint_distance = distance[footprint]
    reference = _WORKER['reference']
    beyond = int((footprint_distance > reference).sum()) if np.isfinite(reference) else 0
    stats = (
        row,
        n_footprint,
        n_interface,
        float(footprint_distance.max()),
        float(footprint_distance.mean()),
        beyond,
    )

    bounds = (
        origin[0], origin[0] + shape[0],
        origin[1], origin[1] + shape[1],
        origin[2], origin[2] + shape[2],
    )
    spots = []
    for index, spot_array in enumerate(_WORKER['spots']):
        # 1 nonzero voxel == 1 spot
        inside = footprint & (_read_block(spot_array, bounds) != 0)
        if not inside.any():
            continue
        coords = None
        if _WORKER['want_coords']:
            local = np.argwhere(inside)
            coords = (local + np.asarray(origin, dtype=np.int64)).astype(np.int32)
        spots.append((index, distance[inside].astype(np.float32), coords))

    return stats + (spots,)


def _score_batch(batch):
    return [_score_pair(*item) for item in batch]


def _run_pairs(items, n_jobs, init_args, verbose=True):
    import time

    total = len(items)
    total_pairs = sum(len(batch) for batch in items)
    started = time.time()
    state = {'pairs': 0, 'last': 0.0}

    def _tick(done: int, force: bool = False) -> None:
        now = time.time()
        if not verbose or (not force and now - state['last'] < 2.0):
            return
        state['last'] = now
        elapsed = now - started
        rate = state['pairs'] / elapsed if elapsed > 0 else 0.0
        left = (total_pairs - state['pairs']) / rate / 60.0 if rate > 0 else float('nan')
        print(
            f"  scoring: {done}/{total} batches | {state['pairs']}/{total_pairs} pairs "
            f"| {rate:.0f} pairs/s | ~{left:.1f} min left    ",
            end="\r", flush=True,
        )

    if n_jobs <= 1:
        _init_worker(*init_args)
        for done, batch in enumerate(items, start=1):
            yield _score_batch(batch)
            state['pairs'] += len(batch)
            _tick(done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker, initargs=init_args
        ) as pool:
            futures = [pool.submit(_score_batch, batch) for batch in items]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                yield result
                state['pairs'] += len(result)
                _tick(done)
    if verbose:
        elapsed = max(time.time() - started, 1e-9)
        print(
            f"  scoring: {total}/{total} batches | {total_pairs} pairs in "
            f"{elapsed / 60.0:.1f} min ({total_pairs / elapsed:.0f} pairs/s)"
        )


def _batch_pairs(table: Dict[str, Any], batch_size: int) -> List[List[Tuple]]:
    origins = table['origin_zyx']
    order = np.lexsort((origins[:, 2], origins[:, 1], origins[:, 0]))
    spans = table['spans']

    items = [
        (
            int(row),
            tuple(int(v) for v in origins[row]),
            tuple(int(v) for v in table['shape_zyx'][row]),
            tuple(tuple(int(v) for v in spans[name][row]) for name in _COMPONENT_KEYS),
        )
        for row in order
    ]
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]


def score_from_distance(d_p, reference_distance_um: float, power: float = 1.0):
    """w_p = (1 - d_p / D) ** power, where d_p is the spot's distance from the pair's
    interface and D is the reference distance that sets where the weight reaches zero.
    """
    distances = np.asarray(d_p, dtype=np.float64)
    if not np.isfinite(reference_distance_um) or reference_distance_um <= 0:
        return np.full(distances.shape, np.nan)
    ramp = np.clip(1.0 - distances / reference_distance_um, 0.0, 1.0)
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
    return os.path.join(directory, f"{_zarr_stem(interscellar_zarr)}_scores.csv")


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
    keep = ['pair_id', 'total_interscellar_volume_um3']
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
    return frame.rename(columns={'total_interscellar_volume_um3': 'volumes_csv_volume_um3'})


def _resolve_reference_distance(requested, root: Any, group: Any):
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        return np.inf, "auto (deepest interface distance observed)"
    if requested is not None:
        value = float(requested)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"reference_distance_um must be a positive number or 'auto', got {requested}"
            )
        return value, "argument"
    for holder, label in ((root, "store"), (group, _ARCHIVE_GROUP)):
        attrs = getattr(holder, "attrs", None)
        stated = attrs.get("max_distance_um") if attrs is not None else None
        if stated is None:
            continue
        value = float(stated)
        if np.isfinite(value) and value > 0:
            return value, f"the {label} 'max_distance_um' attribute"
    raise ValueError(
        "No reference distance for the decay ramp: the store records no "
        "'max_distance_um' attribute. Pass --reference-distance-um, or 'auto' to use the "
        "deepest interface distance in the data."
    )


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
    reference_distance_um: Any = None,
    spot_keys: Optional[Mapping[str, str]] = None,
    pair_batch: int = 32,
    per_point_csv: Optional[str] = None,
    verbose: bool = True,
) -> str:
    interscellar_zarr = os.path.abspath(interscellar_zarr.rstrip(os.sep))
    if not os.path.exists(interscellar_zarr):
        raise FileNotFoundError(f"Interscellar zarr not found: {interscellar_zarr}")
    if not np.isfinite(decay_power) or decay_power <= 0:
        raise ValueError(f"decay_power must be a positive number, got {decay_power}")
    if pair_batch <= 0:
        raise ValueError(f"pair_batch must be positive, got {pair_batch}")
    n_jobs = max(1, int(n_jobs))

    spots_in = _normalize_spot_inputs(spot_zarrs, biomarker, spot_keys)
    for name, path, _ in spots_in:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Spot zarr for '{name}' not found: {path}")

    output_csv = os.path.abspath(output_csv or _default_output_csv(interscellar_zarr))

    print(f"Loading interscellar zarr: {interscellar_zarr}")
    root, group = _open_archive(interscellar_zarr)
    table = _read_archive_table(group, interscellar_zarr)
    n_rows = table['n_rows']
    print(f"  Lossless pair archive '{_ARCHIVE_GROUP}': {n_rows} pair footprints")

    declared_shape, declared_source = _declared_volume_shape(root, group)
    _, label_voxel = _parse_ngff_grid(root)
    spot_specs, grid_shape, grid_source = _resolve_grid(
        spots_in, declared_shape, declared_source, label_voxel
    )
    _check_grid(grid_shape, table['origin_zyx'], table['shape_zyx'], table['pair_id'],
                grid_source)
    print(f"  Voxel grid (Z, Y, X): {grid_shape} from {grid_source}; all pair crops fit")

    if voxel_size_um is not None:
        vz, vy, vx = (float(v) for v in voxel_size_um)
        source = "argument"
    elif label_voxel is not None:
        vz, vy, vx = label_voxel
        source = "OME-NGFF metadata"
    else:
        stored = root.attrs.get("voxel_size_um") if hasattr(root, "attrs") else None
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

    reference, reference_source = _resolve_reference_distance(reference_distance_um, root, group)
    print(
        f"  Weight w = (1 - d/D)^{decay_power:g}, d measured from the pair's contact or "
        f"facing interface"
    )
    if np.isfinite(reference):
        print(f"  Reference distance D: {reference:g} um from {reference_source}")
    else:
        print(f"  Reference distance D: {reference_source}, resolved after the pass")

    volumes_frame = None
    if volumes_csv:
        volumes_frame = _load_volumes_csv(volumes_csv)
        print(f"Loaded {len(volumes_frame)} pairs from {volumes_csv}")

    want_coords = per_point_csv is not None
    batches = _batch_pairs(table, int(pair_batch))
    print(
        f"Scoring {n_rows} pairs in {len(batches)} batches of up to {pair_batch} with "
        f"{n_jobs} worker(s)"
    )

    n_footprint = np.zeros(n_rows, dtype=np.int64)
    n_interface = np.zeros(n_rows, dtype=np.int64)
    dist_max = np.full(n_rows, np.nan, dtype=np.float64)
    dist_mean = np.full(n_rows, np.nan, dtype=np.float64)
    beyond = np.zeros(n_rows, dtype=np.int64)
    collected = [{'rows': [], 'dist': [], 'coords': []} for _ in spot_specs]

    init_args = (interscellar_zarr, tuple(spot_specs), (vz, vy, vx), reference, want_coords)
    for results in _run_pairs(batches, n_jobs, init_args, verbose):
        for row, footprint, interface_voxels, d_max, d_mean, n_beyond, spots in results:
            n_footprint[row] = footprint
            n_interface[row] = interface_voxels
            dist_max[row] = d_max
            dist_mean[row] = d_mean
            beyond[row] = n_beyond
            for index, distances, coords in spots:
                collected[index]['rows'].append(np.full(distances.size, row, dtype=np.int64))
                collected[index]['dist'].append(distances)
                if coords is not None:
                    collected[index]['coords'].append(coords)

    def _join(chunks, dtype, width=None):
        if not chunks:
            return np.empty((0, width) if width else 0, dtype=dtype)
        return np.concatenate(chunks)

    spot_data = [
        (
            _join(entry['rows'], np.int64),
            _join(entry['dist'], np.float32),
            _join(entry['coords'], np.int32, width=3) if entry['coords'] else None,
        )
        for entry in collected
    ]

    observed_max = float(np.nanmax(dist_max)) if np.isfinite(dist_max).any() else np.nan
    if not np.isfinite(reference):
        if not np.isfinite(observed_max) or observed_max <= 0:
            raise ValueError(
                "reference_distance_um='auto' needs at least one pair with a measurable "
                "interface distance, but none was found."
            )
        reference = observed_max
        print(f"  Resolved D = {reference:.3f} um (deepest interface distance in the data)")

    interface_kind = table['interface_kind']
    table_out = pd.DataFrame({
        'pair_id': table['pair_id'],
        'cell_a_id': (np.zeros(n_rows, dtype=np.int64) if table['cell_a_id'] is None
                      else table['cell_a_id'].astype(np.int64)),
        'cell_b_id': (np.zeros(n_rows, dtype=np.int64) if table['cell_b_id'] is None
                      else table['cell_b_id'].astype(np.int64)),
        'interface_kind': (
            ['unknown'] * n_rows if interface_kind is None
            else [_INTERFACE_KIND_NAME.get(int(v), 'unknown') for v in interface_kind]
        ),
        'n_voxels': n_footprint,
        'interscellar_volume_um3': n_footprint * voxel_volume_um3,
        'interface_voxels': n_interface,
        'interface_dist_max_um': dist_max,
        'interface_dist_mean_um': dist_mean,
        'voxels_beyond_reference': beyond,
        'fraction_beyond_reference': np.where(
            n_footprint > 0, beyond / np.maximum(n_footprint, 1), np.nan
        ),
        'reference_distance_um': reference,
        'decay_power': decay_power,
    })

    for column, key in (
        ('territory_a_voxels', 'n_territory_a'),
        ('corridor_voxels', 'n_corridor'),
        ('territory_b_voxels', 'n_territory_b'),
        ('interface_a_voxels', 'n_interface_a'),
        ('interface_b_voxels', 'n_interface_b'),
    ):
        stored = table[key]
        table_out[column] = -1 if stored is None else stored.astype(np.int64)

    table_out['min_surface_separation_um'] = (
        np.nan if table['min_surface_separation_um'] is None
        else table['min_surface_separation_um'].astype(np.float64)
    )
    centroid = table['centroid_zyx_um']
    for axis, index in (('z', 0), ('y', 1), ('x', 2)):
        table_out[f'centroid_{axis}_um'] = (
            np.nan if centroid is None else centroid[:, index].astype(np.float64)
        )

    shared = table['shared_voxels']
    has_preview = hasattr(root, "keys") and "overlap_count" in root
    if shared is not None and (has_preview or int(np.asarray(shared).max(initial=0)) > 0):
        shared = np.asarray(shared).astype(np.int64)
        table_out['shared_voxels'] = shared
        table_out['shared_fraction'] = np.where(
            n_footprint > 0, shared / np.maximum(n_footprint, 1), np.nan
        )
    else:
        table_out['shared_voxels'] = -1
        table_out['shared_fraction'] = np.nan
        print(
            "  Note: this store records no per-pair shared-voxel count (it was built "
            "without the overlap preview), so shared_voxels is reported as -1. Scores "
            "are unaffected: each pair is scored from its own complete footprint."
        )

    if volumes_frame is not None:
        table_out = table_out.merge(volumes_frame, on='pair_id', how='left')
        missing_from_archive = sorted(
            set(volumes_frame['pair_id'].tolist()) - set(table['pair_id'].tolist())
        )
        if missing_from_archive:
            print(
                f"  Note: {len(missing_from_archive)} pairs in {volumes_csv} have no "
                f"archived footprint and were not scored (examples: "
                f"{missing_from_archive[:10]}); rejected pairs are listed in the "
                f"rejected-pairs CSV"
            )
        absent = int(table_out['volumes_csv_volume_um3'].isna().sum())
        if absent:
            print(f"  Note: {absent} archived pairs are absent from {volumes_csv}")
    else:
        table_out['volumes_csv_volume_um3'] = np.nan

    volume_um3 = table_out['interscellar_volume_um3'].to_numpy()
    per_biomarker, point_frames = [], []

    for (name, _, _), (rows, spot_distances, spot_coords) in zip(spot_specs, spot_data):
        distances = spot_distances.astype(np.float64)
        weights = score_from_distance(distances, reference, decay_power)
        n_spots = np.bincount(rows, minlength=n_rows)
        score_sum = np.bincount(rows, weights=weights, minlength=n_rows)
        dist_sum = np.bincount(rows, weights=distances, minlength=n_rows)
        past = np.bincount(rows, weights=(distances > reference).astype(np.float64),
                           minlength=n_rows)

        median = np.full(n_rows, np.nan, dtype=np.float64)
        if rows.size:
            grouped = pd.Series(distances).groupby(rows).median()
            median[grouped.index.to_numpy()] = grouped.to_numpy()

        scorable = n_interface > 0
        counted = n_spots.astype(np.int64)
        frame = table_out.copy()
        frame['biomarker'] = name
        frame['n_spots'] = counted
        frame['spots_beyond_reference'] = past.astype(np.int64)
        frame['score_sum'] = np.where(scorable, score_sum, np.nan)
        frame['score_mean'] = np.where(
            scorable & (counted > 0), score_sum / np.maximum(counted, 1), np.nan
        )
        frame['score_per_um3'] = np.where(
            scorable & (volume_um3 > 0), score_sum / np.where(volume_um3 > 0, volume_um3, 1.0),
            np.nan,
        )
        frame['mean_dist_um'] = np.where(counted > 0, dist_sum / np.maximum(counted, 1), np.nan)
        frame['median_dist_um'] = median
        per_biomarker.append(frame)
        print(f"  {name}: {int(rows.size)} spots inside an interscellar volume")

        if want_coords and rows.size and spot_coords is not None:
            point_frames.append(pd.DataFrame({
                'pair_id': table['pair_id'][rows],
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

    empty = int((n_footprint == 0).sum())
    if empty:
        print(f"  Warning: {empty} archived pairs decoded to an empty footprint")
    no_interface = int(((n_interface == 0) & (n_footprint > 0)).sum())
    if no_interface:
        print(
            f"  Warning: {no_interface} pairs have no archived interface voxels, so their "
            f"spots could not be weighted (scores are NaN). Rebuild those pairs with "
            f"compute_interscellar_volumes_3d_adaptive.py to store their interface."
        )
    if np.isfinite(observed_max):
        deep = int((dist_max > reference).sum())
        print(
            f"  Interface depth: deepest voxel {observed_max:.2f} um from its interface; "
            f"{deep} of {n_rows} volumes reach past D = {reference:.2f} um"
        )
        if deep:
            clipped = float(np.nansum(beyond)) / max(1.0, float(n_footprint.sum()))
            print(
                f"  Note: {100 * clipped:.1f}% of all footprint voxels lie beyond D and "
                f"score 0. Raise --reference-distance-um (or pass 'auto' for "
                f"{observed_max:.2f}) if that clipping is not what you want."
            )
    shared_reported = table_out['shared_voxels'].to_numpy()
    overlapping = int((shared_reported > 0).sum())
    if overlapping:
        mean_fraction = float(np.nanmean(
            table_out.loc[shared_reported > 0, 'shared_fraction']
        ))
        print(
            f"  Note: {overlapping} pairs share voxels with another pair (mean "
            f"{100 * mean_fraction:.1f}% of the footprint). Each pair was scored from its "
            f"complete footprint, so those voxels count once for every pair claiming them."
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
        f"Wrote {len(long_table)} rows ({len(table_out)} pairs x "
        f"{len(spot_specs)} biomarkers) to: {output_csv}"
    )
    return output_csv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interscellar score for point-form biomarkers. Reads every pair's complete "
            "footprint from the lossless 'pair_volumes' archive written by "
            "compute_interscellar_volumes_3d_adaptive.py, so a spot voxel claimed by "
            "several interscellar volumes is scored independently in each of them. Each "
            "spot is weighted by w = (1 - d/D)^power, where d is its distance from that "
            "pair's direct-contact or facing interface and D is the reference distance at "
            "which the weight reaches zero. Emits a long-format CSV, one row per pair_id "
            "and biomarker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interscellar-zarr", required=True,
        help=(
            "Interscellar volumes zarr from the adaptive volume build, holding the "
            "'pair_volumes' archive of per-pair footprints and interfaces."
        ),
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
            "Per-pair volumes CSV from the volume build. Optional; the pair archive "
            "already supplies the roster and the cell IDs, so this only cross-checks "
            "the volume figures."
        ),
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Output CSV. Defaults to <zarr_dir>/<stem>_scores.csv",
    )
    parser.add_argument(
        "--per-point-csv", default=None,
        help="Optional per-spot dump: pair_id, biomarker, z, y, x, dist_um, weight.",
    )
    parser.add_argument(
        "--reference-distance-um", default=None, metavar="UM",
        help=(
            "Distance D at which the weight reaches zero, in micrometers, or 'auto' for "
            "the deepest interface distance in the data. Defaults to the store's "
            "max_distance_um attribute."
        ),
    )
    parser.add_argument(
        "--decay-power", type=float, default=1.0,
        help="Exponent on the ramp: 1.0 is linear, >1 concentrates weight at the interface.",
    )
    parser.add_argument(
        "--voxel-size-um", nargs=3, type=float, default=None, metavar=("Z", "Y", "X"),
        help=(
            "Voxel size in micrometers. Defaults to the zarr's OME-NGFF scale, then its "
            f"voxel_size_um attribute, then {DEFAULT_VOXEL_SIZE_UM}."
        ),
    )
    parser.add_argument(
        "--spot-key", default=None, action="append", metavar="NAME=KEY",
        help="Array key inside a named spot zarr (auto-detected if omitted).",
    )
    parser.add_argument(
        "--pair-batch", type=int, default=32,
        help="Pairs per work item. Larger batches cut task overhead on small volumes.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker processes. Pairs are distributed across them.",
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
            reference_distance_um=args.reference_distance_um,
            spot_keys=spot_keys,
            pair_batch=args.pair_batch,
            per_point_csv=args.per_point_csv,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

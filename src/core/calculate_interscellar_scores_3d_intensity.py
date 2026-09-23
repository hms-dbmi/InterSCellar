import argparse
import os
import sys
from sys import version_info as _py_version
import unicodedata
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy
import zarr

_WORKER_CTX: Dict[str, Any] = {}


def stats(array):
    """
    Statistics on non-zero entries of ``array`` (masked intensities that are exactly 0 are dropped).

    If every masked voxel is zero, returns NaNs so downstream CSV still gets a full row.
    """
    arr = array[array != 0]
    if arr.size == 0:
        nan = float("nan")
        return map(float, (nan,) * 13)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    p5, p25, p50, p75, p95 = (float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = float(scipy.stats.skew(arr.ravel()))
        ku = float(scipy.stats.kurtosis(arr.ravel()))
    return map(
        float,
        (
            mean,
            std,
            min_val,
            max_val,
            float(np.sum(arr)),
            max_val - min_val,
            p5,
            p25,
            p50,
            p75,
            p95,
            sk,
            ku,
        ),
    )


def _zarr_child_keys(obj: Any) -> Optional[List[str]]:
    keys_fn = getattr(obj, "keys", None)
    if keys_fn is None:
        return None
    try:
        return list(keys_fn())
    except Exception:
        return None


def _normalize_cli_path(path_value: str) -> str:
    path_value = unicodedata.normalize("NFKC", path_value).strip()
    path_value = (
        path_value.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    path_value = path_value.replace('\\"', '"').replace("\\'", "'")
    quote_chars = ('"', "'")
    while len(path_value) >= 2 and path_value[0] in quote_chars and path_value[-1] in quote_chars:
        path_value = path_value[1:-1].strip()
    path_value = os.path.expanduser(path_value)
    return os.path.normpath(path_value)


def _candidate_input_paths(path_hint: str, script_dir: str) -> List[Path]:
    normalized = _normalize_cli_path(path_hint)
    base = Path(normalized)
    if base.is_absolute():
        return [base]
    return [
        base,
        Path(script_dir) / base,
        Path(script_dir).parent / base,
    ]


def _is_zarr_store_dir(path: Path) -> bool:
    return path.is_dir() and (
        path.name.endswith(".zarr")
        or (path / "zarr.json").exists()
        or (path / ".zgroup").exists()
    )


def _find_zarr_store(path_hint: str, script_dir: str) -> str:
    candidates = _candidate_input_paths(path_hint, script_dir)
    existing = [p for p in candidates if p.exists()]
    p = existing[0] if existing else candidates[0]
    if _is_zarr_store_dir(p):
        return str(p.resolve())

    cur = p if p.exists() else p.parent
    while True:
        if cur.name.endswith(".zarr") and cur.exists():
            return str(cur.resolve())
        if _is_zarr_store_dir(cur):
            return str(cur.resolve())
        if cur.parent == cur:
            break
        cur = cur.parent
    return str(p.resolve(strict=False))


def _node_shape_ndim(node: Any) -> Tuple[Tuple[int, ...], int]:
    if not hasattr(node, "shape"):
        raise TypeError("not an array-like zarr node")
    sh = tuple(int(x) for x in node.shape)
    return sh, len(sh)


def _to_spatial_shape_zyx(arr: Any) -> Tuple[int, int, int]:
    sh, nd = _node_shape_ndim(arr)
    if nd == 5:
        return (sh[2], sh[3], sh[4])
    if nd == 4:
        return (sh[1], sh[2], sh[3])
    if nd == 3:
        return (sh[0], sh[1], sh[2])
    raise ValueError(f"Expected 3D-5D array, got ndim={nd} shape={sh}")


def _raw_node_to_czyx_shape(arr: Any) -> Tuple[int, int, int, int]:
    sh, nd = _node_shape_ndim(arr)
    if nd == 5:
        return (sh[1], sh[2], sh[3], sh[4])
    if nd == 4:
        return (sh[0], sh[1], sh[2], sh[3])
    if nd == 3:
        return (1, sh[0], sh[1], sh[2])
    raise ValueError(f"Raw expression array must be 3D/4D/5D. Got shape={sh}")


def _downsampled_spatial_shape(shape_zyx: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
    factor = 2**level
    if factor <= 1:
        return shape_zyx
    return tuple((int(v) + factor - 1) // factor for v in shape_zyx)


def _effective_raw_shape_czyx(raw_arr: Any, downsample_level: int) -> Tuple[int, int, int, int]:
    base_shape = _raw_node_to_czyx_shape(raw_arr)
    if downsample_level <= 0:
        return base_shape
    ds_shape = _downsampled_spatial_shape(
        (int(base_shape[1]), int(base_shape[2]), int(base_shape[3])), downsample_level
    )
    return (int(base_shape[0]), ds_shape[0], ds_shape[1], ds_shape[2])


def _resolve_group_path(group_root: Any, key_path: str) -> Any:
    node = group_root
    for part in [p for p in key_path.split("/") if p]:
        node = node[part]
    return node


def _read_label_slice_xy(label_arr: Any, z_idx: int) -> np.ndarray:
    _, nd = _node_shape_ndim(label_arr)
    if nd == 5:
        return np.asarray(label_arr[0, 0, z_idx])
    if nd == 4:
        return np.asarray(label_arr[0, z_idx])
    if nd == 3:
        return np.asarray(label_arr[z_idx])
    raise ValueError(f"Label array must be 3D/4D/5D. Got ndim={nd}")


def _scan_label_tight_boxes(label_arr: Any) -> Dict[int, Dict[str, int]]:
    """Tight per-label (Z,Y,X) bounding boxes by scanning one Z slice at a time."""
    z_size, _, _ = _to_spatial_shape_zyx(label_arr)
    boxes: Dict[int, Dict[str, int]] = {}
    for z_idx in range(z_size):
        label_slice = _read_label_slice_xy(label_arr, z_idx)
        present_labels = np.unique(label_slice)
        present_labels = present_labels[present_labels > 0]
        for label_id in present_labels:
            ys, xs = np.where(label_slice == label_id)
            if ys.size == 0:
                continue
            lid = int(label_id)
            y_min = int(ys.min())
            y_max = int(ys.max()) + 1
            x_min = int(xs.min())
            x_max = int(xs.max()) + 1
            if lid not in boxes:
                boxes[lid] = {
                    "z0": z_idx,
                    "z1": z_idx + 1,
                    "y0": y_min,
                    "y1": y_max,
                    "x0": x_min,
                    "x1": x_max,
                }
                continue
            box = boxes[lid]
            box["z0"] = min(box["z0"], z_idx)
            box["z1"] = max(box["z1"], z_idx + 1)
            box["y0"] = min(box["y0"], y_min)
            box["y1"] = max(box["y1"], y_max)
            box["x0"] = min(box["x0"], x_min)
            box["x1"] = max(box["x1"], x_max)
    return boxes


def _read_label_plane_bboxed(
    label_arr: Any, z_idx: int, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    return _read_label_slice_xy(label_arr, z_idx)[y0:y1, x0:x1]


def _read_raw_plane_czyx(
    raw_arr: Any,
    z_idx: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    downsample_level: int,
    *,
    labels_on_full_raw_grid: bool,
    z_origin: int,
) -> np.ndarray:
    """
    One analysis-grid Z plane as (C, Y, X), reading only the requested XY window from zarr.
    """
    _, nd = _node_shape_ndim(raw_arr)
    factor = 2**downsample_level
    full_shape = _raw_node_to_czyx_shape(raw_arr)
    if labels_on_full_raw_grid:
        fz0 = z_origin + z_idx * factor
        fz1 = min(fz0 + factor, full_shape[1])
        fy0, fy1 = y0, min(y1, full_shape[2])
        fx0, fx1 = x0, min(x1, full_shape[3])
        z_sl = slice(fz0, fz1, factor)
        y_sl = slice(fy0, fy1, factor)
        x_sl = slice(fx0, fx1, factor)
    else:
        fz0 = z_idx * factor
        fz1 = min((z_idx + 1) * factor, full_shape[1])
        fy0 = y0 * factor
        fy1 = min(y1 * factor, full_shape[2])
        fx0 = x0 * factor
        fx1 = min(x1 * factor, full_shape[3])
        z_sl = slice(fz0, fz1, factor)
        y_sl = slice(fy0, fy1, factor)
        x_sl = slice(fx0, fx1, factor)

    if nd == 5:
        plane = np.asarray(raw_arr[0, :, z_sl, y_sl, x_sl])
    elif nd == 4:
        plane = np.asarray(raw_arr[:, z_sl, y_sl, x_sl])
    elif nd == 3:
        plane = np.asarray(raw_arr[z_sl, y_sl, x_sl])[np.newaxis, ...]
    else:
        raise ValueError(f"Raw expression array must be 3D/4D/5D. Got ndim={nd}")

    if plane.ndim == 4 and plane.shape[1] == 1:
        plane = plane[:, 0, :, :]
    return plane


def _gather_masked_intensities_zslab(
    seg_arr: Any,
    raw_arr: Any,
    obj_id: int,
    box: Dict[str, int],
    channel_count: int,
    raw_downsample_level: int,
    labels_on_full_raw_grid: bool,
) -> Tuple[int, List[np.ndarray]]:
    """
    Collect per-channel 1D masked intensities by visiting one (or factor) Z plane(s) at a time.

    Peak memory is O(bbox XY × factor × C), not the full 3D tight bbox.
    """
    z0, z1 = box["z0"], box["z1"]
    y0, y1 = box["y0"], box["y1"]
    x0, x1 = box["x0"], box["x1"]
    factor = 2**raw_downsample_level if raw_downsample_level > 0 else 1
    parts: List[List[np.ndarray]] = [[] for _ in range(channel_count)]
    voxel_count = 0

    if raw_downsample_level > 0 and labels_on_full_raw_grid:
        n_analysis_z = (z1 - z0 + factor - 1) // factor
        for za in range(n_analysis_z):
            z_block_start = z0 + za * factor
            z_block_end = min(z_block_start + factor, z1)
            planes: List[np.ndarray] = []
            for z_full in range(z_block_start, z_block_end):
                lab = _read_label_plane_bboxed(seg_arr, z_full, y0, y1, x0, x1)
                planes.append(lab == obj_id)
            sub_mask = np.stack(planes, axis=0)
            if factor > 1:
                plane_mask = _block_reduce_mask_any(sub_mask, factor)
            else:
                plane_mask = sub_mask[0]
            if not np.any(plane_mask):
                continue
            voxel_count += int(np.count_nonzero(plane_mask))
            raw_plane = _read_raw_plane_czyx(
                raw_arr,
                za,
                y0,
                y1,
                x0,
                x1,
                raw_downsample_level,
                labels_on_full_raw_grid=True,
                z_origin=z0,
            )
            if raw_plane.shape[1:] != plane_mask.shape:
                raise RuntimeError(
                    f"Raw plane {raw_plane.shape[1:]} != mask plane {plane_mask.shape} "
                    f"(object {obj_id}, analysis z={za})."
                )
            for ch in range(channel_count):
                parts[ch].append(
                    np.asarray(raw_plane[ch][plane_mask], dtype=np.float64)
                )
    else:
        for z_idx in range(z0, z1):
            lab = _read_label_plane_bboxed(seg_arr, z_idx, y0, y1, x0, x1)
            plane_mask = lab == obj_id
            if not np.any(plane_mask):
                continue
            voxel_count += int(np.count_nonzero(plane_mask))
            raw_plane = _read_raw_plane_czyx(
                raw_arr,
                z_idx,
                y0,
                y1,
                x0,
                x1,
                raw_downsample_level,
                labels_on_full_raw_grid=False,
                z_origin=0,
            )
            if raw_plane.shape[1:] != plane_mask.shape:
                raise RuntimeError(
                    f"Raw plane {raw_plane.shape[1:]} != mask plane {plane_mask.shape} "
                    f"(object {obj_id}, z={z_idx})."
                )
            for ch in range(channel_count):
                parts[ch].append(
                    np.asarray(raw_plane[ch][plane_mask], dtype=np.float64)
                )

    merged = [
        np.concatenate(ch_parts) if ch_parts else np.empty(0, dtype=np.float64)
        for ch_parts in parts
    ]
    return voxel_count, merged


def _chunk_keys_for_spatial_box(
    box: Dict[str, int],
    chunk_shape: Tuple[int, ...],
    label_ndim: int,
) -> List[str]:
    """Zarr chunk keys (dot-separated indices) that intersect the spatial bbox."""
    if label_ndim == 5:
        starts = [0, 0, box["z0"], box["y0"], box["x0"]]
        ends = [1, 1, box["z1"], box["y1"], box["x1"]]
    elif label_ndim == 4:
        starts = [0, box["z0"], box["y0"], box["x0"]]
        ends = [1, box["z1"], box["y1"], box["x1"]]
    elif label_ndim == 3:
        starts = [box["z0"], box["y0"], box["x0"]]
        ends = [box["z1"], box["y1"], box["x1"]]
    else:
        raise ValueError(f"Unsupported label ndim={label_ndim}")

    ranges: List[range] = []
    for st, en, cs in zip(starts, ends, chunk_shape):
        cs = max(1, int(cs))
        i0 = int(st) // cs
        i1 = (max(int(en), int(st) + 1) - 1) // cs
        ranges.append(range(i0, i1 + 1))
    return [".".join(map(str, coord)) for coord in product(*ranges)]


def _block_reduce_mask_any(mask: np.ndarray, factor: int) -> np.ndarray:
    """OR-pool a boolean 3D mask by non-overlapping factor×factor×factor blocks (pad with False)."""
    if factor <= 1:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    z, y, x = mask.shape
    pz = (factor - z % factor) % factor
    py = (factor - y % factor) % factor
    px = (factor - x % factor) % factor
    if pz or py or px:
        mask = np.pad(mask, ((0, pz), (0, py), (0, px)), mode="constant", constant_values=False)
    z, y, x = mask.shape
    return mask.reshape(z // factor, factor, y // factor, factor, x // factor, factor).any(axis=(1, 3, 5))


def _init_worker(
    segmentation_zarr: str,
    raw_expression_zarr: str,
    seg_key_path: str,
    raw_key_path: str,
    channel_count: int,
    raw_downsample_level: int,
    labels_on_full_raw_grid: bool,
) -> None:
    seg_zarr = zarr.open(segmentation_zarr, mode="r")
    raw_zarr = zarr.open(raw_expression_zarr, mode="r")
    seg_arr = _resolve_group_path(seg_zarr, seg_key_path) if seg_key_path != "<root>" else seg_zarr
    raw_arr = _resolve_group_path(raw_zarr, raw_key_path) if raw_key_path != "<root>" else raw_zarr
    _WORKER_CTX["seg_arr"] = seg_arr
    _WORKER_CTX["raw_arr"] = raw_arr
    _WORKER_CTX["channel_count"] = channel_count
    _WORKER_CTX["raw_downsample_level"] = int(raw_downsample_level)
    _WORKER_CTX["labels_on_full_raw_grid"] = bool(labels_on_full_raw_grid)


def _compute_single_object_stats(task: Tuple[int, Dict[str, int], List[str]]) -> List[Any]:
    obj_id, box, chunk_keys = task
    seg_arr = _WORKER_CTX["seg_arr"]
    raw_arr = _WORKER_CTX["raw_arr"]
    channel_count = _WORKER_CTX["channel_count"]
    raw_downsample_level = int(_WORKER_CTX.get("raw_downsample_level", 0))
    labels_on_full_raw_grid = bool(_WORKER_CTX.get("labels_on_full_raw_grid", False))

    voxel_count, channel_values = _gather_masked_intensities_zslab(
        seg_arr,
        raw_arr,
        obj_id,
        box,
        channel_count,
        raw_downsample_level,
        labels_on_full_raw_grid,
    )
    if voxel_count == 0:
        raise RuntimeError(f"Object {obj_id} had empty mask inside its tight bounding box.")

    out = [int(obj_id), voxel_count, chunk_keys]
    for values in channel_values:
        stats_out = stats(values)
        out.extend([float(x) for x in list(stats_out)])
    return out


def _append_row_to_csv(output_csv: str, row: List[Any], columns: List[str]) -> None:
    write_header = not os.path.exists(output_csv)
    pd.DataFrame([row], columns=columns).to_csv(
        output_csv,
        mode="a",
        index=False,
        header=write_header,
    )


def _iter_segmentation_candidates(seg: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    gkeys = _zarr_child_keys(seg)
    if gkeys is None:
        return out

    def add(name: str, node: Any) -> None:
        if hasattr(node, "ndim") and node.ndim >= 3:
            out.append((name, node))

    if "0" in gkeys:
        node = seg["0"]
        sub = _zarr_child_keys(node)
        if sub is not None and "0" in sub:
            add("0/0", node["0"])
        elif hasattr(node, "ndim") and node.ndim >= 3:
            add("0", node)
    if "labels" in gkeys:
        add("labels", seg["labels"])
    if "interscellar_meshes" in gkeys:
        add("interscellar_meshes", seg["interscellar_meshes"])

    seen = {id(n) for _, n in out}
    for key in sorted(gkeys):
        node = seg[key]
        if id(node) in seen:
            continue
        if hasattr(node, "ndim") and node.ndim >= 3:
            add(key, node)
    return out


def _iter_raw_expression_candidates(raw_root: Any) -> List[Tuple[str, Any]]:
    keys = _zarr_child_keys(raw_root)
    if keys is None:
        if hasattr(raw_root, "ndim") and raw_root.ndim >= 3:
            return [("<root>", raw_root)]
        return []

    out: List[Tuple[str, Any]] = []
    seen: Set[int] = set()

    def add_candidate(name: str, node: Any) -> None:
        if not (hasattr(node, "ndim") and node.ndim >= 3):
            return
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        out.append((name, node))

    for key in sorted(keys):
        node = raw_root[key]
        if hasattr(node, "ndim"):
            add_candidate(key, node)
            continue
        subkeys = _zarr_child_keys(node)
        if subkeys is None:
            continue
        for sk in sorted(subkeys):
            add_candidate(f"{key}/{sk}", node[sk])

    return out


def _infer_resolution_level_from_key(key_name: str) -> Optional[int]:
    if key_name == "<root>":
        return 0
    parts = [p for p in key_name.split("/") if p]
    for part in reversed(parts):
        try:
            return int(part)
        except ValueError:
            continue
    return None


def _pick_label_array_matching_spatial(seg: zarr.Group, expected_zyx: Tuple[int, int, int]) -> Tuple[str, Any]:
    candidates = _iter_segmentation_candidates(seg)
    if not candidates:
        gk = _zarr_child_keys(seg) or []
        raise RuntimeError(f"No 3D+ arrays found in segmentation zarr. Keys: {gk}")

    for name, node in candidates:
        try:
            if _to_spatial_shape_zyx(node) == expected_zyx:
                return name, node
        except ValueError:
            continue
    raise ValueError(f"No segmentation array matches raw spatial shape {expected_zyx}.")


def _pick_label_volume(
    seg: Any,
    data_raw: Any,
    raw_strided_level: int,
    expected_zyx: Tuple[int, int, int],
) -> Tuple[str, Any, bool]:
    """
    Pick a segmentation array compatible with the raw analysis grid.

    Returns (key_path, array, labels_on_full_raw_grid).

    Prefer labels whose (Z,Y,X) equals the effective (downsampled) raw shape. If that fails
    but raw uses strided downsampling on a single full-res array, fall back to labels on the
    **full** raw grid and OR-pool masks per 2^L block to align with strided intensities.
    """
    try:
        name, node = _pick_label_array_matching_spatial(seg, expected_zyx)
        return name, node, False
    except ValueError:
        if raw_strided_level <= 0:
            raise
    full_zyx = tuple(int(v) for v in _raw_node_to_czyx_shape(data_raw)[1:])
    try:
        name, node = _pick_label_array_matching_spatial(seg, full_zyx)
    except ValueError as exc:
        raise ValueError(
            f"No segmentation volume matches effective raw grid {expected_zyx} or full raw grid "
            f"{full_zyx}. Use labels on the analysis resolution, or full-resolution labels when "
            "using --raw-resolution-level > 0 with a single-array raw store."
        ) from exc
    print(
        f"Segmentation matches full raw shape {full_zyx}; using block-OR mask downsampling "
        f"to align with strided raw (level {raw_strided_level})."
    )
    return name, node, True


def _resolve_raw_array(raw_zarr: Any, resolution_level: int) -> Tuple[str, Any, int]:
    """
    Returns (raw_zarr_key_path, array_node, strided_downsample_level).

    ``strided_downsample_level`` is 0 when reading a native multiscale level array.
    If the store is a single array and ``resolution_level > 0``, returns that array
    with ``strided_downsample_level == resolution_level`` (same convention as feature_extraction_3d).
    """
    candidates = _iter_raw_expression_candidates(raw_zarr)
    if not candidates:
        raise RuntimeError("Could not find any 3D+ raw expression arrays in the provided zarr store.")

    if resolution_level < 0:
        raise ValueError("--raw-resolution-level must be >= 0.")

    level_matches: List[Tuple[str, Any]] = []
    level_hints: List[Tuple[str, Optional[int]]] = []
    for name, node in candidates:
        inferred = _infer_resolution_level_from_key(name)
        level_hints.append((name, inferred))
        if inferred == resolution_level:
            level_matches.append((name, node))

    if level_matches:
        level_matches.sort(key=lambda x: x[0])
        key, node = level_matches[0]
        return key, node, 0

    if len(candidates) == 1 and candidates[0][0] == "<root>":
        _, node = candidates[0]
        if resolution_level == 0:
            return "<root>", node, 0
        print(
            "Raw store has a single array only; applying strided downsampling for "
            f"--raw-resolution-level {resolution_level} (same as feature_extraction_3d)."
        )
        return "<root>", node, resolution_level

    lines = [
        f"Requested --raw-resolution-level {resolution_level} was not found.",
        "  Available raw arrays (key -> inferred level):",
    ]
    for name, level in level_hints:
        lines.append(f"    {name} -> {level}")
    raise ValueError("\n".join(lines))


def sub_volume_analysis(
    segmentation_zarr: str,
    raw_expression_zarr: str,
    output_csv: str,
    raw_resolution_level: int = 0,
    object_id_column: str = "pair_id",
    n_jobs: int = 1,
):
    """
    Per-object subvolume statistics: segmentation labels vs raw intensity at a chosen pyramid level.

    The raw array at ``raw_resolution_level`` sets the analysis grid. Segmentation is chosen
    to match that effective (Z, Y, X) when possible. If raw uses strided downsampling on a
    single full-res array and labels only exist on the full grid, labels are matched to the
    full raw shape and each object mask is block-OR pooled to the strided grid.

    Parameters
    ----------
    segmentation_zarr:
        Path to the segmentation OME-Zarr store.
    raw_expression_zarr:
        Path to the raw expression OME-Zarr store (multiscale keys inferred from path names).
    output_csv:
        CSV path for results and resume checkpoints.
    raw_resolution_level:
        Pyramid level index (0 = full resolution). For multiscale stores, the matching
        array key must exist. For a single-array store, level > 0 uses strided downsampling
        on that array (see feature_extraction_3d).
    object_id_column:
        Name of the ID column in the CSV.
    n_jobs:
        Process pool size for parallel objects.

    Returns
    -------
    None; appends rows to ``output_csv``.
    """
    seg_zarr = zarr.open(segmentation_zarr, mode="r")
    raw_zarr = zarr.open(raw_expression_zarr, mode="r")
    raw_key_path, data_raw, raw_strided_level = _resolve_raw_array(raw_zarr, raw_resolution_level)
    raw_shape_czyx = _effective_raw_shape_czyx(data_raw, raw_strided_level)
    expected_zyx = tuple(int(v) for v in raw_shape_czyx[1:])
    seg_key_used, data, labels_on_full_raw_grid = _pick_label_volume(
        seg_zarr, data_raw, raw_strided_level, expected_zyx
    )
    channel_count = int(raw_shape_czyx[0])
    print(f"Using segmentation volume '{seg_key_used}' and raw zarr key '{raw_key_path}'.")
    print(f"Requested --raw-resolution-level: {raw_resolution_level}")
    if raw_strided_level > 0:
        print(f"Strided read factor 2^{raw_strided_level} on single-array raw store.")
    print(f"Effective raw shape (C, Z, Y, X): {raw_shape_czyx}")
    print(f"Output CSV (append / resume): {output_csv}")

    # Read the csv file and get the already worked IDs
    worked_ids = set()
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        df[object_id_column] = df[object_id_column].astype(int)
        worked_ids = set(df[object_id_column].tolist())

    columns = [object_id_column, "voxels", "chunk_keys"]
    for channel in range(channel_count):
        columns.append(str(channel) + "_mean")
        columns.append(str(channel) + "_std")
        columns.append(str(channel) + "_min")
        columns.append(str(channel) + "_max")
        columns.append(str(channel) + "_sum")
        columns.append(str(channel) + "_range")
        columns.append(str(channel) + "_p5")
        columns.append(str(channel) + "_p25")
        columns.append(str(channel) + "_p50")
        columns.append(str(channel) + "_p75")
        columns.append(str(channel) + "_p95")
        columns.append(str(channel) + "_skew")
        columns.append(str(channel) + "_kurtosis")

    print("Scanning tight per-object bounding boxes (one Z slice at a time)...")
    tight_boxes = _scan_label_tight_boxes(data)
    chunk_shape = data.chunks
    label_ndim = int(data.ndim)

    pending_tasks: List[Tuple[int, Dict[str, int], List[str]]] = []
    for obj_id, box in tight_boxes.items():
        if int(obj_id) in worked_ids:
            continue
        chunk_keys = _chunk_keys_for_spatial_box(box, chunk_shape, label_ndim)
        pending_tasks.append((int(obj_id), box, chunk_keys))
    total = len(pending_tasks)
    if total == 0:
        print("No new objects to process; output is already up to date.")
        return

    # Keep raw/seg paths for worker init and resolve selected keys.
    seg_key_path = seg_key_used
    workers = max(1, int(n_jobs))
    print(f"Processing {total} objects with {workers} worker(s).")

    if workers == 1:
        _init_worker(
            segmentation_zarr,
            raw_expression_zarr,
            seg_key_path,
            raw_key_path,
            channel_count,
            raw_strided_level,
            labels_on_full_raw_grid,
        )
        done = 0
        for task in pending_tasks:
            row = _compute_single_object_stats(task)
            _append_row_to_csv(output_csv, row, columns)
            done += 1
            print("\r", f"{done}/{total}", end="")
    else:
        pool_kwargs: Dict[str, Any] = {
            "max_workers": workers,
            "initializer": _init_worker,
            "initargs": (
                segmentation_zarr,
                raw_expression_zarr,
                seg_key_path,
                raw_key_path,
                channel_count,
                raw_strided_level,
                labels_on_full_raw_grid,
            ),
        }
        if _py_version >= (3, 11):
            pool_kwargs["max_tasks_per_child"] = 1
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {
                ex.submit(_compute_single_object_stats, task): task[0]
                for task in pending_tasks
            }
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                _append_row_to_csv(output_csv, row, columns)
                done += 1
                print("\r", f"{done}/{total}", end="")
    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-object subvolume statistics from a segmentation zarr and raw "
            "expression zarr with the same CLI I/O contract as feature_extraction_3d.py."
        )
    )
    parser.add_argument(
        "--segmentation-zarr",
        required=True,
        help="Path to segmentation zarr (interscellar or cell-only labels).",
    )
    parser.add_argument(
        "--raw-expression-zarr",
        required=True,
        help="Path to raw expression OME-zarr (3D/4D/5D).",
    )
    parser.add_argument(
        "--raw-resolution-level",
        type=int,
        default=0,
        help=(
            "Pyramid level: 0 = full resolution, 1 ≈ 2× downsampled, 2 ≈ 4×, … "
            "If the store has multiscale keys (e.g. 0, 1), the matching level is used. "
            "If the store is a single array, level>0 uses strided downsampling on that array."
        ),
    )
    parser.add_argument(
        "--object-id-column",
        type=str,
        default="pair_id",
        metavar="NAME",
        help="CSV column name for object IDs (default: pair_id).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help=(
            "Optional output CSV. If omitted, writes "
            "<segmentation_stem>_subvolume_stats_L<level>.csv next to the segmentation zarr. "
            "If you pass only a file name (no directory), it is placed next to the segmentation zarr."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes. Objects are distributed across them (default: 1).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    seg_path = _find_zarr_store(args.segmentation_zarr, script_dir)
    raw_path = _find_zarr_store(args.raw_expression_zarr, script_dir)

    if not os.path.exists(seg_path):
        print(f"Error: segmentation zarr not found: {seg_path}")
        sys.exit(1)
    if not os.path.exists(raw_path):
        print(f"Error: raw expression zarr not found: {raw_path}")
        sys.exit(1)

    seg_parent = Path(seg_path).parent
    stem = Path(seg_path).name
    if stem.endswith(".zarr"):
        stem = stem[:-5]

    if args.output_csv is None:
        output_csv = str(seg_parent / f"{stem}_subvolume_stats_L{args.raw_resolution_level}.csv")
    else:
        out_hint = Path(os.path.expanduser(_normalize_cli_path(args.output_csv)))
        if out_hint.is_absolute():
            output_csv = str(out_hint)
        elif str(out_hint.parent) in (".", ""):
            output_csv = str(seg_parent / out_hint.name)
        else:
            output_csv = str(out_hint.resolve())

    sub_volume_analysis(
        segmentation_zarr=seg_path,
        raw_expression_zarr=raw_path,
        output_csv=output_csv,
        raw_resolution_level=args.raw_resolution_level,
        object_id_column=args.object_id_column,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import zarr


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
    raise ValueError(f"Expected 3D–5D array, got ndim={nd} shape={sh}")


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
    factor = 2 ** level
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


def _read_raw_bbox_czyx(
    raw_arr: Any,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    downsample_level: int = 0,
) -> np.ndarray:
    _, nd = _node_shape_ndim(raw_arr)
    factor = 2 ** downsample_level
    full_shape = _raw_node_to_czyx_shape(raw_arr)
    fz0, fz1 = z0 * factor, min(z1 * factor, full_shape[1])
    fy0, fy1 = y0 * factor, min(y1 * factor, full_shape[2])
    fx0, fx1 = x0 * factor, min(x1 * factor, full_shape[3])
    if nd == 5:
        return np.asarray(raw_arr[0, :, fz0:fz1:factor, fy0:fy1:factor, fx0:fx1:factor])
    if nd == 4:
        return np.asarray(raw_arr[:, fz0:fz1:factor, fy0:fy1:factor, fx0:fx1:factor])
    if nd == 3:
        return np.asarray(raw_arr[fz0:fz1:factor, fy0:fy1:factor, fx0:fx1:factor])[np.newaxis, ...]
    raise ValueError(f"Raw expression array must be 3D/4D/5D. Got ndim={nd}")


def _read_label_slice_xy(label_arr: Any, z_idx: int) -> np.ndarray:
    _, nd = _node_shape_ndim(label_arr)
    if nd == 5:
        return np.asarray(label_arr[0, 0, z_idx])
    if nd == 4:
        return np.asarray(label_arr[0, z_idx])
    if nd == 3:
        return np.asarray(label_arr[z_idx])
    raise ValueError(f"Label array must be 3D/4D/5D. Got ndim={nd}")


def _read_label_bbox_zyx(
    label_arr: Any,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    _, nd = _node_shape_ndim(label_arr)
    if nd == 5:
        return np.asarray(label_arr[0, 0, z0:z1, y0:y1, x0:x1])
    if nd == 4:
        return np.asarray(label_arr[0, z0:z1, y0:y1, x0:x1])
    if nd == 3:
        return np.asarray(label_arr[z0:z1, y0:y1, x0:x1])
    raise ValueError(f"Label array must be 3D/4D/5D. Got ndim={nd}")


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


def _resolve_group_path(group_root: Any, key_path: str) -> Any:
    node = group_root
    for part in [p for p in key_path.split("/") if p]:
        node = node[part]
    return node


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


def _select_expression_array_node_matching_segmentation(
    raw_root: Any,
    preferred_key: Optional[str],
    preferred_level: Optional[int],
    segmentation_spatial_shapes: List[Tuple[int, int, int]],
) -> Tuple[str, Any, int]:
    if preferred_level is not None and preferred_level < 0:
        raise ValueError("--raw-resolution-level must be >= 0.")
    if preferred_key is not None and preferred_level is not None:
        raise ValueError("Use either --raw-key or --raw-resolution-level, not both.")

    if preferred_key:
        keys = _zarr_child_keys(raw_root)
        if keys is None:
            raise ValueError(
                "--raw-key is set but the zarr store root is a single array, not a group. "
                "Omit --raw-key or use a group store."
            )
        try:
            chosen_node = _resolve_group_path(raw_root, preferred_key)
        except Exception as exc:
            raise KeyError(
                f"Requested --raw-key '{preferred_key}' not found or invalid. "
                f"Top-level available keys: {keys}"
            ) from exc
        if not (hasattr(chosen_node, "ndim") and chosen_node.ndim >= 3):
            raise ValueError(
                f"Requested --raw-key '{preferred_key}' does not point to a 3D+ array."
            )
        return preferred_key, chosen_node, 0

    raw_candidates = _iter_raw_expression_candidates(raw_root)
    if not raw_candidates:
        raise RuntimeError("Could not find any 3D+ raw expression arrays in the provided zarr store.")

    if preferred_level is not None:
        level_matches: List[Tuple[str, Any]] = []
        candidate_levels: List[Tuple[str, Optional[int]]] = []
        seg_shapes = set(segmentation_spatial_shapes)
        for name, node in raw_candidates:
            level = _infer_resolution_level_from_key(name)
            candidate_levels.append((name, level))
            if level == preferred_level:
                level_matches.append((name, node))
        if not level_matches:
            if len(raw_candidates) == 1 and raw_candidates[0][0] == "<root>":
                root_node = raw_candidates[0][1]
                root_shape = _to_spatial_shape_zyx(root_node)
                ds_shape = _downsampled_spatial_shape(root_shape, preferred_level)
                print(
                    "Raw store has a single array only; applying strided downsampling "
                    f"for requested level {preferred_level} ({root_shape} -> {ds_shape})."
                )
                return f"<root>@level{preferred_level}(strided)", root_node, preferred_level
            lines = [
                f"Requested --raw-resolution-level {preferred_level} was not found.",
                "  Available raw candidates (key -> inferred level):",
            ]
            for name, level in candidate_levels:
                lines.append(f"    {name} -> {level}")
            raise ValueError("\n".join(lines))
        level_shape_matches: List[Tuple[str, Any, Tuple[int, int, int]]] = []
        for name, node in level_matches:
            spatial_zyx = _to_spatial_shape_zyx(node)
            if spatial_zyx in seg_shapes:
                level_shape_matches.append((name, node, spatial_zyx))

        if level_shape_matches:
            level_shape_matches.sort(key=lambda x: x[0])
            chosen_name, chosen_node, chosen_spatial = level_shape_matches[0]
            print(
                "Using raw expression volume "
                f"'{chosen_name}' from --raw-resolution-level {preferred_level} "
                f"(spatial match {chosen_spatial})."
            )
            return chosen_name, chosen_node, 0

        level_matches.sort(key=lambda x: x[0])
        chosen_name, chosen_node = level_matches[0]
        print(
            f"Using raw expression volume '{chosen_name}' from --raw-resolution-level "
            f"{preferred_level}' (no exact segmentation-shape match at this level)."
        )
        return chosen_name, chosen_node, 0

    seg_shapes = set(segmentation_spatial_shapes)
    matching: List[Tuple[int, str, Any, Tuple[int, int, int]]] = []
    all_candidates: List[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]] = []
    for name, node in raw_candidates:
        try:
            sh, _ = _node_shape_ndim(node)
            spatial_zyx = _to_spatial_shape_zyx(node)
        except ValueError:
            continue

        all_candidates.append((name, sh, spatial_zyx))
        if spatial_zyx in seg_shapes:
            priority = -(spatial_zyx[0] * spatial_zyx[1] * spatial_zyx[2])
            matching.append((priority, name, node, spatial_zyx))

    if matching:
        matching.sort(key=lambda x: (x[0], x[1]))
        chosen_name, chosen_node, chosen_spatial = matching[0][1], matching[0][2], matching[0][3]
        print(
            "Using raw expression volume "
            f"'{chosen_name}' (spatial shape {chosen_spatial}) to match segmentation resolution."
        )
        return chosen_name, chosen_node, 0

    lines = [
        "No raw expression multiscale level matches segmentation spatial shape.",
        f"  Segmentation spatial candidates (Z, Y, X): {sorted(seg_shapes)}",
        "  Raw expression candidates:",
    ]
    for name, sh, spatial_zyx in all_candidates:
        lines.append(f"    {name}: ndarray shape {sh} -> spatial {spatial_zyx}")
    raise ValueError("\n".join(lines))


def _pick_label_array_matching_spatial(
    seg: zarr.Group, expected_zyx: Tuple[int, int, int]
) -> Tuple[str, Any]:
    candidates = _iter_segmentation_candidates(seg)
    if not candidates:
        gk = _zarr_child_keys(seg) or []
        raise RuntimeError(
            f"No 3D+ arrays found in segmentation zarr. Keys: {gk}"
        )

    matching: List[Tuple[int, str, Any]] = []
    for name, node in candidates:
        try:
            zyx = _to_spatial_shape_zyx(node)
        except ValueError:
            continue
        if zyx == expected_zyx:
            if name == "0/0":
                priority = 0
            elif name == "0":
                priority = 1
            elif name == "labels":
                priority = 2
            elif name == "interscellar_meshes":
                priority = 3
            else:
                priority = 4
            matching.append((priority, name, node))

    if not matching:
        lines = ["Segmentation zarr has no array with spatial shape matching raw expression."]
        lines.append(f"  Expected (Z, Y, X): {expected_zyx}")
        lines.append("  Candidates:")
        for name, node in candidates:
            try:
                zyx = _to_spatial_shape_zyx(node)
                sh, _ = _node_shape_ndim(node)
                lines.append(f"    {name}: ndarray shape {sh} -> spatial {zyx}")
            except ValueError as e:
                lines.append(f"    {name}: (skip) {e}")
        raise ValueError("\n".join(lines))

    matching.sort(key=lambda x: (x[0], x[1]))
    chosen_name, chosen_node = matching[0][1], matching[0][2]
    print(f"Using segmentation volume '{chosen_name}' (spatial shape matches raw expression).")
    return chosen_name, chosen_node


def _collect_axis_hints(z: zarr.Group) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}
    for k in (
        "axes",
        "dimension_order",
        "_ARRAY_DIMENSIONS",
        "coordinate_system",
        "multiscales",
    ):
        if hasattr(z, "attrs") and k in z.attrs:
            try:
                hints[k] = z.attrs[k]
            except Exception:
                hints[k] = "<unreadable>"
    return hints


def assert_spatial_compatible(
    labels_shape_zyx: Tuple[int, int, int],
    raw_shape_czyx: Tuple[int, int, int, int],
    seg_path: str,
    raw_path: str,
    seg_zarr: zarr.Group,
    raw_zarr: zarr.Group,
) -> None:
    lshape = labels_shape_zyx
    rspatial = raw_shape_czyx[1:]
    if lshape != rspatial:
        seg_hints = _collect_axis_hints(seg_zarr)
        raw_hints = _collect_axis_hints(raw_zarr)
        msg = [
            "Segmentation and raw expression are not spatially compatible (dimension mismatch).",
            f"  Segmentation zarr: {seg_path}",
            f"    Label volume shape (Z, Y, X): {lshape}",
            f"  Raw expression zarr: {raw_path}",
            f"    Shape (C, Z, Y, X): {raw_shape_czyx}",
            f"    Spatial part (Z, Y, X): {rspatial}",
            "  Fix: use the same grid and axis order as the original OME/segmentation pipeline "
            "(same Z,Y,X extent and ordering). Resample or re-export if needed.",
        ]
        if seg_hints:
            msg.append(f"  Segmentation zarr attrs (axis hints): {seg_hints}")
        if raw_hints:
            msg.append(f"  Raw zarr attrs (axis hints): {raw_hints}")
        raise ValueError("\n".join(msg))

    print("Spatial check OK: label volume (Z,Y,X) matches raw expression spatial dimensions.")
    seg_hints = _collect_axis_hints(seg_zarr)
    raw_hints = _collect_axis_hints(raw_zarr)
    if seg_hints or raw_hints:
        print("  Axis metadata (informational): segmentation attrs:", seg_hints or "{}")
        print("  Axis metadata (informational): raw attrs:", raw_hints or "{}")


def _safe_channel_names(n_channels: int, names_csv: Optional[str]) -> List[str]:
    if names_csv is None:
        return [f"channel_{i}" for i in range(n_channels)]
    names = [x.strip() for x in names_csv.split(",") if x.strip()]
    if len(names) != n_channels:
        raise ValueError(
            f"--channel-names has {len(names)} names but raw data has {n_channels} channels."
        )
    return names


def _scan_label_bounding_boxes(
    labels_arr: Any, include_background: bool
) -> Dict[int, Dict[str, int]]:
    z_size, _, _ = _to_spatial_shape_zyx(labels_arr)
    boxes: Dict[int, Dict[str, int]] = {}
    for z_idx in range(z_size):
        label_slice = _read_label_slice_xy(labels_arr, z_idx)
        present_labels = np.unique(label_slice)
        if not include_background:
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
                    "voxel_count": int(ys.size),
                }
                continue
            box = boxes[lid]
            box["z0"] = min(box["z0"], z_idx)
            box["z1"] = max(box["z1"], z_idx + 1)
            box["y0"] = min(box["y0"], y_min)
            box["y1"] = max(box["y1"], y_max)
            box["x0"] = min(box["x0"], x_min)
            box["x1"] = max(box["x1"], x_max)
            box["voxel_count"] += int(ys.size)
    return boxes


def _compute_single_label_stats(
    label_id: int,
    box: Dict[str, int],
    labels_arr: Any,
    raw_arr: Any,
    n_channels: int,
    raw_downsample_level: int,
) -> Dict[str, Any]:
    z0, z1 = box["z0"], box["z1"]
    y0, y1 = box["y0"], box["y1"]
    x0, x1 = box["x0"], box["x1"]
    labels_bbox = _read_label_bbox_zyx(labels_arr, z0, z1, y0, y1, x0, x1)
    obj_mask = labels_bbox == label_id
    voxel_count = int(np.count_nonzero(obj_mask))
    if voxel_count == 0:
        raise RuntimeError(f"Object {label_id} had empty mask inside its own bounding box.")

    raw_bbox = _read_raw_bbox_czyx(
        raw_arr, z0, z1, y0, y1, x0, x1, downsample_level=raw_downsample_level
    )
    if raw_bbox.shape[1:] != labels_bbox.shape:
        raise RuntimeError(
            f"Shape mismatch for object {label_id}: raw bbox spatial {raw_bbox.shape[1:]} "
            f"!= label bbox {labels_bbox.shape}"
        )

    row: Dict[str, Any] = {
        "label_id": int(label_id),
        "voxel_count": voxel_count,
    }
    for c in range(n_channels):
        values = np.asarray(raw_bbox[c])[obj_mask].astype(np.float64, copy=False)
        row[f"channel_{c}_sum"] = float(values.sum())
        row[f"channel_{c}_mean"] = float(values.mean())
        row[f"channel_{c}_std"] = float(values.std())
        row[f"channel_{c}_min"] = float(values.min())
        row[f"channel_{c}_max"] = float(values.max())
        row[f"channel_{c}_nonzero_fraction"] = float(np.count_nonzero(values) / voxel_count)
    return row


def _compute_features_chunked_parallel(
    labels_arr: Any,
    raw_arr: Any,
    include_background: bool,
    object_id_column: str = "pair_id",
    num_workers: int = 4,
    raw_downsample_level: int = 0,
) -> pd.DataFrame:
    boxes = _scan_label_bounding_boxes(labels_arr, include_background=include_background)
    if not boxes:
        raise ValueError("No labeled objects found (after background filtering).")

    n_channels = _raw_node_to_czyx_shape(raw_arr)[0]
    label_ids = sorted(boxes.keys())
    results: List[Dict[str, Any]] = []

    print(f"Discovered {len(label_ids)} objects. Computing per-object bbox features...")
    if num_workers <= 1:
        for lid in label_ids:
            results.append(
                _compute_single_label_stats(
                    lid, boxes[lid], labels_arr, raw_arr, n_channels, raw_downsample_level
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_label = {
                ex.submit(
                    _compute_single_label_stats,
                    lid,
                    boxes[lid],
                    labels_arr,
                    raw_arr,
                    n_channels,
                    raw_downsample_level,
                ): lid
                for lid in label_ids
            }
            for fut in as_completed(future_to_label):
                results.append(fut.result())

    df = pd.DataFrame(results)
    df = df.rename(columns={"label_id": object_id_column})
    return df.sort_values(object_id_column).reset_index(drop=True)


def _apply_channel_name_aliases(df: pd.DataFrame, channel_names: List[str]) -> pd.DataFrame:
    renamed = {}
    for idx, channel_name in enumerate(channel_names):
        prefix = f"channel_{idx}_"
        for col in df.columns:
            if col.startswith(prefix):
                renamed[col] = col.replace(prefix, f"{channel_name}_", 1)
    return df.rename(columns=renamed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-object 3D expression features from a segmentation zarr "
            "(interscellar or cell-only) and a raw expression zarr. "
            "The segmentation volume is chosen automatically so its (Z,Y,X) matches the raw data; "
            "dimensions must be compatible or the run fails with a clear error."
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
        "--raw-key",
        default=None,
        help="Optional zarr key for raw expression (default: auto-detect).",
    )
    parser.add_argument(
        "--raw-resolution-level",
        type=int,
        default=None,
        help=(
            "Optional raw multiscale resolution level (0=full res, 1=2x downsample, "
            "2=4x, ...). Use instead of --raw-key."
        ),
    )
    parser.add_argument(
        "--channel-names",
        default=None,
        help="Optional comma-separated names for channels (must match channel count).",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include label 0 as an object (off by default).",
    )
    parser.add_argument(
        "--object-id-column",
        type=str,
        default="pair_id",
        metavar="NAME",
        help=(
            "CSV column name for voxel label IDs. "
            "Use pair_id for interscellar volumes labels (same as graph/zarr pair_id). "
            "Use cell_id for cell-only label volumes if you prefer."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to <segmentation_stem>_features_3d.csv",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for per-object bbox feature extraction.",
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

    output_csv = args.output_csv
    if output_csv is None:
        stem = Path(seg_path).name
        if stem.endswith(".zarr"):
            stem = stem[:-5]
        output_csv = str(Path(seg_path).with_name(f"{stem}_features_3d.csv"))

    print("Opening zarr stores...")
    seg_zarr = zarr.open(seg_path, mode="r")
    raw_zarr = zarr.open(raw_path, mode="r")

    seg_candidates = _iter_segmentation_candidates(seg_zarr)
    if not seg_candidates:
        gk = _zarr_child_keys(seg_zarr) or []
        raise RuntimeError(f"No 3D+ arrays found in segmentation zarr. Keys: {gk}")
    seg_spatial_shapes: List[Tuple[int, int, int]] = []
    for _name, node in seg_candidates:
        try:
            seg_spatial_shapes.append(_to_spatial_shape_zyx(node))
        except ValueError:
            continue

    raw_key_used, raw_arr, raw_downsample_level = _select_expression_array_node_matching_segmentation(
        raw_zarr,
        args.raw_key,
        args.raw_resolution_level,
        seg_spatial_shapes,
    )
    raw_shape_czyx = _effective_raw_shape_czyx(raw_arr, raw_downsample_level)
    expected_zyx = tuple(int(x) for x in raw_shape_czyx[1:])

    print(f"Raw expression shape (C, Z, Y, X): {raw_shape_czyx}")
    print(f"Raw expression level used: {raw_key_used}")
    print(f"Expected spatial grid (Z, Y, X): {expected_zyx}")

    _seg_key_name, labels_arr = _pick_label_array_matching_spatial(seg_zarr, expected_zyx)
    labels_shape_zyx = _to_spatial_shape_zyx(labels_arr)

    assert_spatial_compatible(
        labels_shape_zyx, raw_shape_czyx, seg_path, raw_path, seg_zarr, raw_zarr
    )

    channel_names = _safe_channel_names(raw_shape_czyx[0], args.channel_names)
    print(f"Labels shape (Z, Y, X): {labels_shape_zyx}")
    print(f"Channels: {channel_names}")
    print(f"Computing features with {max(1, args.num_workers)} worker(s)...")

    features_df = _compute_features_chunked_parallel(
        labels_arr=labels_arr,
        raw_arr=raw_arr,
        include_background=args.include_background,
        object_id_column=args.object_id_column,
        num_workers=max(1, args.num_workers),
        raw_downsample_level=raw_downsample_level,
    )
    features_df = _apply_channel_name_aliases(features_df, channel_names)
    features_df.to_csv(output_csv, index=False)

    print("Feature extraction completed.")
    print(f"Objects extracted: {len(features_df)}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _slice_label_like_visualize(arr: Any) -> np.ndarray:
    sh, nd = _node_shape_ndim(arr)
    if nd == 5:
        return np.asarray(arr[0, 0])
    if nd == 4:
        return np.asarray(arr[0])
    if nd == 3:
        return np.asarray(arr)
    raise ValueError(f"Label array must be 3D/4D/5D. Got shape={sh}")


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


def _load_expression_array(raw_root: Any, preferred_key: Optional[str]) -> np.ndarray:
    keys = _zarr_child_keys(raw_root)
    if keys is None:
        if preferred_key:
            raise ValueError(
                "--raw-key is set but the zarr store root is a single array, not a group. "
                "Omit --raw-key or use a group store."
            )
        arr = raw_root
    elif preferred_key:
        if preferred_key not in keys:
            raise KeyError(
                f"Requested --raw-key '{preferred_key}' not found. "
                f"Available keys: {keys}"
            )
        arr = raw_root[preferred_key]
    elif "0" in keys:
        node = raw_root["0"]
        sub = _zarr_child_keys(node)
        if sub is not None and "0" in sub:
            arr = node["0"]
        else:
            arr = node
    elif "labels" in keys:
        arr = raw_root["labels"]
    else:
        arr = None
        for key in keys:
            node = raw_root[key]
            if hasattr(node, "ndim") and node.ndim >= 3:
                arr = node
                print(f"Using raw expression data from key '{key}'")
                break
        if arr is None:
            raise RuntimeError(
                f"Could not find a valid expression array. Keys: {keys}"
            )

    sh, nd = _node_shape_ndim(arr)
    if nd == 5:
        return np.asarray(arr[0])
    if nd == 4:
        return np.asarray(arr)
    if nd == 3:
        return np.asarray(arr)[np.newaxis, ...]
    raise ValueError(f"Raw expression array must be 3D/4D/5D. Got shape={sh}")


def _pick_label_array_matching_spatial(
    seg: zarr.Group, expected_zyx: Tuple[int, int, int]
) -> Tuple[str, np.ndarray]:
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
    labels_3d = _slice_label_like_visualize(chosen_node)
    if labels_3d.shape != expected_zyx:
        raise RuntimeError(
            f"Internal error: after slicing, label shape {labels_3d.shape} != expected {expected_zyx}"
        )
    return chosen_name, labels_3d


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
    labels_3d: np.ndarray,
    raw_czyx: np.ndarray,
    seg_path: str,
    raw_path: str,
    seg_zarr: zarr.Group,
    raw_zarr: zarr.Group,
) -> None:
    lshape = labels_3d.shape
    rspatial = raw_czyx.shape[1:]
    if lshape != rspatial:
        seg_hints = _collect_axis_hints(seg_zarr)
        raw_hints = _collect_axis_hints(raw_zarr)
        msg = [
            "Segmentation and raw expression are not spatially compatible (dimension mismatch).",
            f"  Segmentation zarr: {seg_path}",
            f"    Label volume shape (Z, Y, X): {lshape}",
            f"  Raw expression zarr: {raw_path}",
            f"    Shape (C, Z, Y, X): {raw_czyx.shape}",
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


def _init_channel_stats(label_ids: np.ndarray, n_channels: int) -> Dict[str, np.ndarray]:
    n_labels = len(label_ids)
    return {
        "sum": np.zeros((n_labels, n_channels), dtype=np.float64),
        "sum_sq": np.zeros((n_labels, n_channels), dtype=np.float64),
        "nonzero_count": np.zeros((n_labels, n_channels), dtype=np.int64),
        "min": np.full((n_labels, n_channels), np.inf, dtype=np.float64),
        "max": np.full((n_labels, n_channels), -np.inf, dtype=np.float64),
    }


def _compute_features(
    labels_3d: np.ndarray,
    raw_czyx: np.ndarray,
    include_background: bool,
    object_id_column: str = "pair_id",
) -> pd.DataFrame:
    unique_labels, counts = np.unique(np.asarray(labels_3d), return_counts=True)
    if not include_background:
        keep = unique_labels > 0
        unique_labels = unique_labels[keep]
        counts = counts[keep]
    if len(unique_labels) == 0:
        raise ValueError("No labeled objects found (after background filtering).")

    label_to_idx = {int(lbl): i for i, lbl in enumerate(unique_labels)}
    n_channels = raw_czyx.shape[0]
    stats = _init_channel_stats(unique_labels, n_channels)

    for z_idx in range(labels_3d.shape[0]):
        label_slice = np.asarray(labels_3d[z_idx])
        present_labels = np.unique(label_slice)
        if not include_background:
            present_labels = present_labels[present_labels > 0]
        if len(present_labels) == 0:
            continue

        for label_id in present_labels:
            obj_mask = label_slice == label_id
            if not np.any(obj_mask):
                continue
            row_idx = label_to_idx[int(label_id)]
            for c in range(n_channels):
                values = np.asarray(raw_czyx[c, z_idx])[obj_mask].astype(np.float64)
                if values.size == 0:
                    continue
                stats["sum"][row_idx, c] += values.sum()
                stats["sum_sq"][row_idx, c] += np.square(values).sum()
                stats["nonzero_count"][row_idx, c] += np.count_nonzero(values)
                local_min = float(values.min())
                local_max = float(values.max())
                if local_min < stats["min"][row_idx, c]:
                    stats["min"][row_idx, c] = local_min
                if local_max > stats["max"][row_idx, c]:
                    stats["max"][row_idx, c] = local_max

    voxel_count = counts.astype(np.float64)
    mean = stats["sum"] / voxel_count[:, None]
    variance = np.clip(stats["sum_sq"] / voxel_count[:, None] - np.square(mean), a_min=0.0, a_max=None)
    std = np.sqrt(variance)
    frac_nonzero = stats["nonzero_count"] / voxel_count[:, None]

    data = {
        object_id_column: unique_labels.astype(np.int64),
        "voxel_count": counts.astype(np.int64),
    }
    for c in range(n_channels):
        data[f"channel_{c}_sum"] = stats["sum"][:, c]
        data[f"channel_{c}_mean"] = mean[:, c]
        data[f"channel_{c}_std"] = std[:, c]
        data[f"channel_{c}_min"] = stats["min"][:, c]
        data[f"channel_{c}_max"] = stats["max"][:, c]
        data[f"channel_{c}_nonzero_fraction"] = frac_nonzero[:, c]

    return pd.DataFrame(data).sort_values(object_id_column).reset_index(drop=True)


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

    raw_czyx = _load_expression_array(raw_zarr, args.raw_key)
    expected_zyx = tuple(int(x) for x in raw_czyx.shape[1:])

    print(f"Raw expression shape (C, Z, Y, X): {raw_czyx.shape}")
    print(f"Expected spatial grid (Z, Y, X): {expected_zyx}")

    _seg_key_name, labels_3d = _pick_label_array_matching_spatial(seg_zarr, expected_zyx)

    assert_spatial_compatible(
        labels_3d, raw_czyx, seg_path, raw_path, seg_zarr, raw_zarr
    )

    channel_names = _safe_channel_names(raw_czyx.shape[0], args.channel_names)
    print(f"Labels shape (Z, Y, X): {labels_3d.shape}")
    print(f"Channels: {channel_names}")
    print("Computing features...")

    features_df = _compute_features(
        labels_3d=labels_3d,
        raw_czyx=raw_czyx,
        include_background=args.include_background,
        object_id_column=args.object_id_column,
    )
    features_df = _apply_channel_name_aliases(features_df, channel_names)
    features_df.to_csv(output_csv, index=False)

    print("Feature extraction completed.")
    print(f"Objects extracted: {len(features_df)}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()

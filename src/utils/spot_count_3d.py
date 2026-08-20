from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from .exclude_truncated import _pick_segmentation_label_array
from .feature_extraction_3d import (
    _find_zarr_store,
    _read_label_bbox_zyx,
    _to_spatial_shape_zyx,
)

_AXIS_EXACT = {
    "x": ("x", "X"),
    "y": ("y", "Y"),
    "z": ("z", "Z"),
}
_AXIS_PATTERNS = {
    "x": re.compile(r"(^|[_\-. ])x([_\-. ]|$)|(^|[_\-. ])xpos(ition)?([_\-. ]|$)", re.I),
    "y": re.compile(r"(^|[_\-. ])y([_\-. ]|$)|(^|[_\-. ])ypos(ition)?([_\-. ]|$)", re.I),
    "z": re.compile(r"(^|[_\-. ])z([_\-. ]|$)|(^|[_\-. ])zpos(ition)?([_\-. ]|$)", re.I),
}


def _default_output_csv(metadata_csv: str, biomarker: str) -> str:
    metadata_dir = os.path.dirname(metadata_csv) or "."
    stem = os.path.splitext(os.path.basename(metadata_csv))[0]
    return os.path.join(metadata_dir, f"{stem}_{biomarker}_spotcount.csv")


def _count_column_name(biomarker: str) -> str:
    name = str(biomarker).strip()
    if not name:
        raise ValueError("Biomarker name must be a non-empty string.")
    if any(c.isspace() for c in name):
        name = "_".join(name.split())
    return f"{name}_spotcount"


def _resolve_axis_column(
    columns: List[str],
    axis: str,
    explicit: Optional[str],
) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Spot CSV is missing --{axis}-column '{explicit}'. "
                f"Available columns: {list(columns)}"
            )
        return explicit

    lower_map = {c.lower(): c for c in columns}
    for candidate in _AXIS_EXACT[axis]:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    pattern = _AXIS_PATTERNS[axis]
    matches = [c for c in columns if pattern.search(c)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {axis.upper()} column in spots CSV among {matches}. "
            f"Pass --{axis}-column explicitly."
        )
    raise ValueError(
        f"Could not find an {axis.upper()} column in spots CSV. "
        f"Available columns: {list(columns)}. Pass --{axis}-column."
    )


def _resolve_xyz_columns(
    spots_df: pd.DataFrame,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    z_column: Optional[str] = None,
) -> Tuple[str, str, str]:
    cols = list(spots_df.columns.astype(str))
    x_col = _resolve_axis_column(cols, "x", x_column)
    y_col = _resolve_axis_column(cols, "y", y_column)
    z_col = _resolve_axis_column(cols, "z", z_column)
    if len({x_col, y_col, z_col}) < 3:
        raise ValueError(
            f"Resolved XYZ columns are not unique: x={x_col}, y={y_col}, z={z_col}"
        )
    return x_col, y_col, z_col


def _find_csv(path_hint: str, script_dir: str) -> str:
    candidates = [
        os.path.abspath(path_hint),
        os.path.join(script_dir, path_hint),
        os.path.join(os.path.dirname(script_dir), path_hint),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(path_hint)


def _spot_labels_at_voxels(
    labels: object,
    zs: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    shape_zyx: Tuple[int, int, int],
) -> np.ndarray:
    """Return label IDs at rounded voxel coords; OOB spots get 0."""
    z_size, y_size, x_size = shape_zyx
    n = int(zs.size)
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out

    in_bounds = (
        (zs >= 0)
        & (zs < z_size)
        & (ys >= 0)
        & (ys < y_size)
        & (xs >= 0)
        & (xs < x_size)
    )
    if not np.any(in_bounds):
        return out

    idx = np.nonzero(in_bounds)[0]
    zb = zs[idx]
    yb = ys[idx]
    xb = xs[idx]

    order = np.argsort(zb, kind="mergesort")
    idx = idx[order]
    zb = zb[order]
    yb = yb[order]
    xb = xb[order]

    unique_z, starts = np.unique(zb, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = zb.size

    for z_val, start, end in tqdm(
        zip(unique_z, starts, ends),
        total=int(unique_z.size),
        desc="Looking up spot labels",
        unit="z",
    ):
        z_i = int(z_val)
        slab = np.asarray(_read_label_bbox_zyx(labels, z_i, z_i + 1, 0, y_size, 0, x_size))
        xy = slab[0] if slab.ndim == 3 else slab
        out[idx[start:end]] = xy[yb[start:end], xb[start:end]].astype(np.int64, copy=False)
    return out


def count_spots_per_volume_3d(
    combined_zarr: str,
    metadata_csv: str,
    spots_csv: str,
    biomarker: str,
    output_csv: Optional[str] = None,
    volume_id_column: str = "id",
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    z_column: Optional[str] = None,
) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combined_path = _find_zarr_store(combined_zarr, script_dir)
    metadata_path = _find_csv(metadata_csv, script_dir)
    spots_path = _find_csv(spots_csv, script_dir)

    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined zarr not found: {combined_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not os.path.isfile(spots_path):
        raise FileNotFoundError(f"Spots CSV not found: {spots_path}")

    count_col = _count_column_name(biomarker)
    if output_csv is None:
        output_csv = _default_output_csv(metadata_path, biomarker.strip().replace(" ", "_"))
    output_csv = os.path.abspath(output_csv)

    print(f"Loading combined zarr: {combined_path}")
    root = zarr.open(combined_path, mode="r")
    key, labels = _pick_segmentation_label_array(root)
    z_size, y_size, x_size = _to_spatial_shape_zyx(labels)
    print(f"  Label key '{key}' shape (Z, Y, X): {(z_size, y_size, x_size)}")

    print(f"Loading metadata: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    if volume_id_column not in metadata_df.columns:
        raise ValueError(
            f"Metadata CSV is missing '{volume_id_column}'. "
            f"Available columns: {list(metadata_df.columns)}"
        )
    if count_col in metadata_df.columns:
        print(f"  Warning: overwriting existing column '{count_col}'")

    print(f"Loading spots: {spots_path}")
    spots_df = pd.read_csv(spots_path)
    total_spots = int(len(spots_df))
    x_col, y_col, z_col = _resolve_xyz_columns(
        spots_df, x_column=x_column, y_column=y_column, z_column=z_column
    )
    print(f"  Using spot columns: x={x_col}, y={y_col}, z={z_col}")
    print(f"  Total spots (CSV rows): {total_spots}")

    x_f = pd.to_numeric(spots_df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y_f = pd.to_numeric(spots_df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    z_f = pd.to_numeric(spots_df[z_col], errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~(np.isfinite(x_f) & np.isfinite(y_f) & np.isfinite(z_f))
    if np.any(invalid):
        print(f"  Warning: {int(invalid.sum())} spots have non-numeric/NaN coordinates")
        x_f = np.where(invalid, -1.0, x_f)
        y_f = np.where(invalid, -1.0, y_f)
        z_f = np.where(invalid, -1.0, z_f)
    xs = np.rint(x_f).astype(np.int64)
    ys = np.rint(y_f).astype(np.int64)
    zs = np.rint(z_f).astype(np.int64)

    spot_labels = _spot_labels_at_voxels(
        labels, zs, ys, xs, (z_size, y_size, x_size)
    )
    inside = spot_labels > 0
    n_inside = int(inside.sum())
    print(
        f"{n_inside} out of total {total_spots} spots were detected inside "
        "an interscellar or cell only volume"
    )

    counts: Dict[int, int] = {}
    if n_inside:
        uniq, freq = np.unique(spot_labels[inside], return_counts=True)
        counts = {int(vid): int(n) for vid, n in zip(uniq, freq)}

    volume_ids = pd.to_numeric(metadata_df[volume_id_column], errors="coerce")
    metadata_out = metadata_df.copy()
    metadata_out[count_col] = [
        counts.get(int(vid), 0) if pd.notna(vid) else 0 for vid in volume_ids
    ]

    unmatched = sorted(vid for vid in counts if vid not in set(
        int(v) for v in volume_ids.dropna().astype(np.int64)
    ))
    if unmatched:
        print(
            f"  Warning: {len(unmatched)} volume IDs had spots but are missing "
            f"from metadata (examples: {unmatched[:10]})"
        )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    metadata_out.to_csv(output_csv, index=False)
    print(f"Wrote {count_col} for {len(metadata_out)} volumes to: {output_csv}")
    return output_csv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count spotmask XYZ points (voxel coordinates) that fall inside each "
            "volume ID in a combined 3D label zarr. Appends <biomarker>_spotcount "
            "to the metadata CSV."
        )
    )
    parser.add_argument(
        "--combined-zarr",
        required=True,
        help="Path to combined volumes zarr from combine_volumes_3d.",
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="CSV of volume IDs in the combined zarr (must include id by default).",
    )
    parser.add_argument(
        "--spots-csv",
        required=True,
        help="CSV of spot XYZ coordinates in voxel units.",
    )
    parser.add_argument(
        "--biomarker",
        required=True,
        help="Biomarker name used for the output column <biomarker>_spotcount.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<metadata_dir>/<metadata_stem>_<biomarker>_spotcount.csv"
        ),
    )
    parser.add_argument(
        "--volume-id-column",
        default="id",
        help="Metadata column with combined-zarr volume IDs (default: id).",
    )
    parser.add_argument(
        "--x-column",
        default=None,
        help="Spots CSV column for X (auto-detected if omitted).",
    )
    parser.add_argument(
        "--y-column",
        default=None,
        help="Spots CSV column for Y (auto-detected if omitted).",
    )
    parser.add_argument(
        "--z-column",
        default=None,
        help="Spots CSV column for Z (auto-detected if omitted).",
    )
    args = parser.parse_args(argv)

    try:
        count_spots_per_volume_3d(
            combined_zarr=args.combined_zarr,
            metadata_csv=args.metadata_csv,
            spots_csv=args.spots_csv,
            biomarker=args.biomarker,
            output_csv=args.output_csv,
            volume_id_column=args.volume_id_column,
            x_column=args.x_column,
            y_column=args.y_column,
            z_column=args.z_column,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

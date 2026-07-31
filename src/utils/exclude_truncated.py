from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

from .feature_extraction_3d import (
    _find_zarr_store,
    _iter_segmentation_candidates,
    _read_label_slice_xy,
    _to_spatial_shape_zyx,
    _zarr_child_keys,
)


def _pick_segmentation_label_array(seg_zarr: Any) -> Tuple[str, Any]:
    candidates = _iter_segmentation_candidates(seg_zarr)
    if not candidates:
        keys = _zarr_child_keys(seg_zarr) or []
        raise RuntimeError(f"No 3D+ arrays found in segmentation zarr. Keys: {keys}")

    priority = {
        "labels": 0,
        "0/0": 1,
        "0": 2,
    }
    ranked: List[Tuple[int, str, Any]] = []
    for name, node in candidates:
        try:
            _to_spatial_shape_zyx(node)
        except ValueError:
            continue
        ranked.append((priority.get(name, 10), name, node))

    if not ranked:
        raise RuntimeError("No usable ZYX label volume found in segmentation zarr.")

    ranked.sort(key=lambda x: (x[0], x[1]))
    name, node = ranked[0][1], ranked[0][2]
    return name, node


def _scan_tight_z_extents(
    labels_arr: Any, include_background: bool = False
) -> Dict[int, Tuple[int, int]]:
    z_size, _, _ = _to_spatial_shape_zyx(labels_arr)
    extents: Dict[int, Tuple[int, int]] = {}

    for z_idx in range(z_size):
        label_slice = _read_label_slice_xy(labels_arr, z_idx)
        present = np.unique(label_slice)
        if not include_background:
            present = present[present > 0]
        for label_id in present:
            lid = int(label_id)
            if lid not in extents:
                extents[lid] = (z_idx, z_idx + 1)
            else:
                z0, z1 = extents[lid]
                extents[lid] = (min(z0, z_idx), max(z1, z_idx + 1))

    return extents


def compute_percentile_z_thresholds(
    z_extents: Dict[int, Tuple[int, int]],
    buffer_voxel_min: int = 5,
    buffer_voxel_max: int = 5,
    z0_percentile: float = 1.0,
    z1_percentile: float = 99.0,
) -> Dict[str, float]:
    if buffer_voxel_min < 0:
        raise ValueError(f"buffer_voxel_min must be >= 0, got {buffer_voxel_min}")
    if buffer_voxel_max < 0:
        raise ValueError(f"buffer_voxel_max must be >= 0, got {buffer_voxel_max}")
    if not z_extents:
        raise ValueError("No cells found; cannot compute percentile thresholds.")

    z0_vals = np.asarray([z0 for z0, _ in z_extents.values()], dtype=np.float64)
    z1_vals = np.asarray([z1 for _, z1 in z_extents.values()], dtype=np.float64)

    z0_at_pct = float(np.percentile(z0_vals, z0_percentile))
    z1_at_pct = float(np.percentile(z1_vals, z1_percentile))
    top_threshold = z0_at_pct + float(buffer_voxel_min)
    bottom_threshold = z1_at_pct - float(buffer_voxel_max)

    if bottom_threshold <= top_threshold:
        raise ValueError(
            "Invalid Z thresholds: bottom_threshold "
            f"({bottom_threshold}) <= top_threshold ({top_threshold}). "
            "Try smaller --buffer-voxel-min / --buffer-voxel-max."
        )

    return {
        "n_cells": float(len(z_extents)),
        "z0_min": float(z0_vals.min()),
        "z0_max": float(z0_vals.max()),
        "z1_min": float(z1_vals.min()),
        "z1_max": float(z1_vals.max()),
        "z0_percentile": float(z0_percentile),
        "z1_percentile": float(z1_percentile),
        "z0_at_percentile": z0_at_pct,
        "z1_at_percentile": z1_at_pct,
        "buffer_voxel_min": float(buffer_voxel_min),
        "buffer_voxel_max": float(buffer_voxel_max),
        "top_threshold": top_threshold,
        "bottom_threshold": bottom_threshold,
    }


def _write_z_bboxes_csv(
    z_extents: Dict[int, Tuple[int, int]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("cell_id,z0,z1\n")
        for cell_id in sorted(z_extents):
            z0, z1 = z_extents[cell_id]
            f.write(f"{cell_id},{z0},{z1}\n")


def _write_percentile_summary(meta: Dict[str, Any], path: Path) -> None:
    summary_lines = [
        f"segmentation_zarr: {meta['segmentation_zarr']}",
        f"label_key: {meta['label_key']}",
        f"max_z: {meta['max_z']}",
        f"n_cells: {int(meta['n_cells'])}",
        f"n_excluded: {int(meta['n_excluded'])}",
        f"z0 range: [{meta['z0_min']}, {meta['z0_max']}]",
        f"z1 range: [{meta['z1_min']}, {meta['z1_max']}]",
        (
            f"z0 {meta['z0_percentile']}th percentile: "
            f"{meta['z0_at_percentile']}"
        ),
        (
            f"z1 {meta['z1_percentile']}th percentile: "
            f"{meta['z1_at_percentile']}"
        ),
        f"buffer_voxel_min: {meta['buffer_voxel_min']}",
        f"buffer_voxel_max: {meta['buffer_voxel_max']}",
        f"top_threshold (z0_p + buffer_voxel_min): {meta['top_threshold']}",
        f"bottom_threshold (z1_p - buffer_voxel_max): {meta['bottom_threshold']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _write_excluded_cell_ids(excluded: List[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cell_id in excluded:
            f.write(f"{cell_id}\n")


def cells_outside_percentile_z_window(
    z_extents: Dict[int, Tuple[int, int]],
    top_threshold: float,
    bottom_threshold: float,
) -> List[int]:
    excluded: List[int] = []
    for cell_id, (z0, z1) in z_extents.items():
        if z0 < top_threshold or z1 > bottom_threshold:
            excluded.append(int(cell_id))
    return sorted(excluded)


def find_edge_excluded_cell_ids(
    segmentation_zarr: str,
    buffer_voxel_min: int = 5,
    buffer_voxel_max: int = 5,
    z0_percentile: float = 1.0,
    z1_percentile: float = 99.0,
) -> Tuple[List[int], Dict[int, Tuple[int, int]], Dict[str, Any]]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seg_path = _find_zarr_store(segmentation_zarr, script_dir)
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation zarr not found: {seg_path}")

    seg_zarr = zarr.open(seg_path, mode="r")
    key_name, labels_arr = _pick_segmentation_label_array(seg_zarr)
    max_z, max_y, max_x = _to_spatial_shape_zyx(labels_arr)

    print(f"Segmentation zarr: {seg_path}")
    print(f"Using label volume '{key_name}' with shape (Z, Y, X)=({max_z}, {max_y}, {max_x})")
    print("Scanning tight Z extents for all cells...")

    z_extents = _scan_tight_z_extents(labels_arr, include_background=False)
    thresholds = compute_percentile_z_thresholds(
        z_extents,
        buffer_voxel_min=buffer_voxel_min,
        buffer_voxel_max=buffer_voxel_max,
        z0_percentile=z0_percentile,
        z1_percentile=z1_percentile,
    )
    excluded = cells_outside_percentile_z_window(
        z_extents,
        top_threshold=thresholds["top_threshold"],
        bottom_threshold=thresholds["bottom_threshold"],
    )

    meta = {
        "segmentation_zarr": seg_path,
        "label_key": key_name,
        "max_z": int(max_z),
        "n_excluded": len(excluded),
        **thresholds,
        "n_cells": int(thresholds["n_cells"]),
    }
    return excluded, z_extents, meta


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-cell tight Z bboxes and percentiles, then exclude cells "
            "outside a percentile-based window: "
            "top = 1st percentile of z0 + buffer_voxel_min, "
            "bottom = 99th percentile of z1 - buffer_voxel_max."
        )
    )
    parser.add_argument(
        "--segmentation-zarr",
        required=True,
        help="Path to cell segmentation OME-Zarr (directory).",
    )
    parser.add_argument(
        "--buffer-voxel-min",
        type=int,
        default=5,
        help=(
            "Margin added to the z0 percentile for the top (low-Z) threshold "
            "(top = p1(z0)+buffer_voxel_min; default: 5)."
        ),
    )
    parser.add_argument(
        "--buffer-voxel-max",
        type=int,
        default=5,
        help=(
            "Margin subtracted from the z1 percentile for the bottom (high-Z) threshold "
            "(bottom = p99(z1)-buffer_voxel_max; default: 5)."
        ),
    )
    parser.add_argument(
        "--z0-percentile",
        type=float,
        default=1.0,
        help="Percentile of z0 used for the top threshold (default: 1).",
    )
    parser.add_argument(
        "--z1-percentile",
        type=float,
        default=99.0,
        help="Percentile of z1 used for the bottom threshold (default: 99).",
    )
    parser.add_argument(
        "--output-bboxes",
        default=None,
        help=(
            "CSV of per-cell tight Z extents (cell_id,z0,z1). "
            "Defaults to <segmentation_stem>_z_bboxes.csv next to the zarr."
        ),
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help=(
            "Percentile summary text path. "
            "Defaults to <segmentation_stem>_z_bbox_percentiles.txt next to the zarr."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. .csv writes a cell_id column; otherwise one ID per line. "
            "Defaults to <segmentation_stem>_z_edge_excluded_cells.txt next to the zarr."
        ),
    )
    args = parser.parse_args(argv)

    try:
        excluded, z_extents, meta = find_edge_excluded_cell_ids(
            args.segmentation_zarr,
            buffer_voxel_min=args.buffer_voxel_min,
            buffer_voxel_max=args.buffer_voxel_max,
            z0_percentile=args.z0_percentile,
            z1_percentile=args.z1_percentile,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    stem = Path(meta["segmentation_zarr"]).name
    if stem.endswith(".zarr"):
        stem = stem[:-5]
    base = Path(meta["segmentation_zarr"]).with_name(stem)

    bboxes_path = Path(
        args.output_bboxes if args.output_bboxes is not None else f"{base}_z_bboxes.csv"
    )
    summary_path = Path(
        args.output_summary
        if args.output_summary is not None
        else f"{base}_z_bbox_percentiles.txt"
    )
    excluded_path = Path(
        args.output
        if args.output is not None
        else f"{base}_z_edge_excluded_cells.txt"
    )

    _write_z_bboxes_csv(z_extents, bboxes_path)
    _write_percentile_summary(meta, summary_path)
    print(f"Cells: {meta['n_cells']}")
    print(
        f"z0 {meta['z0_percentile']}th percentile: {meta['z0_at_percentile']}"
    )
    print(
        f"z1 {meta['z1_percentile']}th percentile: {meta['z1_at_percentile']}"
    )
    print(f"Wrote per-cell Z bboxes to: {bboxes_path}")
    print(f"Wrote percentile summary to: {summary_path}")

    print(
        f"Top threshold (z0_p + buffer_voxel_min={meta['buffer_voxel_min']:.0f}): "
        f"{meta['top_threshold']:.4g} (exclude z0 < this)"
    )
    print(
        f"Bottom threshold (z1_p - buffer_voxel_max={meta['buffer_voxel_max']:.0f}): "
        f"{meta['bottom_threshold']:.4g} (exclude z1 > this)"
    )

    _write_excluded_cell_ids(excluded, excluded_path)
    print(
        f"Cells scanned: {meta['n_cells']}; excluded (Z-edge): {meta['n_excluded']}"
    )
    print(f"Wrote excluded cell IDs to: {excluded_path}")
    if excluded:
        preview = ", ".join(str(x) for x in excluded[:20])
        more = "" if len(excluded) <= 20 else f", ... ({len(excluded) - 20} more)"
        print(f"Excluded IDs: {preview}{more}")
    else:
        print("Excluded IDs: (none)")


if __name__ == "__main__":
    main()

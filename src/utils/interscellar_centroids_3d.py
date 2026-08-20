from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from .combine_volumes_3d import _iter_z_slabs, _spatial_z_chunk
from .exclude_truncated import _pick_segmentation_label_array
from .feature_extraction_3d import (
    _find_zarr_store,
    _read_label_bbox_zyx,
    _to_spatial_shape_zyx,
)

DEFAULT_VOXEL_SIZE_UM = (0.56, 0.28, 0.28)


def _default_output_csv(combined_zarr: str) -> str:
    combined_dir = os.path.dirname(combined_zarr) or "."
    basename = os.path.basename(combined_zarr.rstrip(os.sep))
    stem = basename[:-5] if basename.endswith(".zarr") else os.path.splitext(basename)[0]
    return os.path.join(combined_dir, f"{stem}_interscellar_centroids.csv")


def _interscellar_id_range(root: Any) -> Tuple[int, int]:
    start = root.attrs.get("interscellar_id_start")
    end = root.attrs.get("max_combined_id")
    if start is not None and end is not None:
        return int(start), int(end)

    if "interscellar_id_map" in root:
        id_map = np.asarray(root["interscellar_id_map"])
        if id_map.size:
            remapped = id_map[:, 1] if id_map.ndim == 2 else id_map
            remapped = remapped.astype(np.int64, copy=False)
            remapped = remapped[remapped > 0]
            if remapped.size:
                return int(remapped.min()), int(remapped.max())

    max_cell = root.attrs.get("max_cell_only_id")
    if max_cell is not None:
        start = int(max_cell) + 1
        return start, start - 1

    raise ValueError(
        "Could not determine interscellar ID range. Combined zarr is missing "
        "'interscellar_id_start' / 'max_combined_id' attrs and 'interscellar_id_map'."
    )


def _parse_voxel_size_um(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if len(vals) != 3:
        return None
    return vals


def _resolve_voxel_size_um(
    root: Any,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    parsed = _parse_voxel_size_um(voxel_size_um)
    if parsed is not None:
        return parsed

    parsed = _parse_voxel_size_um(root.attrs.get("voxel_size_um"))
    if parsed is not None:
        return parsed

    inter_path = root.attrs.get("interscellar_zarr")
    if inter_path and os.path.exists(str(inter_path)):
        try:
            inter_root = zarr.open(str(inter_path), mode="r")
            parsed = _parse_voxel_size_um(inter_root.attrs.get("voxel_size_um"))
            if parsed is not None:
                return parsed
        except Exception:
            pass

    print(
        f"Warning: voxel_size_um not found in zarr attributes, "
        f"using default: {DEFAULT_VOXEL_SIZE_UM}"
    )
    return DEFAULT_VOXEL_SIZE_UM


def compute_interscellar_centroids_3d(
    combined_zarr: str,
    output_csv: Optional[str] = None,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combined_path = _find_zarr_store(combined_zarr, script_dir)
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined zarr not found: {combined_path}")

    if output_csv is None:
        output_csv = _default_output_csv(combined_path)
    output_csv = os.path.abspath(output_csv)

    print(f"Loading combined zarr: {combined_path}")
    root = zarr.open(combined_path, mode="r")
    key, labels = _pick_segmentation_label_array(root)
    z_size, y_size, x_size = _to_spatial_shape_zyx(labels)
    print(f"  Label key '{key}' shape (Z, Y, X): {(z_size, y_size, x_size)}")

    vz, vy, vx = _resolve_voxel_size_um(root, voxel_size_um)
    print(f"  Voxel size (Z, Y, X) μm: {(vz, vy, vx)}")
    print("  Centroids will be written in microns")

    id_start, id_end = _interscellar_id_range(root)
    n_ids = int(id_end) - int(id_start) + 1
    if n_ids <= 0:
        print("No interscellar volumes found; writing empty CSV")
        df = pd.DataFrame(columns=["volume_id", "x", "y", "z"])
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")
        return output_csv

    print(f"  Interscellar remapped IDs: {id_start}..{id_end} ({n_ids} IDs)")
    z_chunk = _spatial_z_chunk(labels)

    sum_x = np.zeros(n_ids, dtype=np.float64)
    sum_y = np.zeros(n_ids, dtype=np.float64)
    sum_z = np.zeros(n_ids, dtype=np.float64)
    counts = np.zeros(n_ids, dtype=np.int64)

    for z0, z1 in tqdm(
        list(_iter_z_slabs(z_size, z_chunk)),
        desc="Accumulating centroids",
        unit="slab",
    ):
        slab = np.asarray(_read_label_bbox_zyx(labels, z0, z1, 0, y_size, 0, x_size))
        mask = (slab >= id_start) & (slab <= id_end)
        if not np.any(mask):
            continue
        idx = slab[mask].astype(np.int64, copy=False) - id_start
        zz, yy, xx = np.nonzero(mask)
        counts += np.bincount(idx, minlength=n_ids)
        sum_z += np.bincount(idx, weights=(zz + z0).astype(np.float64), minlength=n_ids)
        sum_y += np.bincount(idx, weights=yy.astype(np.float64), minlength=n_ids)
        sum_x += np.bincount(idx, weights=xx.astype(np.float64), minlength=n_ids)

    present = counts > 0
    volume_ids = (np.arange(n_ids, dtype=np.int64) + id_start)[present]
    df = pd.DataFrame(
        {
            "volume_id": volume_ids,
            "x": (sum_x[present] / counts[present]) * vx,
            "y": (sum_y[present] / counts[present]) * vy,
            "z": (sum_z[present] / counts[present]) * vz,
        }
    )

    missing = int((~present).sum())
    if missing:
        print(f"  Warning: {missing} remapped interscellar IDs had no voxels")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} interscellar centroids to: {output_csv}")
    return output_csv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute XYZ centroids in microns for interscellar volumes "
            "in a combined 3D label zarr from combine_volumes_3d. Cell-only "
            "labels are ignored. Output CSV columns: volume_id, x, y, z."
        )
    )
    parser.add_argument(
        "--combined-zarr",
        required=True,
        help="Path to combined volumes zarr from combine_volumes_3d.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<combined_dir>/<stem>_interscellar_centroids.csv"
        ),
    )
    parser.add_argument(
        "--voxel-size-um",
        nargs=3,
        type=float,
        default=None,
        metavar=("Z", "Y", "X"),
        help=(
            "Voxel size in micrometers as three values: z y x. "
            "Defaults to voxel_size_um from the combined zarr "
            f"(or {DEFAULT_VOXEL_SIZE_UM} if missing)."
        ),
    )
    args = parser.parse_args(argv)

    voxel_size = tuple(args.voxel_size_um) if args.voxel_size_um is not None else None
    try:
        compute_interscellar_centroids_3d(
            combined_zarr=args.combined_zarr,
            output_csv=args.output_csv,
            voxel_size_um=voxel_size,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

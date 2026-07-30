"""Visualize included vs excluded cells from exclude_truncated output in Napari."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Set, Tuple

import numpy as np

try:
    import zarr
except ImportError:
    print("Error: zarr not installed. Install with: pip install zarr")
    sys.exit(1)

try:
    import napari
except ImportError:
    print("Error: napari not installed. Install with: pip install 'napari[all]'")
    sys.exit(1)

from .feature_extraction_3d import _find_zarr_store, _to_spatial_shape_zyx
from .exclude_truncated import _pick_segmentation_label_array


def _find_file(filename: str, script_dir: str) -> str:
    possible_paths = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(os.path.dirname(script_dir), filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return filename


def _load_excluded_ids(path: str) -> Set[int]:
    ids: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower() in {"cell_id", "id"}:
                continue
            # support CSV rows like "123" or "123,..."
            token = line.split(",")[0].strip()
            if token:
                ids.add(int(token))
    return ids


def _as_zyx_labels(labels_arr: Any) -> np.ndarray:
    """Materialize a ZYX label volume from a 3D–5D zarr/numpy array."""
    arr = np.asarray(labels_arr)
    if arr.ndim == 5:
        print(f"Label array shape (5D): {arr.shape}; using [0, 0, ...]")
        return np.asarray(arr[0, 0])
    if arr.ndim == 4:
        print(f"Label array shape (4D): {arr.shape}; using [0, ...]")
        return np.asarray(arr[0])
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 3D–5D labels, got shape={arr.shape}")


def _split_included_excluded(
    labels_zyx: np.ndarray, excluded_ids: Set[int]
) -> Tuple[np.ndarray, np.ndarray]:
    if not excluded_ids:
        print("Warning: excluded ID list is empty; excluded layer will be empty.")
        excluded_vol = np.zeros_like(labels_zyx)
        included_vol = labels_zyx.copy()
        included_vol[included_vol < 0] = 0
        return included_vol, excluded_vol

    excluded_arr = np.fromiter(excluded_ids, dtype=labels_zyx.dtype)
    excl_mask = np.isin(labels_zyx, excluded_arr)
    excluded_vol = np.where(excl_mask, labels_zyx, 0).astype(labels_zyx.dtype, copy=False)
    included_vol = labels_zyx.copy()
    included_vol[excl_mask] = 0
    included_vol[included_vol < 0] = 0
    return included_vol, excluded_vol


def _write_label_zarr(
    path: str,
    labels_zyx: np.ndarray,
    description: str,
    n_cells: int,
    source_segmentation: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)

    root = zarr.open(path, mode="w")
    chunks = (
        1,
        1,
        min(64, int(labels_zyx.shape[0])),
        min(64, int(labels_zyx.shape[1])),
        min(64, int(labels_zyx.shape[2])),
    )
    # Prefer the package's gzip helper when available (zarr v2/v3).
    try:
        from ..core.compute_interscellar_volumes_3d import _zarr_gzip_dataset_kwargs

        comp_kwargs = _zarr_gzip_dataset_kwargs(level=6)
    except Exception:
        comp_kwargs = {}

    root.create_dataset(
        "0",
        data=labels_zyx[None, None, ...],
        chunks=chunks,
        **comp_kwargs,
    )
    root.attrs["description"] = description
    root.attrs["shape"] = (1, 1) + tuple(int(x) for x in labels_zyx.shape)
    root.attrs["dtype"] = str(labels_zyx.dtype)
    root.attrs["n_cells"] = int(n_cells)
    root.attrs["source_segmentation"] = source_segmentation
    root.attrs["axes"] = ["t", "c", "z", "y", "x"]
    print(f"Wrote {description} zarr: {path} (n_cells={n_cells})")


def _count_unique_labels(labels_zyx: np.ndarray) -> int:
    uniq = np.unique(labels_zyx)
    return int(np.count_nonzero(uniq))


# Napari RGBA colors (0–1). Mid-light gray for included; orange for excluded.
INCLUDED_COLOR = (0.72, 0.72, 0.72, 1.0)
EXCLUDED_COLOR = (1.0, 0.55, 0.0, 1.0)


def _labels_for_solid_color(labels_zyx: np.ndarray) -> np.ndarray:
    """Collapse to binary labels so every foreground voxel shares one Napari color."""
    return (labels_zyx > 0).astype(np.uint8)


def _solid_label_colormap(rgba: Tuple[float, float, float, float]):
    """Build a napari DirectLabelColormap for a single solid foreground color."""
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(
        color_dict={
            None: (0.0, 0.0, 0.0, 0.0),
            0: (0.0, 0.0, 0.0, 0.0),
            1: rgba,
        }
    )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a cell segmentation into included vs excluded label volumes "
            "using the excluded cell ID list from exclude-truncated, optionally "
            "write both as zarr stores, and visualize them in Napari."
        )
    )
    parser.add_argument(
        "--segmentation-zarr",
        required=True,
        help="Path to cell segmentation OME-Zarr.",
    )
    parser.add_argument(
        "--excluded-ids",
        required=True,
        help=(
            "Text/CSV of excluded cell IDs from exclude-truncated "
            "(one ID per line, or a cell_id column)."
        ),
    )
    parser.add_argument(
        "--output-included-zarr",
        default=None,
        help="Output path for included-cells zarr. Default: <stem>_included_cells.zarr",
    )
    parser.add_argument(
        "--output-excluded-zarr",
        default=None,
        help="Output path for excluded-cells zarr. Default: <stem>_excluded_cells.zarr",
    )
    parser.add_argument(
        "--skip-write-zarr",
        action="store_true",
        help="Do not write included/excluded zarr stores (view only).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Write zarrs only; do not launch Napari.",
    )
    parser.add_argument(
        "--included-opacity",
        type=float,
        default=0.5,
        help="Opacity for included-cells layer (0.0-1.0; default: 0.5).",
    )
    parser.add_argument(
        "--excluded-opacity",
        type=float,
        default=1.0,
        help="Opacity for excluded-cells layer (0.0-1.0; default: 1.0).",
    )
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    seg_path = _find_zarr_store(args.segmentation_zarr, script_dir)
    excluded_path = _find_file(args.excluded_ids, script_dir)

    if not os.path.exists(seg_path):
        print(f"Error: segmentation zarr not found: {seg_path}")
        sys.exit(1)
    if not os.path.exists(excluded_path):
        print(f"Error: excluded IDs file not found: {excluded_path}")
        sys.exit(1)

    print(f"Segmentation zarr: {seg_path}")
    print(f"Excluded IDs file: {excluded_path}")

    excluded_ids = _load_excluded_ids(excluded_path)
    print(f"Loaded {len(excluded_ids)} excluded cell IDs")

    seg_zarr = zarr.open(seg_path, mode="r")
    key_name, labels_arr = _pick_segmentation_label_array(seg_zarr)
    zyx_shape = _to_spatial_shape_zyx(labels_arr)
    print(f"Using label volume '{key_name}' with spatial shape (Z, Y, X)={zyx_shape}")

    print("Loading full label volume into memory...")
    labels_zyx = _as_zyx_labels(labels_arr)
    print(f"Loaded labels shape: {labels_zyx.shape}, dtype: {labels_zyx.dtype}")

    print("Splitting included vs excluded volumes...")
    included_vol, excluded_vol = _split_included_excluded(labels_zyx, excluded_ids)
    n_included = _count_unique_labels(included_vol)
    n_excluded = _count_unique_labels(excluded_vol)
    print(f"Included unique cell labels: {n_included}")
    print(f"Excluded unique cell labels present in volume: {n_excluded}")

    stem = Path(seg_path).name
    if stem.endswith(".zarr"):
        stem = stem[:-5]
    base = Path(seg_path).with_name(stem)

    included_zarr_path = (
        args.output_included_zarr
        if args.output_included_zarr is not None
        else f"{base}_included_cells.zarr"
    )
    excluded_zarr_path = (
        args.output_excluded_zarr
        if args.output_excluded_zarr is not None
        else f"{base}_excluded_cells.zarr"
    )

    if not args.skip_write_zarr:
        _write_label_zarr(
            included_zarr_path,
            included_vol,
            description="Included cells (not in Z-edge exclusion list)",
            n_cells=n_included,
            source_segmentation=seg_path,
        )
        _write_label_zarr(
            excluded_zarr_path,
            excluded_vol,
            description="Excluded cells (Z-edge exclusion list from exclude-truncated)",
            n_cells=n_excluded,
            source_segmentation=seg_path,
        )

    if args.no_viewer:
        return

    print("\nLaunching Napari viewer...")
    viewer = napari.Viewer(title="Included vs Excluded Cells")
    viewer.add_labels(
        _labels_for_solid_color(included_vol),
        name="included_cells",
        opacity=args.included_opacity,
        colormap=_solid_label_colormap(INCLUDED_COLOR),
    )
    viewer.add_labels(
        _labels_for_solid_color(excluded_vol),
        name="excluded_cells",
        opacity=args.excluded_opacity,
        colormap=_solid_label_colormap(EXCLUDED_COLOR),
    )
    if included_vol.shape:
        viewer.camera.center = (
            included_vol.shape[2] / 2,
            included_vol.shape[1] / 2,
        )
        viewer.camera.zoom = 0.5

    print("Viewer launched successfully!")
    print("Included = mid-light gray; excluded = orange.")
    print("Toggle layer visibility to compare included vs excluded cells.")
    napari.run()


if __name__ == "__main__":
    main()

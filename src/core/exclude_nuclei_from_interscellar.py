from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

try:
    from zarr.hierarchy import Group as ZarrGroup
except ImportError:
    ZarrGroup = zarr.Group


def _zarr_gzip_dataset_kwargs(
    level: int = 6, copy_compressors_from: Any = None
) -> Dict[str, Any]:
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import GzipCodec

        if copy_compressors_from is not None:
            comps = getattr(copy_compressors_from, "compressors", None)
            if comps:
                return {"compressors": list(comps)}
        return {"compressors": [GzipCodec(level=level)]}
    # zarr v2: prefer the source compressor object. Reconstructing via
    # compression='blosc' + compression_opts=<int> breaks Blosc encode
    # ("expected bytes, int found").
    if copy_compressors_from is not None:
        compressor = getattr(copy_compressors_from, "compressor", None)
        if compressor is not None:
            return {"compressor": compressor}
        compression = getattr(copy_compressors_from, "compression", None)
        if compression:
            kwargs: Dict[str, Any] = {"compression": compression}
            co = getattr(copy_compressors_from, "compression_opts", None)
            if co is not None:
                kwargs["compression_opts"] = co
            return kwargs
    return {"compression": "gzip", "compression_opts": level}


def _resolve_interscellar_read_dataset_name(zarr_group: Any) -> Optional[str]:
    if "0" in zarr_group:
        return "0"
    if "interscellar_meshes" in zarr_group:
        return "interscellar_meshes"
    return None


def _load_label_volume_3d_from_zarr(
    zarr_path: str, preferred_keys: Optional[List[str]] = None
) -> Tuple[Any, Any, str]:

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    preferred = preferred_keys or ["labels", "0", "interscellar_meshes"]

    def _as_3d(arr: Any) -> Any:
        if arr.ndim == 5:
            return arr[0, 0]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected 3D or 5D array, got ndim={arr.ndim}")

    for key in preferred:
        if key == "0" and "0" in root:
            node = root["0"]
            if isinstance(node, ZarrGroup):
                if "0" in node:
                    return root, _as_3d(node["0"]), "0"
            elif hasattr(node, "ndim") and node.ndim >= 3:
                return root, _as_3d(node), "0"
        elif key in root:
            node = root[key]
            if hasattr(node, "ndim") and node.ndim >= 3:
                return root, _as_3d(node), key

    for key in root.keys():
        node = root[key]
        if hasattr(node, "ndim") and node.ndim >= 3:
            return root, _as_3d(node), key
        if isinstance(node, ZarrGroup) and "0" in node and hasattr(node["0"], "ndim"):
            return root, _as_3d(node["0"]), key

    raise ValueError(
        f"Could not find 3D label volume in {zarr_path}. Keys: {list(root.keys())}"
    )


def _default_output_path(interscellar_volumes_zarr: str) -> str:
    interscellar_dir = os.path.dirname(interscellar_volumes_zarr) or "."
    basename = os.path.basename(interscellar_volumes_zarr)
    base_name = os.path.splitext(basename)[0]

    while base_name.endswith("_interscellar_volumes"):
        base_name = base_name[: -len("_interscellar_volumes")]

    if base_name.endswith("_interscellar_nuclei_excluded"):
        output_basename = base_name + ".zarr"
    else:
        output_basename = base_name + "_interscellar_nuclei_excluded.zarr"

    return os.path.join(interscellar_dir, output_basename)


def create_interscellar_nuclei_excluded_volumes_zarr(
    cell_segmentation_zarr: str,
    nuclei_segmentation_zarr: str,
    interscellar_volumes_zarr: str,
    output_zarr_path: str,
) -> np.ndarray:

    print("Creating interscellar volumes with nuclei excluded (cookie cutter)...")
    print(f"  Cell segmentation: {cell_segmentation_zarr}")
    print(f"  Nuclei segmentation: {nuclei_segmentation_zarr}")
    print(f"  Interscellar volumes: {interscellar_volumes_zarr}")
    print(f"  Output: {output_zarr_path}")

    print("Loading cell segmentation...")
    cell_zarr, cell_seg, _ = _load_label_volume_3d_from_zarr(
        cell_segmentation_zarr, preferred_keys=["labels", "0"]
    )
    cell_seg = np.asarray(cell_seg)
    print(f"  Cell segmentation shape: {cell_seg.shape}, dtype: {cell_seg.dtype}")

    print("Loading nuclei segmentation...")
    _, nuclei_seg, _ = _load_label_volume_3d_from_zarr(
        nuclei_segmentation_zarr, preferred_keys=["labels", "0"]
    )
    nuclei_seg = np.asarray(nuclei_seg)
    print(f"  Nuclei segmentation shape: {nuclei_seg.shape}, dtype: {nuclei_seg.dtype}")

    print("Loading interscellar volumes...")
    interscellar_zarr = zarr.open(interscellar_volumes_zarr, mode="r")
    interscellar_key = _resolve_interscellar_read_dataset_name(interscellar_zarr)
    if interscellar_key is None:
        available_keys = list(interscellar_zarr.keys())
        raise ValueError(
            f"Could not find either '0' or 'interscellar_meshes' key in "
            f"{interscellar_volumes_zarr}. Available keys: {available_keys}"
        )
    interscellar_arr = interscellar_zarr[interscellar_key]
    if interscellar_arr.ndim == 5:
        interscellar_volumes = np.asarray(interscellar_arr[0, 0])
    else:
        interscellar_volumes = np.asarray(interscellar_arr)
    print(
        f"  Interscellar volumes shape: {interscellar_volumes.shape}, "
        f"dtype: {interscellar_volumes.dtype}, key: '{interscellar_key}'"
    )

    if cell_seg.shape != interscellar_volumes.shape:
        raise ValueError(
            f"Shape mismatch: cell {cell_seg.shape} vs interscellar {interscellar_volumes.shape}"
        )
    if nuclei_seg.shape != interscellar_volumes.shape:
        raise ValueError(
            f"Shape mismatch: nuclei {nuclei_seg.shape} vs interscellar {interscellar_volumes.shape}"
        )

    print("Applying cookie cutter subtraction (zero interscellar where nuclei > 0)...")
    nuclei_mask = nuclei_seg > 0
    result = np.where(nuclei_mask, 0, interscellar_volumes).astype(np.uint32)

    print("Creating output zarr file...")
    output_key = interscellar_key
    chunks = (1, 1, 64, 64, 64)
    comp_kwargs = _zarr_gzip_dataset_kwargs(level=6)
    if hasattr(interscellar_arr, "chunks") and interscellar_arr.chunks:
        if interscellar_arr.ndim == 5:
            chunks = (1, 1) + tuple(interscellar_arr.chunks[-3:])
        elif interscellar_arr.ndim == 3:
            chunks = (1, 1) + tuple(interscellar_arr.chunks)
        comp_kwargs = _zarr_gzip_dataset_kwargs(
            level=6, copy_compressors_from=interscellar_arr
        )

    output_zarr = zarr.open(output_zarr_path, mode="w")
    result_5d = result[None, None, :, :, :]
    print(f"  Creating dataset '{output_key}' with shape {result_5d.shape}")

    try:
        if int(zarr.__version__.split(".")[0]) >= 3:
            ds = output_zarr.create_dataset(
                output_key,
                shape=result_5d.shape,
                dtype=result_5d.dtype,
                chunks=chunks,
                fill_value=0,
                **comp_kwargs,
            )
            ds[:] = result_5d
        else:
            output_zarr.create_dataset(
                output_key,
                data=result_5d,
                chunks=chunks,
                **comp_kwargs,
            )
    except Exception as e:
        raise RuntimeError(f"Failed to create zarr dataset '{output_key}': {e}") from e

    print("Copying metadata from interscellar volumes...")
    try:
        for key, value in interscellar_zarr.attrs.items():
            try:
                output_zarr.attrs[key] = value
            except Exception as e:
                print(f"Warning: Could not copy metadata key '{key}': {e}")
    except Exception as e:
        print(f"Warning: Could not copy metadata: {e}")

    output_zarr.attrs["description"] = (
        "Interscellar volumes with overlapping nuclei voxels excluded (cookie cutter)"
    )
    output_zarr.attrs["subtracted_volumes"] = "nuclei_segmentation"
    output_zarr.attrs["cell_segmentation_zarr"] = os.path.abspath(cell_segmentation_zarr)
    output_zarr.attrs["nuclei_segmentation_zarr"] = os.path.abspath(nuclei_segmentation_zarr)
    output_zarr.attrs["source_interscellar_zarr"] = os.path.abspath(interscellar_volumes_zarr)

    original_interscellar_voxels = int((interscellar_volumes > 0).sum())
    remaining_interscellar_voxels = int((result > 0).sum())
    nuclei_voxels = int(nuclei_mask.sum())
    overlapping_voxels = int(((interscellar_volumes > 0) & nuclei_mask).sum())

    output_zarr.attrs["original_interscellar_voxels"] = original_interscellar_voxels
    output_zarr.attrs["remaining_interscellar_voxels"] = remaining_interscellar_voxels
    output_zarr.attrs["nuclei_voxels"] = nuclei_voxels
    output_zarr.attrs["overlapping_voxels_removed"] = overlapping_voxels
    output_zarr.attrs["n_cells"] = int(len(np.unique(cell_seg)) - 1)

    if not os.path.exists(output_zarr_path):
        raise RuntimeError(f"Output zarr file was not created: {output_zarr_path}")

    try:
        verify_zarr = zarr.open(output_zarr_path, mode="r")
        if output_key not in verify_zarr:
            raise RuntimeError(
                f"Output zarr created but missing expected key '{output_key}'"
            )
        if verify_zarr[output_key].size == 0:
            raise RuntimeError(f"Output zarr dataset '{output_key}' is empty")
        print("Verified output zarr file created successfully")
        del verify_zarr
    except Exception as e:
        print(f"Warning: Could not verify output zarr: {e}")

    del output_zarr, cell_zarr, interscellar_zarr

    print(f"Created nuclei-excluded interscellar zarr with shape {result.shape}")
    print(f"  Original interscellar voxels: {original_interscellar_voxels:,}")
    print(f"  Nuclei voxels: {nuclei_voxels:,}")
    print(f"  Overlapping voxels removed: {overlapping_voxels:,}")
    print(f"  Remaining interscellar voxels: {remaining_interscellar_voxels:,}")
    if original_interscellar_voxels > 0:
        print(
            f"  Fraction removed: "
            f"{overlapping_voxels / original_interscellar_voxels * 100:.1f}% of interscellar volume"
        )

    return result


def exclude_nuclei_from_interscellar(
    cell_segmentation_zarr: str,
    nuclei_segmentation_zarr: str,
    interscellar_volumes_zarr: str,
    output_zarr_path: Optional[str] = None,
):
    if output_zarr_path is None:
        output_zarr_path = _default_output_path(interscellar_volumes_zarr)

    print("=" * 60)
    print("InterSCellar: Exclude nuclei from interscellar volumes")
    print("=" * 60)
    print(f"Cell segmentation:     {cell_segmentation_zarr}")
    print(f"Nuclei segmentation:   {nuclei_segmentation_zarr}")
    print(f"Interscellar volumes:  {interscellar_volumes_zarr}")
    print(f"Output zarr:           {output_zarr_path}")

    for path, label in (
        (cell_segmentation_zarr, "Cell segmentation zarr"),
        (nuclei_segmentation_zarr, "Nuclei segmentation zarr"),
        (interscellar_volumes_zarr, "Interscellar volumes zarr"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    result = create_interscellar_nuclei_excluded_volumes_zarr(
        cell_segmentation_zarr=cell_segmentation_zarr,
        nuclei_segmentation_zarr=nuclei_segmentation_zarr,
        interscellar_volumes_zarr=interscellar_volumes_zarr,
        output_zarr_path=output_zarr_path,
    )
    print(f"\nWrote nuclei-excluded interscellar volumes to: {output_zarr_path}")
    return result, output_zarr_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract nuclei segmentation masks from interscellar volumes using "
            "cookie-cutter subtraction (same approach as cell-only volumes). "
            "Any voxel where the nuclei mask is > 0 is zeroed in the interscellar "
            "volume. Output is written next to the interscellar zarr by default. "
            "This is a standalone post-processing step and does not affect "
            "interscellar volume computation."
        )
    )
    parser.add_argument(
        "--cell-segmentation-zarr",
        required=True,
        help="Path to cell segmentation zarr (used for shape/alignment checks).",
    )
    parser.add_argument(
        "--nuclei-segmentation-zarr",
        required=True,
        help="Path to nuclei segmentation zarr (cookie-cutter mask).",
    )
    parser.add_argument(
        "--interscellar-zarr",
        required=True,
        help="Path to interscellar volumes zarr.",
    )
    parser.add_argument(
        "--output-zarr",
        default=None,
        help=(
            "Output zarr path. Defaults to "
            "<interscellar_dir>/<stem>_interscellar_nuclei_excluded.zarr"
        ),
    )
    args = parser.parse_args(argv)

    try:
        exclude_nuclei_from_interscellar(
            cell_segmentation_zarr=args.cell_segmentation_zarr,
            nuclei_segmentation_zarr=args.nuclei_segmentation_zarr,
            interscellar_volumes_zarr=args.interscellar_zarr,
            output_zarr_path=args.output_zarr,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

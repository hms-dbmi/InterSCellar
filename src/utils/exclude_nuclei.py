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

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])

# Keys tried in order when the caller does not name one. Covers the layouts this
# package writes: OME-Zarr multiscale ("0"), plain labels, and interscellar meshes.
DEFAULT_KEY_PREFERENCE = ["labels", "0", "interscellar_meshes"]


def _zarr_gzip_dataset_kwargs(
    level: int = 6, copy_compressors_from: Any = None
) -> Dict[str, Any]:
    if _ZARR_MAJOR >= 3:
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


def _create_array(store: Any, name: str, shape, dtype, chunks, fill_value=0, **kwargs):
    if hasattr(store, "create_array"):
        try:
            return store.create_array(
                name, shape=shape, dtype=dtype, chunks=chunks,
                fill_value=fill_value, **kwargs,
            )
        except TypeError:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
    return store.create_dataset(
        name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill_value, **kwargs
    )


def _find_label_array(
    zarr_path: str,
    dataset_key: Optional[str] = None,
    preferred_keys: Optional[List[str]] = None,
) -> Tuple[Any, Any, str]:
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")

    root = zarr.open(zarr_path, mode="r")

    def _accept(node: Any) -> bool:
        return hasattr(node, "ndim") and node.ndim in (3, 5)

    if dataset_key:
        node = root
        for part in [p for p in dataset_key.split("/") if p]:
            if part not in node:
                raise ValueError(f"Key '{dataset_key}' not found in {zarr_path}")
            node = node[part]
        if not _accept(node):
            raise ValueError(
                f"Key '{dataset_key}' in {zarr_path} is not a 3D or 5D array "
                f"(ndim={getattr(node, 'ndim', None)})"
            )
        return root, node, dataset_key

    for key in (preferred_keys or DEFAULT_KEY_PREFERENCE):
        if key not in root:
            continue
        node = root[key]
        if _accept(node):
            return root, node, key
        # OME-Zarr style nesting: "0/0"
        if isinstance(node, ZarrGroup) and "0" in node and _accept(node["0"]):
            return root, node["0"], f"{key}/0"

    for key in list(root.keys()):
        node = root[key]
        if _accept(node):
            return root, node, key
        if isinstance(node, ZarrGroup) and "0" in node and _accept(node["0"]):
            return root, node["0"], f"{key}/0"

    raise ValueError(
        f"Could not find a 3D or 5D label array in {zarr_path}. Keys: {list(root.keys())}"
    )


def _as_3d(array_node: Any) -> np.ndarray:
    if array_node.ndim == 5:
        return np.asarray(array_node[0, 0])
    return np.asarray(array_node)


def _default_output_path(input_zarr: str) -> str:
    directory = os.path.dirname(input_zarr) or "."
    base = os.path.splitext(os.path.basename(input_zarr))[0]

    for suffix in ("_interscellar_volumes", "_volumes", "_nuclei_excluded"):
        while base.endswith(suffix):
            base = base[: -len(suffix)]

    return os.path.join(directory, f"{base}_nuclei_excluded.zarr")


def subtract_nuclei_from_volume(
    input_zarr: str,
    nuclei_segmentation_zarr: str,
    output_zarr_path: str,
    input_key: Optional[str] = None,
    nuclei_key: Optional[str] = None,
    reference_zarr: Optional[str] = None,
) -> np.ndarray:
    print("Subtracting nuclei from label volume (cookie cutter)...")
    print(f"  Input volume:        {input_zarr}")
    print(f"  Nuclei segmentation: {nuclei_segmentation_zarr}")
    if reference_zarr:
        print(f"  Reference (check):   {reference_zarr}")
    print(f"  Output:              {output_zarr_path}")

    input_root, input_node, input_key_used = _find_label_array(input_zarr, input_key)
    volume = _as_3d(input_node)
    print(f"  Input:  shape {volume.shape}, dtype {volume.dtype}, key '{input_key_used}'")

    _, nuclei_node, nuclei_key_used = _find_label_array(
        nuclei_segmentation_zarr, nuclei_key
    )
    nuclei = _as_3d(nuclei_node)
    print(f"  Nuclei: shape {nuclei.shape}, dtype {nuclei.dtype}, key '{nuclei_key_used}'")

    if nuclei.shape != volume.shape:
        raise ValueError(
            f"Shape mismatch: nuclei {nuclei.shape} vs input volume {volume.shape}. "
            "Both must be on the same grid; check for a cropped or upsampled copy."
        )

    if reference_zarr:
        _, ref_node, ref_key = _find_label_array(reference_zarr, None)
        ref_shape = (
            tuple(int(v) for v in ref_node.shape[-3:])
            if ref_node.ndim == 5
            else tuple(int(v) for v in ref_node.shape)
        )
        if ref_shape != volume.shape:
            raise ValueError(
                f"Shape mismatch: reference {ref_shape} (key '{ref_key}') vs input {volume.shape}"
            )
        print(f"  Reference shape {ref_shape} matches (key '{ref_key}')")

    nuclei_mask = nuclei > 0
    result = np.where(nuclei_mask, 0, volume).astype(volume.dtype, copy=False)

    original_voxels = int((volume > 0).sum())
    remaining_voxels = int((result > 0).sum())
    nuclei_voxels = int(nuclei_mask.sum())
    removed_voxels = int(((volume > 0) & nuclei_mask).sum())
    labels_before = int(np.unique(volume).size - (1 if (volume == 0).any() else 0))
    labels_after = int(np.unique(result).size - (1 if (result == 0).any() else 0))

    # Preserve the input layout: same key, same dimensionality, same chunking.
    write_5d = input_node.ndim == 5
    data = result[None, None, :, :, :] if write_5d else result
    src_chunks = getattr(input_node, "chunks", None)
    if src_chunks:
        spatial = tuple(int(c) for c in src_chunks[-3:])
    else:
        spatial = tuple(min(64, n) for n in result.shape)
    chunks = ((1, 1) + spatial) if write_5d else spatial
    comp_kwargs = _zarr_gzip_dataset_kwargs(level=6, copy_compressors_from=input_node)

    output_zarr = zarr.open(output_zarr_path, mode="w")
    leaf_key = input_key_used
    print(f"  Creating dataset '{leaf_key}' with shape {data.shape}, chunks {chunks}")
    try:
        _create_array(
            output_zarr, leaf_key, data.shape, data.dtype, chunks, 0, **comp_kwargs
        )[:] = data
    except Exception as exc:
        raise RuntimeError(f"Failed to create zarr dataset '{leaf_key}': {exc}") from exc

    try:
        for key, value in input_root.attrs.items():
            try:
                output_zarr.attrs[key] = value
            except Exception as exc:
                print(f"Warning: could not copy metadata key '{key}': {exc}")
    except Exception as exc:
        print(f"Warning: could not copy metadata: {exc}")

    output_zarr.attrs["description"] = (
        "Label volume with nuclei voxels excluded (cookie cutter). Labels are unchanged; "
        "only nuclei-covered voxels are zeroed."
    )
    output_zarr.attrs["subtracted_volumes"] = "nuclei_segmentation"
    output_zarr.attrs["source_zarr"] = os.path.abspath(input_zarr)
    output_zarr.attrs["source_dataset_key"] = input_key_used
    output_zarr.attrs["nuclei_segmentation_zarr"] = os.path.abspath(nuclei_segmentation_zarr)
    if reference_zarr:
        output_zarr.attrs["reference_zarr"] = os.path.abspath(reference_zarr)
    output_zarr.attrs["original_voxels"] = original_voxels
    output_zarr.attrs["remaining_voxels"] = remaining_voxels
    output_zarr.attrs["nuclei_voxels"] = nuclei_voxels
    output_zarr.attrs["overlapping_voxels_removed"] = removed_voxels
    output_zarr.attrs["labels_before"] = labels_before
    output_zarr.attrs["labels_after"] = labels_after

    if not os.path.exists(output_zarr_path):
        raise RuntimeError(f"Output zarr was not created: {output_zarr_path}")
    try:
        verify = zarr.open(output_zarr_path, mode="r")
        node = verify
        for part in [p for p in leaf_key.split("/") if p]:
            node = node[part]
        if node.size == 0:
            raise RuntimeError(f"Output dataset '{leaf_key}' is empty")
        print("Verified output zarr")
    except Exception as exc:
        print(f"Warning: could not verify output zarr: {exc}")

    print(f"Wrote nuclei-excluded volume, shape {result.shape}")
    print(f"  Labeled voxels before: {original_voxels:,}")
    print(f"  Nuclei voxels:         {nuclei_voxels:,}")
    print(f"  Voxels removed:        {removed_voxels:,}")
    print(f"  Labeled voxels after:  {remaining_voxels:,}")
    if original_voxels:
        print(f"  Fraction removed:      {removed_voxels / original_voxels * 100:.1f}%")
    if labels_after < labels_before:
        print(
            f"  Note: {labels_before - labels_after} label(s) were erased entirely "
            f"({labels_before} -> {labels_after})"
        )

    return result


def exclude_nuclei(
    input_zarr: str,
    nuclei_segmentation_zarr: str,
    output_zarr_path: Optional[str] = None,
    input_key: Optional[str] = None,
    nuclei_key: Optional[str] = None,
    reference_zarr: Optional[str] = None,
):
    """Subtract nuclei from any label zarr. Returns ``(result_array, output_path)``."""
    if output_zarr_path is None:
        output_zarr_path = _default_output_path(input_zarr)

    print("=" * 60)
    print("InterSCellar: Exclude nuclei from a label volume")
    print("=" * 60)

    for path, label in (
        (input_zarr, "Input zarr"),
        (nuclei_segmentation_zarr, "Nuclei segmentation zarr"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if reference_zarr and not os.path.exists(reference_zarr):
        raise FileNotFoundError(f"Reference zarr not found: {reference_zarr}")

    result = subtract_nuclei_from_volume(
        input_zarr=input_zarr,
        nuclei_segmentation_zarr=nuclei_segmentation_zarr,
        output_zarr_path=output_zarr_path,
        input_key=input_key,
        nuclei_key=nuclei_key,
        reference_zarr=reference_zarr,
    )
    print(f"\nWrote to: {output_zarr_path}")
    return result, output_zarr_path


# Backwards-compatible aliases for the previous interscellar-specific API.
def exclude_nuclei_from_interscellar(
    cell_segmentation_zarr: str,
    nuclei_segmentation_zarr: str,
    interscellar_volumes_zarr: str,
    output_zarr_path: Optional[str] = None,
):
    """Deprecated. Kept so existing callers keep working; prefer ``exclude_nuclei``."""
    return exclude_nuclei(
        input_zarr=interscellar_volumes_zarr,
        nuclei_segmentation_zarr=nuclei_segmentation_zarr,
        output_zarr_path=output_zarr_path,
        reference_zarr=cell_segmentation_zarr,
    )


create_interscellar_nuclei_excluded_volumes_zarr = subtract_nuclei_from_volume


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract a nuclei segmentation from any label volume using cookie-cutter "
            "subtraction. The input may be an interscellar volumes zarr, a cell "
            "segmentation, a cell-only volume, or any other label zarr: every voxel "
            "where the nuclei mask is > 0 is zeroed and all other labels are kept. "
            "Output is written next to the input by default."
        )
    )
    parser.add_argument(
        "--input-zarr",
        help="Label zarr to carve: interscellar volumes, cell segmentation, or any other.",
    )
    parser.add_argument(
        "--nuclei-segmentation-zarr",
        required=True,
        help="Nuclei segmentation zarr, used as the cookie-cutter mask.",
    )
    parser.add_argument(
        "--output-zarr",
        default=None,
        help="Output zarr path. Defaults to <input_dir>/<stem>_nuclei_excluded.zarr",
    )
    parser.add_argument(
        "--input-key",
        default=None,
        help="Dataset key inside --input-zarr. Auto-detected if omitted "
             "(tries labels, 0, interscellar_meshes, then any 3D/5D array).",
    )
    parser.add_argument(
        "--nuclei-key",
        default=None,
        help="Dataset key inside the nuclei zarr. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--reference-zarr",
        default=None,
        help="Optional extra zarr used only for a shape/alignment check.",
    )
    # Accepted for backwards compatibility with the previous CLI.
    parser.add_argument("--interscellar-zarr", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cell-segmentation-zarr", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    input_zarr = args.input_zarr or args.interscellar_zarr
    if not input_zarr:
        parser.error("provide --input-zarr (or the legacy --interscellar-zarr)")
    reference_zarr = args.reference_zarr or args.cell_segmentation_zarr

    try:
        exclude_nuclei(
            input_zarr=input_zarr,
            nuclei_segmentation_zarr=args.nuclei_segmentation_zarr,
            output_zarr_path=args.output_zarr,
            input_key=args.input_key,
            nuclei_key=args.nuclei_key,
            reference_zarr=reference_zarr,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

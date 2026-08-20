from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import zarr
from tqdm import tqdm

from .exclude_truncated import _pick_segmentation_label_array
from .feature_extraction_3d import (
    _find_zarr_store,
    _read_label_bbox_zyx,
    _to_spatial_shape_zyx,
)


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


def _spatial_z_chunk(arr: Any, default: int = 64) -> int:
    chunks = getattr(arr, "chunks", None)
    if not chunks:
        return default
    nd = int(getattr(arr, "ndim", 0))
    if nd == 5:
        return max(1, int(chunks[2]))
    if nd == 4:
        return max(1, int(chunks[1]))
    if nd == 3:
        return max(1, int(chunks[0]))
    return default


def _output_chunks(arr: Any, zyx_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int]:
    z, y, x = zyx_shape
    chunks = getattr(arr, "chunks", None)
    if chunks:
        nd = int(getattr(arr, "ndim", 0))
        if nd == 5:
            return (1, 1, min(int(chunks[2]), z), min(int(chunks[3]), y), min(int(chunks[4]), x))
        if nd == 4:
            return (1, 1, min(int(chunks[1]), z), min(int(chunks[2]), y), min(int(chunks[3]), x))
        if nd == 3:
            return (1, 1, min(int(chunks[0]), z), min(int(chunks[1]), y), min(int(chunks[2]), x))
    return (1, 1, min(64, z), min(64, y), min(64, x))


def _iter_z_slabs(z_size: int, z_chunk: int) -> Iterable[Tuple[int, int]]:
    for z0 in range(0, z_size, z_chunk):
        yield z0, min(z0 + z_chunk, z_size)


def _default_output_path(cell_only_zarr: str) -> str:
    cell_dir = os.path.dirname(cell_only_zarr) or "."
    basename = os.path.basename(cell_only_zarr.rstrip(os.sep))
    stem = basename[:-5] if basename.endswith(".zarr") else os.path.splitext(basename)[0]
    if stem.endswith("_cell_only_volumes"):
        stem = stem[: -len("_cell_only_volumes")]
    return os.path.join(cell_dir, f"{stem}_combined_volumes.zarr")


def _choose_output_dtype(max_id: int) -> np.dtype:
    if max_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _copy_group_attrs(src: Any, dst: Any, skip: Optional[Set[str]] = None) -> None:
    skip = skip or set()
    try:
        items = dict(src.attrs)
    except Exception as exc:
        print(f"Warning: Could not read source zarr attrs: {exc}")
        return
    for key, value in items.items():
        if key in skip:
            continue
        try:
            dst.attrs[key] = value
        except Exception as exc:
            print(f"Warning: Could not copy metadata key '{key}': {exc}")


def _create_output_dataset(
    output_zarr: Any,
    key: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunks: Tuple[int, ...],
    comp_kwargs: Dict[str, Any],
) -> Any:
    return output_zarr.create_dataset(
        key,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=0,
        **comp_kwargs,
    )


def _write_id_map(output_zarr: Any, original_ids: np.ndarray, remapped_ids: np.ndarray) -> None:
    data = np.column_stack((original_ids, remapped_ids))
    comp_kwargs = _zarr_gzip_dataset_kwargs(level=6)
    if int(zarr.__version__.split(".")[0]) >= 3:
        ds = output_zarr.create_dataset(
            "interscellar_id_map",
            shape=data.shape,
            dtype=data.dtype,
            chunks=data.shape,
            fill_value=0,
            **comp_kwargs,
        )
        ds[:] = data
        return
    output_zarr.create_dataset(
        "interscellar_id_map",
        data=data,
        **comp_kwargs,
    )


def combine_cell_and_interscellar_volumes_3d(
    cell_only_zarr: str,
    interscellar_zarr: str,
    output_zarr_path: Optional[str] = None,
) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cell_only_path = _find_zarr_store(cell_only_zarr, script_dir)
    interscellar_path = _find_zarr_store(interscellar_zarr, script_dir)

    if not os.path.exists(cell_only_path):
        raise FileNotFoundError(f"Cell-only zarr not found: {cell_only_path}")
    if not os.path.exists(interscellar_path):
        raise FileNotFoundError(f"Interscellar zarr not found: {interscellar_path}")

    if output_zarr_path is None:
        output_zarr_path = _default_output_path(cell_only_path)
    output_zarr_path = os.path.abspath(output_zarr_path)

    print("Loading zarr files...")
    print(f"  Cell-only zarr: {cell_only_path}")
    print(f"  Interscellar zarr: {interscellar_path}")

    cell_root = zarr.open(cell_only_path, mode="r")
    inter_root = zarr.open(interscellar_path, mode="r")
    cell_key, cell_arr = _pick_segmentation_label_array(cell_root)
    inter_key, inter_arr = _pick_segmentation_label_array(inter_root)

    cell_shape = _to_spatial_shape_zyx(cell_arr)
    inter_shape = _to_spatial_shape_zyx(inter_arr)
    print(f"  Cell-only key '{cell_key}' shape (Z, Y, X): {cell_shape}")
    print(f"  Interscellar key '{inter_key}' shape (Z, Y, X): {inter_shape}")
    if cell_shape != inter_shape:
        raise ValueError(
            f"Shape mismatch: cell-only {cell_shape} vs interscellar {inter_shape}"
        )

    z_size, y_size, x_size = cell_shape
    z_chunk = min(_spatial_z_chunk(cell_arr), _spatial_z_chunk(inter_arr))

    print("Scanning IDs...")
    max_cell_id = 0
    unique_cell_ids: Set[int] = set()
    unique_inter_ids: Set[int] = set()
    n_cell_voxels = 0
    n_inter_voxels = 0
    n_overlap_voxels = 0

    for z0, z1 in tqdm(
        list(_iter_z_slabs(z_size, z_chunk)),
        desc="Pass 1/2: scan IDs",
        unit="slab",
    ):
        cell_slab = np.asarray(_read_label_bbox_zyx(cell_arr, z0, z1, 0, y_size, 0, x_size))
        inter_slab = np.asarray(_read_label_bbox_zyx(inter_arr, z0, z1, 0, y_size, 0, x_size))
        if cell_slab.size:
            max_cell_id = max(max_cell_id, int(cell_slab.max()))
            cell_present = cell_slab[cell_slab > 0]
            if cell_present.size:
                unique_cell_ids.update(np.unique(cell_present).tolist())
                n_cell_voxels += int(cell_present.size)
        inter_present = inter_slab[inter_slab > 0]
        if inter_present.size:
            unique_inter_ids.update(np.unique(inter_present).tolist())
            n_inter_voxels += int(inter_present.size)
        n_overlap_voxels += int(((cell_slab > 0) & (inter_slab > 0)).sum())

    original_inter_ids = np.fromiter(
        sorted(int(i) for i in unique_inter_ids if int(i) > 0),
        dtype=np.uint64,
    )
    n_inter_ids = int(original_inter_ids.size)
    id_start = int(max_cell_id) + 1
    remapped_inter_ids = np.arange(id_start, id_start + n_inter_ids, dtype=np.uint64)
    max_combined_id = int(remapped_inter_ids[-1]) if n_inter_ids else int(max_cell_id)
    out_dtype = _choose_output_dtype(max_combined_id)

    lut = None
    if n_inter_ids:
        lut = np.zeros(int(original_inter_ids.max()) + 1, dtype=np.uint64)
        lut[original_inter_ids] = remapped_inter_ids

    print(f"  Cell-only unique IDs: {len(unique_cell_ids)}")
    print(f"  Last cell-only ID: {max_cell_id}")
    print(f"  Interscellar unique IDs: {n_inter_ids}")
    print(f"  Remapped interscellar ID start: {id_start}")
    if n_inter_ids:
        print(f"  Remapped interscellar ID end: {max_combined_id}")
    if n_overlap_voxels:
        print(
            f"  Warning: {n_overlap_voxels:,} voxels are labeled in both volumes; "
            "interscellar IDs will overwrite cell-only IDs at those voxels."
        )

    output_parent = Path(output_zarr_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path)

    chunks = _output_chunks(cell_arr, cell_shape)
    comp_kwargs = _zarr_gzip_dataset_kwargs(level=6, copy_compressors_from=cell_arr)
    output_shape = (1, 1) + cell_shape
    output_key = "0"

    print(f"Writing combined zarr: {output_zarr_path}")
    output_zarr = zarr.open(output_zarr_path, mode="w")
    ds = _create_output_dataset(
        output_zarr,
        output_key,
        shape=output_shape,
        dtype=out_dtype,
        chunks=chunks,
        comp_kwargs=comp_kwargs,
    )

    for z0, z1 in tqdm(
        list(_iter_z_slabs(z_size, z_chunk)),
        desc="Pass 2/2: write combined",
        unit="slab",
    ):
        cell_slab = np.asarray(_read_label_bbox_zyx(cell_arr, z0, z1, 0, y_size, 0, x_size))
        inter_slab = np.asarray(_read_label_bbox_zyx(inter_arr, z0, z1, 0, y_size, 0, x_size))
        cell_out = cell_slab.astype(out_dtype, copy=False)
        if lut is None:
            combined = cell_out
        else:
            remapped = lut[inter_slab.astype(np.int64, copy=False)]
            combined = np.where(inter_slab > 0, remapped.astype(out_dtype, copy=False), cell_out)
        ds[0, 0, z0:z1, :, :] = combined

    if n_inter_ids:
        _write_id_map(output_zarr, original_inter_ids, remapped_inter_ids)

    _copy_group_attrs(
        cell_root,
        output_zarr,
        skip={
            "description",
            "original_cells",
            "remaining_cells",
            "subtracted_volumes",
            "num_pairs",
        },
    )
    output_zarr.attrs["description"] = (
        "Combined cell-only and interscellar volumes; interscellar pair IDs "
        "remapped to start after the last cell-only ID"
    )
    output_zarr.attrs["axes"] = ["t", "c", "z", "y", "x"]
    output_zarr.attrs["shape"] = list(output_shape)
    output_zarr.attrs["dtype"] = str(out_dtype)
    output_zarr.attrs["coordinate_system"] = "same_as_input_segmentation"
    output_zarr.attrs["alignment_reference"] = "input_segmentation_mask"
    output_zarr.attrs["cell_only_zarr"] = os.path.abspath(cell_only_path)
    output_zarr.attrs["interscellar_zarr"] = os.path.abspath(interscellar_path)
    output_zarr.attrs["max_cell_only_id"] = int(max_cell_id)
    output_zarr.attrs["n_cell_only_ids"] = int(len(unique_cell_ids))
    output_zarr.attrs["n_interscellar_ids"] = n_inter_ids
    output_zarr.attrs["interscellar_id_start"] = id_start
    output_zarr.attrs["max_combined_id"] = int(max_combined_id)
    output_zarr.attrs["n_cell_only_voxels"] = int(n_cell_voxels)
    output_zarr.attrs["n_interscellar_voxels"] = int(n_inter_voxels)
    output_zarr.attrs["n_overlap_voxels_overwritten"] = int(n_overlap_voxels)
    if n_inter_ids:
        output_zarr.attrs["interscellar_id_map_columns"] = [
            "original_pair_id",
            "remapped_id",
        ]

    if not os.path.exists(output_zarr_path):
        raise RuntimeError(f"Output zarr file was not created: {output_zarr_path}")
    verify = zarr.open(output_zarr_path, mode="r")
    if output_key not in verify:
        raise RuntimeError(f"Output zarr created but missing expected key '{output_key}'")
    if verify[output_key].size == 0:
        raise RuntimeError(f"Output zarr dataset '{output_key}' is empty")

    print("Verified output zarr file created successfully")
    print(f"  Output shape: {tuple(verify[output_key].shape)}")
    print(f"  Cell-only voxels: {n_cell_voxels:,}")
    print(f"  Interscellar voxels: {n_inter_voxels:,}")
    print(f"  Combined unique IDs: {len(unique_cell_ids) + n_inter_ids}")
    print(f"Wrote combined volumes to: {output_zarr_path}")
    return output_zarr_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine cell-only and interscellar volume zarrs (same XYZ dimensions) "
            "into one label zarr. Cell-only IDs are preserved. Interscellar pair IDs "
            "are overwritten so they start at last_cell_only_id + 1 and occupy the "
            "IDs immediately after the cell-only ID list."
        )
    )
    parser.add_argument(
        "--cell-only-zarr",
        required=True,
        help="Path to cell-only volumes zarr from compute_cell_only_volumes_3d.",
    )
    parser.add_argument(
        "--interscellar-zarr",
        required=True,
        help="Path to interscellar volumes zarr from compute_interscellar_volumes_3d.",
    )
    parser.add_argument(
        "--output-zarr",
        default=None,
        help=(
            "Output combined zarr path. Defaults to "
            "<cell_only_dir>/<stem>_combined_volumes.zarr"
        ),
    )
    args = parser.parse_args(argv)

    try:
        combine_cell_and_interscellar_volumes_3d(
            cell_only_zarr=args.cell_only_zarr,
            interscellar_zarr=args.interscellar_zarr,
            output_zarr_path=args.output_zarr,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

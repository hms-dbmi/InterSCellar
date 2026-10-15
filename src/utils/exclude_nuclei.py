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


# Adaptive volume stores: per-pair archive

_ARCHIVE_GROUP = "pair_volumes"
_FOOTPRINT_KEYS = ("territory_a", "corridor", "territory_b")
_INTERFACE_KEYS = ("interface_a", "interface_b")
_COMPONENT_KEYS = _FOOTPRINT_KEYS + _INTERFACE_KEYS
_COUNT_FIELDS = {
    "territory_a": "n_territory_a", "corridor": "n_corridor", "territory_b": "n_territory_b",
    "interface_a": "n_interface_a", "interface_b": "n_interface_b",
}
_PREVIEW_KEYS = ("interscellar_meshes", "overlap_count")


def _has_pair_archive(root: Any) -> bool:
    return hasattr(root, "keys") and _ARCHIVE_GROUP in root and "pair_id" in root[_ARCHIVE_GROUP]


def _read_slab_3d(node: Any, z0: int, z1: int) -> np.ndarray:
    if node.ndim == 5:
        return np.asarray(node[0, 0, z0:z1])
    return np.asarray(node[z0:z1])


def _resized(group: Any, name: str, length: int) -> Any:
    array = group[name]
    result = array.resize((int(length),) + tuple(array.shape[1:]))
    return group[name] if result is None else result


def _copy_attrs(src: Any, dst: Any) -> None:
    for key, value in src.attrs.items():
        try:
            dst.attrs[key] = value
        except Exception as exc:
            print(f"Warning: could not copy metadata key '{key}': {exc}")


def _like(dst_group: Any, name: str, src: Any, shape) -> Any:
    return _create_array(
        dst_group, name, tuple(shape), src.dtype, src.chunks, 0,
        **_zarr_gzip_dataset_kwargs(level=6, copy_compressors_from=src),
    )


def _pair_geometry(flat: np.ndarray, shape, origin, scale: np.ndarray):
    # Same definitions as the adaptive pipeline's _mask_geometry.
    if flat.size == 0:
        return (np.nan,) * 3, (np.nan,) * 3, np.nan, np.nan
    coords = np.stack(np.unravel_index(flat, shape), axis=1).astype(np.float64)
    coords += np.asarray(origin, dtype=np.float64)
    centroid = coords.mean(axis=0)
    radii = np.sqrt((((coords - centroid) * scale) ** 2).sum(axis=1))
    return tuple(centroid), tuple(centroid * scale), float(radii.min()), float(radii.max())


def _exclude_nuclei_from_pair_archive(
    input_zarr: str,
    nuclei_segmentation_zarr: str,
    output_zarr_path: str,
    nuclei_key: Optional[str] = None,
    batch_rows: int = 2048,
) -> Dict[str, Any]:
    from scipy.ndimage import label

    root = zarr.open(input_zarr, mode="r")
    group = root[_ARCHIVE_GROUP]
    grid = next(
        (tuple(int(v) for v in root[k].shape[-3:]) for k in _PREVIEW_KEYS if k in root), None
    ) or tuple(int(v) for v in root.attrs["volume_shape_zyx"])
    _, nuc_node, nuc_key_used = _find_label_array(nuclei_segmentation_zarr, nuclei_key)
    if tuple(int(v) for v in nuc_node.shape[-3:]) != grid:
        raise ValueError(
            f"Shape mismatch: nuclei {tuple(nuc_node.shape[-3:])} (key '{nuc_key_used}') vs "
            f"interscellar grid {grid}. Both must be on the same grid."
        )
    scale = np.asarray(root.attrs.get("voxel_size_um", (0.56, 0.28, 0.28)), dtype=np.float64)
    stated = group.attrs.get("committed_rows")
    n_rows = int(group["pair_id"].shape[0]) if stated is None else int(stated)
    print("Excluding nuclei from every pair in the lossless pair archive")
    print(f"  Input:  {input_zarr} ({n_rows} pairs, grid {grid})")
    print(f"  Nuclei: {nuclei_segmentation_zarr} (key '{nuc_key_used}')")
    print(f"  Output: {output_zarr_path}")

    nuclei = np.zeros(grid, dtype=bool)
    for z0 in range(0, grid[0], 16):
        nuclei[z0:z0 + 16] = _read_slab_3d(nuc_node, z0, min(z0 + 16, grid[0])) > 0
    # The overlap preview is unchanged at every voxel a pair keeps (nuclei are removed from
    # all pairs alike), so it gives each pair's shared-voxel count after exclusion.
    overlap = np.asarray(root["overlap_count"][...]) if "overlap_count" in root else None

    out = zarr.open(output_zarr_path, mode="w")
    _copy_attrs(root, out)
    for key in _PREVIEW_KEYS:
        if key not in root:
            continue
        src = root[key]
        dst = _like(out, key, src, src.shape)
        for z0 in range(0, grid[0], 16):
            z1 = min(z0 + 16, grid[0])
            dst[z0:z1] = np.where(nuclei[z0:z1], 0, np.asarray(src[z0:z1]))

    rows = {
        name: np.asarray(group[name][:n_rows])
        for name in group.keys()
        if name not in _COMPONENT_KEYS and not name.endswith("_indptr")
    }
    indptr = {name: np.asarray(group[f"{name}_indptr"][: n_rows + 1]).astype(np.int64)
              for name in _COMPONENT_KEYS}
    out_group = out.create_group(_ARCHIVE_GROUP)
    out_data = {name: _like(out_group, name, group[name], group[name].shape)
                for name in _COMPONENT_KEYS}
    new_indptr = {name: np.zeros(n_rows + 1, dtype=np.uint64) for name in _COMPONENT_KEYS}
    written = {name: 0 for name in _COMPONENT_KEYS}

    updated = {
        "n_voxels": np.zeros(n_rows, dtype=np.int64),
        "n_components": np.zeros(n_rows, dtype=np.int64),
        "shared_voxels": np.zeros(n_rows, dtype=np.int64),
        "centroid_zyx_voxel": np.full((n_rows, 3), np.nan),
        "centroid_zyx_um": np.full((n_rows, 3), np.nan),
        "r_min_um": np.full(n_rows, np.nan),
        "r_max_um": np.full(n_rows, np.nan),
    }
    updated.update({field: np.zeros(n_rows, dtype=np.int64) for field in _COUNT_FIELDS.values()})
    removed_volume = removed_interface = 0
    conn = np.ones((3, 3, 3), dtype=bool)

    for r0 in range(0, n_rows, batch_rows):
        r1 = min(r0 + batch_rows, n_rows)
        block = {name: np.asarray(group[name][indptr[name][r0]:indptr[name][r1]])
                 for name in _COMPONENT_KEYS}
        kept_parts: Dict[str, List[np.ndarray]] = {name: [] for name in _COMPONENT_KEYS}
        for row in range(r0, r1):
            origin = rows["origin_zyx"][row]
            shape = tuple(int(v) for v in rows["shape_zyx"][row])
            crop = tuple(slice(int(origin[a]), int(origin[a]) + shape[a]) for a in range(3))
            keep = ~nuclei[crop].ravel()
            footprint = []
            for name in _COMPONENT_KEYS:
                lo = indptr[name][row] - indptr[name][r0]
                hi = indptr[name][row + 1] - indptr[name][r0]
                flat = np.cumsum(block[name][lo:hi], dtype=np.uint64)
                kept = flat[keep[flat]]
                if name in _FOOTPRINT_KEYS:
                    removed_volume += int(flat.size - kept.size)
                    footprint.append(kept)
                else:
                    removed_interface += int(flat.size - kept.size)
                kept_parts[name].append(
                    np.diff(kept, prepend=np.uint64(0)).astype(np.uint32, copy=False)
                )
                updated[_COUNT_FIELDS[name]][row] = kept.size
            flat_fp = np.sort(np.concatenate(footprint))
            updated["n_voxels"][row] = flat_fp.size
            if flat_fp.size:
                mask = np.zeros(int(np.prod(shape)), dtype=bool)
                mask[flat_fp] = True
                updated["n_components"][row] = label(mask.reshape(shape), structure=conn)[1]
                if overlap is not None:
                    updated["shared_voxels"][row] = int(
                        (overlap[crop].ravel()[flat_fp] > 1).sum()
                    )
            c_vox, c_um, r_min, r_max = _pair_geometry(flat_fp, shape, origin, scale)
            updated["centroid_zyx_voxel"][row] = c_vox
            updated["centroid_zyx_um"][row] = c_um
            updated["r_min_um"][row] = r_min
            updated["r_max_um"][row] = r_max
        for name in _COMPONENT_KEYS:
            counts = np.asarray([p.size for p in kept_parts[name]], dtype=np.uint64)
            start = written[name]
            if counts.sum():
                out_data[name][start:start + int(counts.sum())] = np.concatenate(kept_parts[name])
            new_indptr[name][r0 + 1:r1 + 1] = start + np.cumsum(counts)
            written[name] = start + int(counts.sum())
        print(f"  pairs {r1}/{n_rows}", end="\r", flush=True)
    print()

    for name in _COMPONENT_KEYS:
        _resized(out_group, name, written[name])
        _like(out_group, f"{name}_indptr", group[f"{name}_indptr"], (n_rows + 1,))[:] = new_indptr[name]
    rows["volume_um3"] = updated["n_voxels"] * float(np.prod(scale))
    for name, values in updated.items():
        if name in rows:
            rows[name] = values
    for name, values in rows.items():
        src = group[name]
        _like(out_group, name, src, (n_rows,) + tuple(src.shape[1:]))[:] = values.astype(src.dtype)

    _copy_attrs(group, out_group)
    out_group.attrs["committed_rows"] = n_rows
    for name in _COMPONENT_KEYS:
        out_group.attrs[f"committed_{name}"] = written[name]
    out.attrs.update({
        "subtracted_volumes": "nuclei_segmentation",
        "nuclei_exclusion": (
            "nucleus voxels removed from every component of every pair, interfaces "
            "included; pairs are never dropped, so a pair may end up empty or split"
        ),
        "source_zarr": os.path.abspath(input_zarr),
        "nuclei_segmentation_zarr": os.path.abspath(nuclei_segmentation_zarr),
        "nuclei_segmentation_key": nuc_key_used,
    })

    before = np.asarray(group["n_voxels"][:n_rows]).astype(np.int64)
    split = int(((updated["n_components"] > 1) & (np.asarray(group["n_components"][:n_rows]) <= 1)).sum())
    summary = {
        "pairs": n_rows,
        "voxels_before": int(before.sum()),
        "voxels_removed": removed_volume,
        "interface_voxels_removed": removed_interface,
        "pairs_emptied": int(((updated["n_voxels"] == 0) & (before > 0)).sum()),
        "pairs_split": split,
        "updated": updated,
    }
    print(f"  Pair voxels removed: {removed_volume:,} of {int(before.sum()):,} "
          f"({removed_volume / max(1, int(before.sum())) * 100:.1f}%, shared voxels counted per pair)")
    print(f"  Interface voxels removed: {removed_interface:,}")
    print(f"  Pairs left empty: {summary['pairs_emptied']}; pairs newly split into "
          f"several pieces: {split} (all pairs are kept)")
    return summary


def _write_nuclei_excluded_volumes_csv(input_zarr: str, output_zarr_path: str, summary: Dict[str, Any],
                                       pair_ids: np.ndarray) -> Optional[str]:
    stem_in = os.path.splitext(input_zarr.rstrip("/"))[0]
    src_csv = stem_in + ".csv"
    if not os.path.exists(src_csv):
        return None
    import pandas as pd

    frame = pd.read_csv(src_csv)
    u = summary["updated"]
    new = pd.DataFrame({
        "pair_id": pair_ids,
        "total_interscellar_volume_voxels": u["n_voxels"],
        "territory_a_voxels": u["n_territory_a"],
        "corridor_voxels": u["n_corridor"],
        "territory_b_voxels": u["n_territory_b"],
        "n_interface_a": u["n_interface_a"],
        "n_interface_b": u["n_interface_b"],
        "n_components": u["n_components"],
        "shared_voxels": u["shared_voxels"],
        "r_min_um": u["r_min_um"],
        "r_max_um": u["r_max_um"],
    })
    for i, axis in enumerate("zyx"):
        new[f"centroid_{axis}_voxel"] = u["centroid_zyx_voxel"][:, i]
        new[f"centroid_{axis}_um"] = u["centroid_zyx_um"][:, i]
    new = new.set_index("pair_id")
    frame = frame.set_index("pair_id")
    common = [c for c in new.columns if c in frame.columns]
    frame.loc[new.index, common] = new[common]
    if "voxel_volume_um3" in frame.columns:
        frame["total_interscellar_volume_um3"] = (
            frame["total_interscellar_volume_voxels"] * frame["voxel_volume_um3"]
        )
    if "exclusive_voxels" in frame.columns:
        frame["exclusive_voxels"] = frame["total_interscellar_volume_voxels"] - frame["shared_voxels"]
    out_csv = os.path.splitext(output_zarr_path.rstrip("/"))[0] + ".csv"
    frame.reset_index().to_csv(out_csv, index=False)
    return out_csv


def exclude_nuclei(
    input_zarr: str,
    nuclei_segmentation_zarr: str,
    output_zarr_path: Optional[str] = None,
    input_key: Optional[str] = None,
    nuclei_key: Optional[str] = None,
    reference_zarr: Optional[str] = None,
):
    """Subtract nuclei from any label zarr. Returns ``(result, output_path)``.

    Adaptive interscellar stores (with a 'pair_volumes' archive) are carved pair by pair,
    so overlapping volumes each keep their shared voxels; the result is then a summary
    dict. Any other label zarr is carved as a single label volume and the result is the
    carved array.
    """
    if output_zarr_path is None:
        output_zarr_path = _default_output_path(input_zarr)

    if input_key is None and _has_pair_archive(zarr.open(input_zarr, mode="r")):
        print("=" * 60)
        print("InterSCellar: Exclude nuclei from adaptive interscellar volumes")
        print("=" * 60)
        summary = _exclude_nuclei_from_pair_archive(
            input_zarr, nuclei_segmentation_zarr, output_zarr_path, nuclei_key
        )
        pair_ids = np.asarray(zarr.open(output_zarr_path, mode="r")[_ARCHIVE_GROUP]["pair_id"][:])
        out_csv = _write_nuclei_excluded_volumes_csv(
            input_zarr, output_zarr_path, summary, pair_ids
        )
        if out_csv:
            print(f"  Volumes CSV: {out_csv}")
        print(f"\nWrote to: {output_zarr_path}")
        return summary, output_zarr_path

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
            "Adaptive interscellar stores (with a 'pair_volumes' archive) are carved pair "
            "by pair in the same format, so overlapping volumes each keep their shared "
            "voxels; per-pair counts, centroids and the volumes CSV are updated to match. "
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

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

try:
    from zarr.hierarchy import Group as ZarrGroup
except ImportError:
    ZarrGroup = zarr.Group

CELL_KEY_PREFERENCE = ["labels", "0"]
INTERSCELLAR_KEY_PREFERENCE = ["interscellar_meshes", "0", "labels"]


def _create_array(store: Any, name: str, shape, dtype, chunks, fill_value=0):
    if hasattr(store, "create_array"):
        try:
            return store.create_array(
                name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill_value
            )
        except TypeError:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
    return store.create_dataset(
        name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill_value
    )


def _open_label_array(
    zarr_path: str, dataset_key: Optional[str], preferred: List[str]
) -> Tuple[Any, Any, str]:
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")
    root = zarr.open(zarr_path, mode="r")

    def ok(node: Any) -> bool:
        return hasattr(node, "ndim") and node.ndim in (3, 5)

    if dataset_key:
        node = root
        for part in [p for p in dataset_key.split("/") if p]:
            if part not in node:
                raise ValueError(f"Key '{dataset_key}' not found in {zarr_path}")
            node = node[part]
        if not ok(node):
            raise ValueError(f"Key '{dataset_key}' in {zarr_path} is not a 3D or 5D array")
        return root, node, dataset_key

    for key in preferred:
        if key not in root:
            continue
        node = root[key]
        if ok(node):
            return root, node, key
        if isinstance(node, ZarrGroup) and "0" in node and ok(node["0"]):
            return root, node["0"], f"{key}/0"
    for key in list(root.keys()):
        node = root[key]
        if ok(node):
            return root, node, key
        if isinstance(node, ZarrGroup) and "0" in node and ok(node["0"]):
            return root, node["0"], f"{key}/0"
    raise ValueError(
        f"Could not find a 3D or 5D label array in {zarr_path}. Keys: {list(root.keys())}"
    )


def _spatial_shape(node: Any) -> Tuple[int, int, int]:
    return tuple(int(v) for v in (node.shape[-3:] if node.ndim == 5 else node.shape))


def _read_slab(node: Any, z0: int, z1: int) -> np.ndarray:
    return np.asarray(node[0, 0, z0:z1] if node.ndim == 5 else node[z0:z1])


def resolve_pair_cells(
    pair_id: int,
    pairs_csv: Optional[str] = None,
    cell_a_id: Optional[int] = None,
    cell_b_id: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if cell_a_id is not None and cell_b_id is not None:
        return int(cell_a_id), int(cell_b_id)
    if pairs_csv:
        import pandas as pd

        frame = pd.read_csv(pairs_csv)
        frame = frame.rename(columns={"cell_id_a": "cell_a_id", "cell_id_b": "cell_b_id"})
        if "pair_id" not in frame.columns:
            raise ValueError(f"{pairs_csv} has no pair_id column")
        hit = frame[frame["pair_id"] == pair_id]
        if len(hit) == 0:
            lo, hi = int(frame["pair_id"].min()), int(frame["pair_id"].max())
            raise ValueError(f"pair_id {pair_id} not in {pairs_csv} (range {lo}-{hi})")
        row = hit.iloc[0]
        return int(row["cell_a_id"]), int(row["cell_b_id"])
    return None, None


def compute_pair_cell_only(
    cell_segmentation_zarr: str,
    interscellar_zarr: str,
    pair_id: int,
    out_dir: str,
    stem: Optional[str] = None,
    cell_a_id: Optional[int] = None,
    cell_b_id: Optional[int] = None,
    pad: int = 8,
    restrict_to_pair: bool = False,
    crop: bool = True,
    write_whole_volume: bool = False,
    slab: int = 8,
    cell_key: Optional[str] = None,
    interscellar_key: Optional[str] = None,
) -> Dict[str, Any]:
    stem = stem or f"pair{pair_id}"
    _, cell_node, cell_key_used = _open_label_array(
        cell_segmentation_zarr, cell_key, CELL_KEY_PREFERENCE
    )
    isc_root, isc_node, isc_key_used = _open_label_array(
        interscellar_zarr, interscellar_key, INTERSCELLAR_KEY_PREFERENCE
    )

    cell_shape, isc_shape = _spatial_shape(cell_node), _spatial_shape(isc_node)
    if cell_shape != isc_shape:
        raise ValueError(
            f"Shape mismatch: cells {cell_shape} (key '{cell_key_used}') vs "
            f"interscellar {isc_shape} (key '{isc_key_used}'). Both must be on the same grid."
        )
    print(f"  Cells:        {cell_shape} key '{cell_key_used}'")
    print(f"  Interscellar: {isc_shape} key '{isc_key_used}'")

    # One pass to locate the pair. The cell labels sitting under this pair's own voxels
    # are exactly the two cells (its territories lie inside them), so A and B can be
    # recovered here when they were not supplied.
    print(f"Scanning for pair_id {pair_id} (slabs of {slab} z)...")
    lower = [None, None, None]
    upper = [None, None, None]
    found_cells: set = set()
    pair_voxels = 0

    for z0 in range(0, cell_shape[0], slab):
        z1 = min(z0 + slab, cell_shape[0])
        isc_slab = _read_slab(isc_node, z0, z1)
        hit = isc_slab == pair_id
        if not hit.any():
            continue
        pair_voxels += int(hit.sum())
        cell_slab = _read_slab(cell_node, z0, z1)
        found_cells.update(int(v) for v in np.unique(cell_slab[hit]) if v)
        coords = np.argwhere(hit)
        local_lo = coords.min(axis=0)
        local_hi = coords.max(axis=0)
        local_lo[0] += z0
        local_hi[0] += z0
        for axis in range(3):
            lower[axis] = int(local_lo[axis]) if lower[axis] is None else min(lower[axis], int(local_lo[axis]))
            upper[axis] = int(local_hi[axis]) if upper[axis] is None else max(upper[axis], int(local_hi[axis]))

    if pair_voxels == 0:
        raise ValueError(
            f"pair_id {pair_id} has no voxels in {interscellar_zarr}. In a volume where "
            "pairs overlap, the highest pair_id wins each voxel, so a pair can be fully "
            "overwritten -- recompute it directly with the adaptive exporter instead."
        )

    if cell_a_id is None or cell_b_id is None:
        inferred = sorted(found_cells)
        if len(inferred) >= 2:
            cell_a_id, cell_b_id = inferred[0], inferred[1]
            print(f"  Inferred cells from the pair's own voxels: {cell_a_id}, {cell_b_id}")
        else:
            print(
                f"  Warning: could not infer both cells (found {inferred}); "
                "pass --cell-a-id/--cell-b-id or --pairs-csv if you need them named."
            )

    # Widen to include the whole of both cells, so the output is not a clipped cell.
    if cell_a_id is not None and cell_b_id is not None:
        print("Scanning for the extent of both cells...")
        for z0 in range(0, cell_shape[0], slab):
            z1 = min(z0 + slab, cell_shape[0])
            cell_slab = _read_slab(cell_node, z0, z1)
            hit = (cell_slab == cell_a_id) | (cell_slab == cell_b_id)
            if not hit.any():
                continue
            coords = np.argwhere(hit)
            local_lo = coords.min(axis=0)
            local_hi = coords.max(axis=0)
            local_lo[0] += z0
            local_hi[0] += z0
            for axis in range(3):
                lower[axis] = min(lower[axis], int(local_lo[axis]))
                upper[axis] = max(upper[axis], int(local_hi[axis]))

    region = tuple(
        slice(max(0, lower[a] - pad), min(cell_shape[a], upper[a] + pad + 1))
        for a in range(3)
    )
    print(
        f"  Pair region: z {region[0].start}:{region[0].stop}, "
        f"y {region[1].start}:{region[1].stop}, x {region[2].start}:{region[2].stop}"
    )

    cells_region = np.asarray(
        cell_node[(0, 0) + region] if cell_node.ndim == 5 else cell_node[region]
    )
    isc_region = np.asarray(
        isc_node[(0, 0) + region] if isc_node.ndim == 5 else isc_node[region]
    )

    pair_mask = isc_region == pair_id
    cell_only = np.where(pair_mask, 0, cells_region)
    if restrict_to_pair and cell_a_id is not None and cell_b_id is not None:
        cell_only = np.where(
            (cell_only == cell_a_id) | (cell_only == cell_b_id), cell_only, 0
        )
    cell_only = cell_only.astype(cells_region.dtype, copy=False)

    other_pairs = int(((isc_region > 0) & ~pair_mask & (cells_region > 0)).sum())
    removed = int((pair_mask & (cells_region > 0)).sum())

    other_pairs = other_pairs  # keep the diagnostic computed above
    out_shape = cell_only.shape if crop else cell_shape
    chunks = tuple(min(64, n) for n in out_shape)

    # visualize_pair_3d needs three things sitting together: a cell-only volume keyed
    # 'labels', an interscellar volume keyed 'interscellar_meshes' with the SAME shape,
    # and a way to map pair_id -> cell IDs. It derives the last from the interscellar
    # zarr's own name: strip a trailing '_interscellar_volumes' and look for
    # '<stem>_volumes.csv' beside it. Naming the bundle to satisfy that rule makes the
    # viewer work with no symlinks or database.
    os.makedirs(out_dir, exist_ok=True)
    cell_only_path = os.path.join(out_dir, f"{stem}_cell_only_volumes.zarr")
    isc_path = os.path.join(out_dir, f"{stem}_interscellar_volumes.zarr")
    csv_path = os.path.join(out_dir, f"{stem}_volumes.csv")

    def _write(path, key, dtype, filler):
        store = zarr.open(path, mode="w")
        dataset = _create_array(store, key, out_shape, dtype, chunks)
        filler(dataset)
        store.attrs["pair_id"] = int(pair_id)
        if cell_a_id is not None:
            store.attrs["cell_a_id"] = int(cell_a_id)
        if cell_b_id is not None:
            store.attrs["cell_b_id"] = int(cell_b_id)
        store.attrs["cell_segmentation_zarr"] = os.path.abspath(cell_segmentation_zarr)
        store.attrs["source_interscellar_zarr"] = os.path.abspath(interscellar_zarr)
        store.attrs["pair_voxels"] = pair_voxels
        store.attrs["voxels_removed"] = removed
        store.attrs["intracellular_voxels_of_other_pairs_kept"] = other_pairs
        store.attrs["region_zyx"] = [[sl.start, sl.stop] for sl in region]
        store.attrs["cropped"] = bool(crop)
        store.attrs["populated"] = (
            "cropped to the pair region" if crop
            else ("whole volume" if write_whole_volume
                  else "pair region only; voxels outside region_zyx are 0")
        )
        if crop:
            store.attrs["crop_origin_zyx"] = [sl.start for sl in region]
        return store

    def _fill_cells(dataset):
        if crop:
            dataset[:] = cell_only
        elif write_whole_volume:
            print("  Copying the whole segmentation (slab by slab)...")
            for z0 in range(0, cell_shape[0], slab):
                z1 = min(z0 + slab, cell_shape[0])
                cell_slab = _read_slab(cell_node, z0, z1)
                isc_slab = _read_slab(isc_node, z0, z1)
                out = np.where(isc_slab == pair_id, 0, cell_slab)
                if restrict_to_pair and cell_a_id is not None and cell_b_id is not None:
                    out = np.where((out == cell_a_id) | (out == cell_b_id), out, 0)
                dataset[z0:z1] = out.astype(cell_only.dtype, copy=False)
        else:
            dataset[region] = cell_only

    def _fill_isc(dataset):
        pair_only = (pair_mask.astype(np.uint32) * int(pair_id))
        if crop:
            dataset[:] = pair_only
        else:
            dataset[region] = pair_only

    cell_store = _write(cell_only_path, "labels", str(cell_only.dtype), _fill_cells)
    cell_store.attrs["description"] = (
        f"Cell segmentation with only pair {pair_id}'s interscellar volume subtracted"
    )
    cell_store.attrs["subtraction"] = "pairwise: cells minus (interscellar == pair_id)"

    isc_store = _write(isc_path, "interscellar_meshes", "uint32", _fill_isc)
    isc_store.attrs["description"] = (
        f"Interscellar volume of pair {pair_id} alone, extracted from the source zarr"
    )

    # Sidecar the viewer reads to turn pair_id into cell IDs.
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "cell_a_id", "cell_b_id", "pair_voxels",
                        "voxels_removed", "intracellular_voxels_of_other_pairs_kept"],
        )
        writer.writeheader()
        writer.writerow({
            "pair_id": pair_id,
            "cell_a_id": "" if cell_a_id is None else cell_a_id,
            "cell_b_id": "" if cell_b_id is None else cell_b_id,
            "pair_voxels": pair_voxels,
            "voxels_removed": removed,
            "intracellular_voxels_of_other_pairs_kept": other_pairs,
        })

    print(f"Wrote bundle to {out_dir}")
    print(f"  {os.path.basename(cell_only_path)}   key 'labels', shape {out_shape}")
    print(f"  {os.path.basename(isc_path)}   key 'interscellar_meshes', shape {out_shape}")
    print(f"  {os.path.basename(csv_path)}   pair_id -> cell IDs for the viewer")
    print(f"  Pair {pair_id} voxels: {pair_voxels:,} | removed from cells: {removed:,}")
    print(f"  Intracellular voxels of OTHER pairs left intact: {other_pairs:,}"
          + ("  (the global cutter would have removed these)" if other_pairs else ""))
    print()
    print("Visualize with:")
    print(f"  visualize-pair-3d --pair-id {pair_id} \\")
    print(f"    --cell-only-zarr {cell_only_path} \\")
    print(f"    --interscellar-zarr {isc_path}")

    return {
        "pair_id": pair_id,
        "cell_a_id": cell_a_id,
        "cell_b_id": cell_b_id,
        "region": region,
        "pair_voxels": pair_voxels,
        "voxels_removed": removed,
        "other_pair_voxels_kept": other_pairs,
        "out_dir": out_dir,
        "cell_only_zarr": cell_only_path,
        "interscellar_zarr": isc_path,
        "volumes_csv": csv_path,
        "shape": out_shape,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cell-only volume for a single neighboring pair. Subtracts ONLY that pair's "
            "interscellar volume from the cell segmentation, leaving voxels claimed by "
            "other pairs intact -- unlike the global cookie cutter, which removes the "
            "union of every pair."
        )
    )
    parser.add_argument("--cell-segmentation-zarr", required=True,
                        help="Cell segmentation zarr.")
    parser.add_argument("--interscellar-zarr", required=True,
                        help="Interscellar volumes zarr, labeled by pair_id.")
    parser.add_argument("--pair-id", type=int, required=True,
                        help="The pair whose interscellar volume is subtracted.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for the output bundle. Defaults to "
                             "<interscellar_dir>/pair_<ID>_view/")
    parser.add_argument("--stem", default=None,
                        help="Filename stem inside the bundle (default: pair<ID>).")
    parser.add_argument("--full-grid", action="store_true",
                        help="Write on the source volume's full grid instead of cropping to "
                             "the pair. Larger and slower; the crop is what the viewer needs.")
    parser.add_argument("--pairs-csv", default=None,
                        help="Optional CSV with pair_id/cell_a_id/cell_b_id, to name the cells. "
                             "They are otherwise inferred from the pair's own voxels.")
    parser.add_argument("--cell-a-id", type=int, default=None)
    parser.add_argument("--cell-b-id", type=int, default=None)
    parser.add_argument("--restrict-to-pair", action="store_true",
                        help="Keep only the two cells of this pair; drop all other cells.")
    parser.add_argument("--write-whole-volume", action="store_true",
                        help="Populate the entire volume (all cells everywhere) instead of just "
                             "the pair's region. Slower and larger; unnecessary for pair viewing.")

    parser.add_argument("--pad", type=int, default=8, help="Padding voxels around the region.")
    parser.add_argument("--slab", type=int, default=8, help="Z-slab size for the scan.")
    parser.add_argument("--cell-key", default=None, help="Dataset key in the cell zarr.")
    parser.add_argument("--interscellar-key", default=None,
                        help="Dataset key in the interscellar zarr.")
    args = parser.parse_args(argv)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.interscellar_zarr) or ".", f"pair_{args.pair_id}_view"
    )

    print("=" * 60)
    print(f"InterSCellar: Cell-only volume for pair {args.pair_id}")
    print("=" * 60)

    try:
        cell_a_id, cell_b_id = resolve_pair_cells(
            args.pair_id, args.pairs_csv, args.cell_a_id, args.cell_b_id
        )
        compute_pair_cell_only(
            cell_segmentation_zarr=args.cell_segmentation_zarr,
            interscellar_zarr=args.interscellar_zarr,
            pair_id=args.pair_id,
            out_dir=out_dir,
            stem=args.stem,
            cell_a_id=cell_a_id,
            cell_b_id=cell_b_id,
            pad=args.pad,
            restrict_to_pair=args.restrict_to_pair,
            crop=not args.full_grid,
            write_whole_volume=args.write_whole_volume,
            slab=args.slab,
            cell_key=args.cell_key,
            interscellar_key=args.interscellar_key,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

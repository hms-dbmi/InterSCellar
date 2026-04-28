import argparse
import os
import sqlite3
import sys
from typing import Set

import numpy as np
import pandas as pd

try:
    import zarr
    try:
        from zarr.hierarchy import Group as ZarrGroup
    except ImportError:
        ZarrGroup = zarr.Group
except ImportError:
    print("Error: zarr not installed. Install with: pip install zarr")
    sys.exit(1)

try:
    import napari
except ImportError:
    print("Error: napari not installed. Install with: pip install 'napari[all]'")
    sys.exit(1)


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


def _parse_pair_ids(raw: str) -> Set[int]:
    if " " in raw:
        raise ValueError(
            "Use comma-separated pair IDs without spaces, e.g. --pair-ids 17,32,193"
        )
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("No pair IDs provided. Example: --pair-ids 1,2,3")
    try:
        pair_ids = {int(x) for x in items}
    except ValueError as exc:
        raise ValueError(
            "Invalid pair ID in --pair-ids. IDs must be integers."
        ) from exc
    if any(pid <= 0 for pid in pair_ids):
        raise ValueError("Pair IDs must be positive integers.")
    return pair_ids


def _interscellar_zarr_stem(interscellar_zarr_path: str) -> str:
    base = os.path.splitext(os.path.basename(interscellar_zarr_path))[0]
    while base.endswith("_interscellar_volumes"):
        base = base[: -len("_interscellar_volumes")]
    return base


def _resolve_cells_from_db(db_path: str, pair_ids: Set[int]) -> Set[int]:
    conn = sqlite3.connect(db_path)
    try:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "interscellar_volumes" in table_names:
            placeholders = ",".join("?" for _ in pair_ids)
            query = (
                "SELECT cell_a_id, cell_b_id FROM interscellar_volumes "
                f"WHERE pair_id IN ({placeholders})"
            )
            rows = conn.execute(query, tuple(sorted(pair_ids))).fetchall()
            return {int(x) for row in rows for x in row if x is not None}

        if "neighbors" in table_names:
            placeholders = ",".join("?" for _ in pair_ids)
            query = (
                "SELECT cell_id_a, cell_id_b FROM neighbors "
                f"WHERE pair_id IN ({placeholders})"
            )
            rows = conn.execute(query, tuple(sorted(pair_ids))).fetchall()
            return {int(x) for row in rows for x in row if x is not None}
    finally:
        conn.close()

    return set()


def _resolve_cells_from_csv(csv_path: str, pair_ids: Set[int]) -> Set[int]:
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        return set()
    if "cell_a_id" not in df.columns or "cell_b_id" not in df.columns:
        return set()
    sub = df[df["pair_id"].isin(pair_ids)]
    if sub.empty:
        return set()
    return set(sub["cell_a_id"].astype(np.int64).tolist()) | set(
        sub["cell_b_id"].astype(np.int64).tolist()
    )


def _resolve_selected_cell_ids(
    interscellar_zarr_path: str, pair_ids: Set[int], explicit_db: str = None
) -> Set[int]:
    zarr_dir = os.path.dirname(interscellar_zarr_path) or "."
    stem = _interscellar_zarr_stem(interscellar_zarr_path)

    db_candidates = []
    if explicit_db and os.path.exists(explicit_db):
        db_candidates.append(explicit_db)

    for name in (
        f"{stem}_interscellar_volumes.db",
        f"{stem}_neighbor_graph.db",
    ):
        p = os.path.join(zarr_dir, name)
        if p not in db_candidates and os.path.exists(p):
            db_candidates.append(p)

    for db_path in db_candidates:
        try:
            cell_ids = _resolve_cells_from_db(db_path, pair_ids)
        except Exception as exc:
            print(f"Warning: failed to read {db_path}: {exc}")
            continue
        if cell_ids:
            print(f"Resolved cell IDs from DB: {db_path}")
            return cell_ids

    csv_path = os.path.join(zarr_dir, f"{stem}_volumes.csv")
    if os.path.exists(csv_path):
        try:
            cell_ids = _resolve_cells_from_csv(csv_path, pair_ids)
        except Exception as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
            cell_ids = set()
        if cell_ids:
            print(f"Resolved cell IDs from CSV: {csv_path}")
            return cell_ids

    return set()


def _load_cell_only_labels(cell_only_zarr):
    if "labels" in cell_only_zarr:
        cell_only_labels = cell_only_zarr["labels"]
    elif "0" in cell_only_zarr:
        if isinstance(cell_only_zarr["0"], ZarrGroup):
            if "0" in cell_only_zarr["0"]:
                cell_only_labels = cell_only_zarr["0"]["0"]
            else:
                raise RuntimeError("Unexpected zarr structure in cell-only zarr")
        else:
            cell_only_labels = cell_only_zarr["0"]
    else:
        found = False
        for key in cell_only_zarr.keys():
            node = cell_only_zarr[key]
            if hasattr(node, "ndim") and node.ndim >= 3:
                cell_only_labels = node
                found = True
                print(f"  Found cell-only data in key '{key}'")
                break
        if not found:
            raise RuntimeError(
                f"Could not find data in cell-only zarr. Keys: {list(cell_only_zarr.keys())}"
            )

    if cell_only_labels.ndim == 5:
        print(f"Cell-only zarr shape (5D): {cell_only_labels.shape}")
        print("Using [0, 0, ...] for 3D visualization")
        return cell_only_labels[0, 0]
    return cell_only_labels


def _load_interscellar_labels(interscellar_zarr):
    if "interscellar_meshes" in interscellar_zarr:
        interscellar_labels = interscellar_zarr["interscellar_meshes"]
    elif "0" in interscellar_zarr:
        if isinstance(interscellar_zarr["0"], ZarrGroup):
            if "0" in interscellar_zarr["0"]:
                interscellar_labels = interscellar_zarr["0"]["0"]
            else:
                raise RuntimeError("Unexpected zarr structure in interscellar zarr")
        else:
            interscellar_labels = interscellar_zarr["0"]
    else:
        found = False
        for key in interscellar_zarr.keys():
            node = interscellar_zarr[key]
            if hasattr(node, "ndim") and node.ndim >= 3:
                interscellar_labels = node
                found = True
                print(f"  Found interscellar data in key '{key}'")
                break
        if not found:
            raise RuntimeError(
                f"Could not find interscellar data in zarr. Keys: {list(interscellar_zarr.keys())}"
            )

    if interscellar_labels.ndim == 5:
        print(f"Interscellar zarr shape (5D): {interscellar_labels.shape}")
        print("Using [0, 0, ...] for 3D visualization")
        return interscellar_labels[0, 0]
    return interscellar_labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize full cell-only volume and selected interscellar pair_ids in Napari "
            "(full-scale coordinates, no cropping)."
        )
    )
    parser.add_argument(
        "--cell-only-zarr",
        type=str,
        required=True,
        help="Path to cell-only volumes zarr file",
    )
    parser.add_argument(
        "--interscellar-zarr",
        type=str,
        required=True,
        help="Path to interscellar volumes zarr file",
    )
    parser.add_argument(
        "--pair-ids",
        type=str,
        required=True,
        help="Comma-separated pair IDs without spaces, e.g. '12,48,103'",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional DB path for pair_id -> cell_id mapping",
    )
    parser.add_argument(
        "--cell-only-opacity",
        type=float,
        default=0.7,
        help="Opacity for filtered cell-only volumes layer (0.0-1.0)",
    )
    parser.add_argument(
        "--interscellar-opacity",
        type=float,
        default=0.9,
        help="Opacity for selected interscellar layer (0.0-1.0)",
    )
    parser.add_argument(
        "--show-all-cell-labels",
        action="store_true",
        help="Add a low-opacity layer with all cell labels for context",
    )

    args = parser.parse_args()

    try:
        pair_ids = _parse_pair_ids(args.pair_ids)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cell_only_path = _find_file(args.cell_only_zarr, script_dir)
    interscellar_path = _find_file(args.interscellar_zarr, script_dir)
    db_path = None
    if args.db:
        db_path = _find_file(args.db, script_dir)

    if not os.path.exists(cell_only_path):
        print(f"Error: Cell-only zarr file not found: {cell_only_path}")
        sys.exit(1)
    if not os.path.exists(interscellar_path):
        print(f"Error: Interscellar zarr file not found: {interscellar_path}")
        sys.exit(1)
    if db_path and not os.path.exists(db_path):
        print(f"Error: DB file not found: {db_path}")
        sys.exit(1)

    print("Loading zarr files...")
    print(f"Cell-only zarr: {cell_only_path}")
    print(f"Interscellar zarr: {interscellar_path}")
    print(f"Selected pair IDs ({len(pair_ids)}): {sorted(pair_ids)}")

    cell_only_zarr = zarr.open(cell_only_path, mode="r")
    interscellar_zarr = zarr.open(interscellar_path, mode="r")

    try:
        cell_only_3d = _load_cell_only_labels(cell_only_zarr)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    try:
        interscellar_labels = _load_interscellar_labels(interscellar_zarr)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Cell-only shape: {cell_only_3d.shape}")
    print(f"Interscellar shape: {interscellar_labels.shape}")
    if cell_only_3d.shape != interscellar_labels.shape:
        print("Warning: shape mismatch may cause alignment issues.")

    print("Resolving participating cells from selected pair IDs...")
    selected_cell_ids = _resolve_selected_cell_ids(interscellar_path, pair_ids, db_path)
    if selected_cell_ids:
        print(f"Resolved cell IDs: {len(selected_cell_ids)}")
    else:
        print(
            "Warning: could not resolve cell IDs for selected pair IDs from DB/CSV. "
            "Cell-only layer will be empty unless --show-all-cell-labels is used."
        )

    print("Loading arrays and masking selected IDs...")
    interscellar_data = np.asarray(interscellar_labels)
    cell_only_data = np.asarray(cell_only_3d)
    pair_ids_arr = np.array(sorted(pair_ids), dtype=interscellar_data.dtype)
    selected_mask = np.isin(interscellar_data, pair_ids_arr)
    selected_interscellar = np.where(selected_mask, interscellar_data, 0).astype(np.uint32)
    if selected_cell_ids:
        cell_ids_arr = np.array(sorted(selected_cell_ids), dtype=cell_only_data.dtype)
        selected_cells_mask = np.isin(cell_only_data, cell_ids_arr)
        selected_cell_only = np.where(selected_cells_mask, cell_only_data, 0).astype(np.uint32)
    else:
        selected_cells_mask = np.zeros_like(cell_only_data, dtype=bool)
        selected_cell_only = np.zeros_like(cell_only_data, dtype=np.uint32)

    selected_voxels = int(selected_mask.sum())
    selected_cell_voxels = int(selected_cells_mask.sum())
    print(f"Selected interscellar voxels: {selected_voxels}")
    print(f"Selected cell-only voxels: {selected_cell_voxels}")
    if selected_voxels == 0:
        print("Warning: No voxels found for the selected pair IDs.")

    print("Launching Napari viewer...")
    viewer = napari.Viewer(title="Multi Pair 3D Visualization (Full Scale)")

    viewer.add_labels(
        selected_cell_only,
        name="selected_cell_only_volumes",
        opacity=args.cell_only_opacity,
    )
    viewer.add_labels(
        selected_interscellar,
        name="selected_interscellar_pairs",
        opacity=args.interscellar_opacity,
    )
    if args.show_all_cell_labels:
        viewer.add_labels(
            cell_only_data.astype(np.uint32),
            name="all_cell_only_volumes",
            opacity=0.2,
        )

    if cell_only_3d.shape:
        zc, yc, xc = np.array(cell_only_3d.shape) / 2.0
        viewer.camera.center = (xc, yc)
        viewer.camera.zoom = 0.5

    print("Viewer launched successfully.")
    print("Only selected pair IDs and their corresponding cell-only labels are shown.")
    napari.run()


if __name__ == "__main__":
    main()

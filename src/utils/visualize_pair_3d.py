import argparse
import sqlite3
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import zarr

try:
    from zarr.hierarchy import Group as ZarrGroup
except ImportError:
    ZarrGroup = zarr.Group


def _interscellar_zarr_stem(interscellar_zarr_path: str) -> str:
    base = os.path.splitext(os.path.basename(interscellar_zarr_path))[0]
    while base.endswith("_interscellar_volumes"):
        base = base[: -len("_interscellar_volumes")]
    return base


def _find_file(filename, description, script_dir):
    possible_paths = [
        filename,
        Path(script_dir) / filename,
        Path(".") / filename,
        Path(script_dir).parent / filename,
    ]

    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            return str(path_obj.resolve())

    raise FileNotFoundError(
        f"{description} not found: {filename}\n"
        f"Checked locations:\n" + "\n".join(f"    - {p}" for p in possible_paths)
    )


def _cell_only_3d_view(zroot) -> object:
    if "labels" in zroot:
        arr = zroot["labels"]
    elif "0" in zroot:
        if isinstance(zroot["0"], ZarrGroup) and "0" in zroot["0"]:
            arr = zroot["0"]["0"]
        else:
            arr = zroot["0"]
    else:
        for key in zroot.keys():
            node = zroot[key]
            if hasattr(node, "ndim") and node.ndim >= 3:
                arr = node
                print(f"  Using cell-only key '{key}'")
                break
        else:
            raise RuntimeError(
                f"Could not find labels in cell-only zarr. Keys: {list(zroot.keys())}"
            )

    if arr.ndim == 5:
        return arr[0, 0]
    return arr


def _try_pair_from_interscellar_db(conn: sqlite3.Connection, pair_id: int) -> Optional[Tuple[int, int]]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='interscellar_volumes'"
    )
    if not cur.fetchone():
        return None
    row = conn.execute(
        "SELECT cell_a_id, cell_b_id FROM interscellar_volumes WHERE pair_id=?",
        (pair_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1])


def _try_pair_from_neighbors_db(conn: sqlite3.Connection, pair_id: int) -> Optional[Tuple[int, int]]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='neighbors'"
    )
    if not cur.fetchone():
        return None
    row = conn.execute(
        "SELECT cell_id_a, cell_id_b FROM neighbors WHERE pair_id=?",
        (pair_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1])


def resolve_pair_cell_ids(
    pair_id: int,
    zarr_dir: str,
    stem: str,
    explicit_db: Optional[str],
) -> Tuple[int, int]:
    tried = []
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
        tried.append(db_path)
        conn = sqlite3.connect(db_path)
        try:
            pair = _try_pair_from_interscellar_db(conn, pair_id)
            if pair is not None:
                print(f"Resolved pair_id {pair_id} from interscellar_volumes: {db_path}")
                return pair
            pair = _try_pair_from_neighbors_db(conn, pair_id)
            if pair is not None:
                print(f"Resolved pair_id {pair_id} from neighbors graph: {db_path}")
                return pair
        finally:
            conn.close()

    csv_path = os.path.join(zarr_dir, f"{stem}_volumes.csv")
    if os.path.exists(csv_path):
        tried.append(csv_path)
        df = pd.read_csv(csv_path)
        if "pair_id" not in df.columns:
            raise RuntimeError(f"CSV missing pair_id column: {csv_path}")
        sub = df[df["pair_id"] == pair_id]
        if len(sub) == 0:
            pmin, pmax = int(df["pair_id"].min()), int(df["pair_id"].max())
            raise RuntimeError(
                f"pair_id {pair_id} not in {csv_path}. Valid range [{pmin}, {pmax}], n={len(df)}"
            )
        row = sub.iloc[0]
        print(f"Resolved pair_id {pair_id} from CSV: {csv_path}")
        return int(row["cell_a_id"]), int(row["cell_b_id"])

    raise RuntimeError(
        f"Could not resolve pair_id {pair_id} to cell IDs. Tried DBs/CSV under {zarr_dir} "
        f"(stem={stem!r}). Provide --db pointing to interscellar_volumes.db or neighbor_graph.db, "
        f"or ensure {stem}_volumes.csv exists."
    )


def _bbox_from_mesh_pair_id(mesh, pair_id: int, pad: int) -> Optional[Tuple[slice, slice, slice]]:
    """Scan z-slabs (no full 3D load) for voxels equal to pair_id."""
    nz, ny, nx = mesh.shape
    z_min, z_max = None, None
    y_min, y_max = ny, -1
    x_min, x_max = nx, -1

    for zi in range(nz):
        slab = np.asarray(mesh[zi, :, :])
        m = slab == pair_id
        if not np.any(m):
            continue
        ys, xs = np.where(m)
        if z_min is None:
            z_min = zi
        z_max = zi
        y_min = min(y_min, int(ys.min()))
        y_max = max(y_max, int(ys.max()))
        x_min = min(x_min, int(xs.min()))
        x_max = max(x_max, int(xs.max()))

    if z_min is None:
        return None

    return (
        slice(max(0, z_min - pad), min(nz, z_max + pad + 1)),
        slice(max(0, y_min - pad), min(ny, y_max + pad + 1)),
        slice(max(0, x_min - pad), min(nx, x_max + pad + 1)),
    )


def _bbox_from_cell_ids(vol, cell_a: int, cell_b: int, pad: int) -> Optional[Tuple[slice, slice, slice]]:
    nz, ny, nx = vol.shape
    z_min, z_max = None, None
    y_min, y_max = ny, -1
    x_min, x_max = nx, -1

    for zi in range(nz):
        slab = np.asarray(vol[zi, :, :])
        m = (slab == cell_a) | (slab == cell_b)
        if not np.any(m):
            continue
        ys, xs = np.where(m)
        if z_min is None:
            z_min = zi
        z_max = zi
        y_min = min(y_min, int(ys.min()))
        y_max = max(y_max, int(ys.max()))
        x_min = min(x_min, int(xs.min()))
        x_max = max(x_max, int(xs.max()))

    if z_min is None:
        return None

    return (
        slice(max(0, z_min - pad), min(nz, z_max + pad + 1)),
        slice(max(0, y_min - pad), min(ny, y_max + pad + 1)),
        slice(max(0, x_min - pad), min(nx, x_max + pad + 1)),
    )


def _union_slices(
    a: Optional[Tuple[slice, slice, slice]],
    b: Optional[Tuple[slice, slice, slice]],
    shape: Tuple[int, int, int],
) -> Optional[Tuple[slice, slice, slice]]:
    if a is None:
        return b
    if b is None:
        return a
    nz, ny, nx = shape
    return (
        slice(max(0, min(a[0].start, b[0].start)), min(nz, max(a[0].stop, b[0].stop))),
        slice(max(0, min(a[1].start, b[1].start)), min(ny, max(a[1].stop, b[1].stop))),
        slice(max(0, min(a[2].start, b[2].start)), min(nx, max(a[2].stop, b[2].stop))),
    )


def main():
    script_dir = Path(__file__).parent.resolve()

    p = argparse.ArgumentParser(
        description=(
            "Visualize one interscellar pair in napari using unified pair_id "
            "(same as interscellar_meshes labels and interscellar_volumes.db)."
        )
    )
    p.add_argument(
        "--pair-id",
        type=int,
        required=True,
        help="Unified pair_id (matches interscellar mesh voxel labels and DB/CSV)",
    )
    p.add_argument(
        "--cell-only-zarr",
        required=True,
        help="Path to cell-only volumes zarr file",
    )
    p.add_argument(
        "--interscellar-zarr",
        required=True,
        help="Path to interscellar volumes zarr (contains 'interscellar_meshes')",
    )
    p.add_argument(
        "--db",
        required=False,
        default=None,
        help="Optional SQLite DB (interscellar_volumes or neighbor_graph); auto-detected if omitted",
    )
    p.add_argument(
        "--pair-opacity",
        type=float,
        default=0.6,
        help="Opacity for the interscellar volume layer",
    )
    p.add_argument(
        "--cells-opacity",
        type=float,
        default=0.7,
        help="Opacity for the two cell-only layers",
    )
    p.add_argument(
        "--bbox-pad",
        type=int,
        default=10,
        help="Padding voxels around the union bbox for interscellar + two cells",
    )
    p.add_argument(
        "--show-all-cell-labels-in-crop",
        action="store_true",
        help="Add a low-opacity layer with all label IDs in the cropped region",
    )

    args = p.parse_args()

    try:
        cell_only_zarr_path = _find_file(args.cell_only_zarr, "Cell-only zarr", script_dir)
        interscellar_zarr_path = _find_file(args.interscellar_zarr, "Interscellar zarr", script_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    zarr_dir = os.path.dirname(interscellar_zarr_path) or "."
    stem = _interscellar_zarr_stem(interscellar_zarr_path)

    explicit_db = None
    if args.db:
        try:
            explicit_db = _find_file(args.db, "Database", script_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    try:
        cell_a_id, cell_b_id = resolve_pair_cell_ids(
            args.pair_id, zarr_dir, stem, explicit_db
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Pair {args.pair_id}: cell_a_id={cell_a_id}, cell_b_id={cell_b_id}")

    print(f"Opening zarr stores (lazy)...")
    cell_only_zarr = zarr.open(cell_only_zarr_path, mode="r")
    interscellar_zarr = zarr.open(interscellar_zarr_path, mode="r")

    if "interscellar_meshes" not in interscellar_zarr:
        print(f"Error: 'interscellar_meshes' not found in {interscellar_zarr_path}")
        sys.exit(1)

    cell_only_3d = _cell_only_3d_view(cell_only_zarr)
    interscellar_mesh = interscellar_zarr["interscellar_meshes"]

    if cell_only_3d.shape != interscellar_mesh.shape:
        print("Error: cell-only and interscellar volumes have different shapes.")
        print(f"  cell-only:  {cell_only_3d.shape}")
        print(f"  interscellar: {interscellar_mesh.shape}")
        sys.exit(1)

    print(
        f"Computing crop from unified pair_id={args.pair_id} (z-slab scan; full volume not loaded)..."
    )
    pad = args.bbox_pad
    bbox_mesh = _bbox_from_mesh_pair_id(interscellar_mesh, args.pair_id, pad)
    bbox_cells = _bbox_from_cell_ids(cell_only_3d, cell_a_id, cell_b_id, pad)
    union_bbox = _union_slices(bbox_mesh, bbox_cells, cell_only_3d.shape)

    if union_bbox is None:
        print(
            f"Error: pair_id {args.pair_id} not in interscellar mesh and cells "
            f"{cell_a_id}, {cell_b_id} not found in cell-only volume."
        )
        sys.exit(1)

    if bbox_mesh is None:
        print(
            f"Warning: no voxels with value pair_id={args.pair_id} in interscellar_meshes; "
            f"using bbox from cell IDs only."
        )
    if bbox_cells is None:
        print(
            f"Warning: cell IDs {cell_a_id}, {cell_b_id} not found in cell-only volume; "
            f"using bbox from interscellar mesh only."
        )

    print(
        f"Crop: z=[{union_bbox[0].start}:{union_bbox[0].stop}], "
        f"y=[{union_bbox[1].start}:{union_bbox[1].stop}], "
        f"x=[{union_bbox[2].start}:{union_bbox[2].stop}]"
    )
    print(
        f"Region size: {union_bbox[0].stop - union_bbox[0].start} x "
        f"{union_bbox[1].stop - union_bbox[1].start} x "
        f"{union_bbox[2].stop - union_bbox[2].start} voxels"
    )

    print("Loading cropped arrays...")
    cell_only_region = np.asarray(cell_only_3d[union_bbox])
    interscellar_region = np.asarray(interscellar_mesh[union_bbox])

    pair_mask = interscellar_region == args.pair_id
    cell_a_mask = cell_only_region == cell_a_id
    cell_b_mask = cell_only_region == cell_b_id

    print(f"Interscellar voxels (pair_id={args.pair_id}): {pair_mask.sum()}")
    print(f"Cell A voxels: {cell_a_mask.sum()}")
    print(f"Cell B voxels: {cell_b_mask.sum()}")

    if not np.any(pair_mask):
        print(
            f"Warning: no interscellar voxels for pair_id in crop; "
            f"check that pair_id matches mesh labels and DB/CSV."
        )

    try:
        import napari
    except Exception:
        print("Error: napari import failed. Install with: pip install 'napari[all]'")
        raise

    v = napari.Viewer(title=f"Pair {args.pair_id}: cells {cell_a_id} / {cell_b_id}")

    v.add_labels(
        pair_mask.astype(np.uint8),
        name=f"interscellar_pair_{args.pair_id}",
        opacity=args.pair_opacity,
    )
    v.add_labels(
        cell_a_mask.astype(np.uint8),
        name=f"cell_{cell_a_id}_only",
        opacity=args.cells_opacity,
    )
    v.add_labels(
        cell_b_mask.astype(np.uint8),
        name=f"cell_{cell_b_id}_only",
        opacity=args.cells_opacity,
    )

    if args.show_all_cell_labels_in_crop:
        v.add_labels(
            cell_only_region.astype(np.uint32),
            name="all_cell_labels_in_crop",
            opacity=0.2,
        )

    zc = (union_bbox[0].start + union_bbox[0].stop) / 2
    yc = (union_bbox[1].start + union_bbox[1].stop) / 2
    xc = (union_bbox[2].start + union_bbox[2].stop) / 2
    v.camera.center = (xc, yc)
    v.camera.zoom = 0.8

    napari.run()


if __name__ == "__main__":
    main()

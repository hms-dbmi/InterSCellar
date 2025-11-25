import argparse
import sqlite3
import sys
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import zarr

def _load_segmentation_labels(seg_zarr_path: str):
    zg = zarr.open(seg_zarr_path, mode="r")

    if "labels" in zg:
        arr = zg["labels"]
        if arr.ndim == 5:
            return arr[0, 0]
        return arr

    if "0" in zg and isinstance(zg["0"], zarr.hierarchy.Group) and "0" in zg["0"]:
        arr = zg["0"]["0"]
        if arr.ndim == 5:
            return arr[0, 0]
        return arr

    for k in zg.keys():
        node = zg[k]
        if hasattr(node, "ndim") and node.ndim >= 3:
            return node
    raise RuntimeError("Could not find a 3D labels array in segmentation zarr")

def _tight_bbox(mask: np.ndarray, pad: int = 2):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    zmin = max(0, zmin - pad)
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    zmax = min(mask.shape[0], zmax + pad + 1)
    ymax = min(mask.shape[1], ymax + pad + 1)
    xmax = min(mask.shape[2], xmax + pad + 1)
    return (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))

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
        f"Checked locations:\n" +
        "\n".join(f"    - {p}" for p in possible_paths)
    )

def main():
    script_dir = Path(__file__).parent.resolve()
    
    p = argparse.ArgumentParser(description="Visualize one interscellar pair in napari")
    p.add_argument("--pair-id", type=int, required=True, help="Pair ID to visualize")
    p.add_argument("--cell-only-zarr", required=True,
                   help="Path to cell-only volumes zarr file")
    p.add_argument("--interscellar-zarr", required=True,
                   help="Path to interscellar volumes zarr file (contains 'interscellar_meshes')")
    p.add_argument("--db", required=False, default=None,
                   help="SQLite DB with interscellar_volumes table (to map pair->cell IDs).")
    p.add_argument("--pair-opacity", type=float, default=0.6, help="Opacity for the interscellar volume layer")
    p.add_argument("--cells-opacity", type=float, default=0.7, help="Opacity for the cell-only layers")
    args = p.parse_args()

    try:
        cell_only_zarr_path = _find_file(args.cell_only_zarr, "Cell-only zarr", script_dir)
        interscellar_zarr_path = _find_file(args.interscellar_zarr, "Interscellar zarr", script_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    db_path_str = None
    if args.db is None:
        zarr_dir = os.path.dirname(interscellar_zarr_path) if os.path.dirname(interscellar_zarr_path) else "."
        zarr_basename = os.path.basename(interscellar_zarr_path)
        base_name = os.path.splitext(zarr_basename)[0]
        while base_name.endswith('_interscellar_volumes'):
            base_name = base_name[:-len('_interscellar_volumes')]
        possible_db = os.path.join(zarr_dir, f"{base_name}_interscellar_volumes.db")
        if os.path.exists(possible_db):
            db_path_str = possible_db
            print(f"Auto-detected database: {db_path_str}")
        else:
            possible_csv = os.path.join(zarr_dir, f"{base_name}_volumes.csv")
            if os.path.exists(possible_csv):
                db_path_str = None  # Will use CSV directly
                print(f"Auto-detected CSV file: {possible_csv}")
            else:
                print(f"Warning: Could not auto-detect database or CSV. Will try to find CSV in directory.")
                db_path_str = None
    else:
        try:
            db_path_str = _find_file(args.db, "Database", script_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            db_path_str = None
    
    print(f"Loading cell-only zarr: {cell_only_zarr_path}")
    cell_only_zarr = zarr.open(cell_only_zarr_path, mode='r')
    
    cell_only_labels = None
    if 'labels' in cell_only_zarr:
        cell_only_labels = cell_only_zarr['labels']
    elif '0' in cell_only_zarr:
        if isinstance(cell_only_zarr['0'], zarr.hierarchy.Group):
            if '0' in cell_only_zarr['0']:
                cell_only_labels = cell_only_zarr['0']['0']
            else:
                print(f"Error: Unexpected zarr structure in cell-only zarr")
                sys.exit(1)
        else:
            cell_only_labels = cell_only_zarr['0']
    else:
        found = False
        for key in cell_only_zarr.keys():
            node = cell_only_zarr[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                cell_only_labels = node
                found = True
                print(f"Found data in key '{key}'")
                break
        if not found:
            print(f"Error: Could not find data in cell-only zarr")
            print(f"Available keys: {list(cell_only_zarr.keys())}")
            sys.exit(1)
    
    if cell_only_labels.ndim == 5:
        cell_only_3d = cell_only_labels[0, 0]
    else:
        cell_only_3d = cell_only_labels
    
    print(f"Cell-only volumes shape: {cell_only_3d.shape}")
    
    print(f"Loading interscellar zarr: {interscellar_zarr_path}")
    interscellar_zarr = zarr.open(interscellar_zarr_path, mode='r')
    if 'interscellar_meshes' not in interscellar_zarr:
        print(f"Error: 'interscellar_meshes' dataset not found in {interscellar_zarr_path}")
        sys.exit(1)
    interscellar_mesh = interscellar_zarr['interscellar_meshes']
    
    print(f"Interscellar volumes shape: {interscellar_mesh.shape}")
    
    if interscellar_mesh.ndim == 5:
        print("Interscellar zarr is 5D, using [0, 0, ...] slice for visualization")
        interscellar_mesh_3d = interscellar_mesh[0, 0]
    else:
        interscellar_mesh_3d = interscellar_mesh
    
    if cell_only_3d.shape != interscellar_mesh_3d.shape:
        print("Error: cell-only and interscellar volumes have different shapes.")
        print(f"  cell-only shape:  {cell_only_3d.shape}")
        print(f"  interscellar shape: {interscellar_mesh_3d.shape}")
        sys.exit(1)

    print(f"Loading cell IDs for pair {args.pair_id}...")
    cell_a_id, cell_b_id = None, None
    
    if db_path_str and os.path.exists(db_path_str):
        print(f"Loading from database: {db_path_str}")
        conn = sqlite3.connect(db_path_str)
        try:
            row = conn.execute(
                "SELECT cell_a_id, cell_b_id FROM interscellar_volumes WHERE pair_id=?",
                (args.pair_id,)
            ).fetchone()
            
            count = conn.execute("SELECT COUNT(*) FROM interscellar_volumes").fetchone()[0]
            
            if row is not None:
                cell_a_id, cell_b_id = int(row[0]), int(row[1])
            elif count == 0:
                print(f"Database is empty, trying CSV file as fallback...")
                csv_path = db_path_str.replace('.db', '.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    pair_data = df[df['pair_id'] == args.pair_id]
                    if len(pair_data) > 0:
                        cell_a_id = int(pair_data.iloc[0]['cell_a_id'])
                        cell_b_id = int(pair_data.iloc[0]['cell_b_id'])
                        print(f"Found pair in CSV: {csv_path}")
        finally:
            conn.close()
    
    if cell_a_id is None or cell_b_id is None:
        zarr_dir = os.path.dirname(interscellar_zarr_path) if os.path.dirname(interscellar_zarr_path) else "."
        zarr_basename = os.path.basename(interscellar_zarr_path)
        base_name = os.path.splitext(zarr_basename)[0]
        while base_name.endswith('_interscellar_volumes'):
            base_name = base_name[:-len('_interscellar_volumes')]
        csv_path = os.path.join(zarr_dir, f"{base_name}_volumes.csv")
        if os.path.exists(csv_path):
            print(f"Trying CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            pair_data = df[df['pair_id'] == args.pair_id]
            if len(pair_data) > 0:
                cell_a_id = int(pair_data.iloc[0]['cell_a_id'])
                cell_b_id = int(pair_data.iloc[0]['cell_b_id'])
                print(f"Found pair in CSV: {csv_path}")
            else:
                if len(df) > 0:
                    min_id = df['pair_id'].min()
                    max_id = df['pair_id'].max()
                    print(f"Pair {args.pair_id} not found in CSV. Valid range: [{min_id}, {max_id}], Total: {len(df)}")
    
    if cell_a_id is None or cell_b_id is None:
        print(f"\nError: pair_id {args.pair_id} not found in database or CSV")
        print(f"Please provide --db or ensure CSV file exists in the same directory as interscellar-zarr")
        sys.exit(1)
    
    print(f"Pair {args.pair_id}: Cell A={cell_a_id}, Cell B={cell_b_id}")

    print("Finding bounding box for pair region...")
    
    cell_only_array = np.asarray(cell_only_3d)
    interscellar_array = np.asarray(interscellar_mesh_3d)
    
    print(f"Searching for pair_id {args.pair_id} in interscellar volumes...")
    pair_coords = np.argwhere(interscellar_array == args.pair_id)
    
    if pair_coords.size == 0:
        print(f"Warning: pair_id {args.pair_id} not found in interscellar volumes")

        cell_a_coords = np.argwhere(cell_only_array == cell_a_id)
        cell_b_coords = np.argwhere(cell_only_array == cell_b_id)
        
        if cell_a_coords.size == 0 and cell_b_coords.size == 0:
            print(f"Error: Neither cell A={cell_a_id} nor cell B={cell_b_id} found in cell-only volumes")
            sys.exit(1)
        
        all_coords = []
        if cell_a_coords.size > 0:
            all_coords.append(cell_a_coords)
        if cell_b_coords.size > 0:
            all_coords.append(cell_b_coords)
        
        if len(all_coords) > 0:
            all_cell_coords = np.vstack(all_coords)
            z_min, y_min, x_min = all_cell_coords.min(axis=0)
            z_max, y_max, x_max = all_cell_coords.max(axis=0)
            pad = 10
            union_bbox = (
                slice(max(0, z_min - pad), min(cell_only_array.shape[0], z_max + pad + 1)),
                slice(max(0, y_min - pad), min(cell_only_array.shape[1], y_max + pad + 1)),
                slice(max(0, x_min - pad), min(cell_only_array.shape[2], x_max + pad + 1))
            )
            print(f"Using bounding box from cell-only volumes with padding")
        else:
            print(f"Error: Could not determine bounding box")
            sys.exit(1)
    else:
        cell_a_coords = np.argwhere(cell_only_array == cell_a_id)
        cell_b_coords = np.argwhere(cell_only_array == cell_b_id)
        
        all_coords_list = [pair_coords]
        if cell_a_coords.size > 0:
            all_coords_list.append(cell_a_coords)
        if cell_b_coords.size > 0:
            all_coords_list.append(cell_b_coords)
        
        all_coords = np.vstack(all_coords_list)
        z_min, y_min, x_min = all_coords.min(axis=0)
        z_max, y_max, x_max = all_coords.max(axis=0)
        
        pad = 10
        union_bbox = (
            slice(max(0, z_min - pad), min(cell_only_array.shape[0], z_max + pad + 1)),
            slice(max(0, y_min - pad), min(cell_only_array.shape[1], y_max + pad + 1)),
            slice(max(0, x_min - pad), min(cell_only_array.shape[2], x_max + pad + 1))
        )
        print(f"Found {len(pair_coords)} voxels for pair_id {args.pair_id}")
    
    print(f"Union bbox: z=[{union_bbox[0].start}:{union_bbox[0].stop}], "
          f"y=[{union_bbox[1].start}:{union_bbox[1].stop}], "
          f"x=[{union_bbox[2].start}:{union_bbox[2].stop}]")
    print(f"Region size: {union_bbox[0].stop - union_bbox[0].start} x "
          f"{union_bbox[1].stop - union_bbox[1].start} x "
          f"{union_bbox[2].stop - union_bbox[2].start} voxels")
    
    print("Extracting region from volumes...")
    cell_only_region = cell_only_array[union_bbox]
    interscellar_region = interscellar_array[union_bbox]
    
    pair_mask = (interscellar_region == args.pair_id)
    cell_a_mask = (cell_only_region == cell_a_id)
    cell_b_mask = (cell_only_region == cell_b_id)
    
    print(f"Interscellar volume voxels: {pair_mask.sum()}")
    print(f"Cell A only voxels: {cell_a_mask.sum()}")
    print(f"Cell B only voxels: {cell_b_mask.sum()}")
    
    if pair_mask.sum() == 0:
        print(f"Warning: No interscellar volume found for pair_id {args.pair_id} in this region")

    try:
        import napari
    except Exception as e:
        print("Error: napari import failed. Install with: pip install 'napari[all]'")
        raise

    v = napari.Viewer(title=f"Pair {args.pair_id}: Cell A={cell_a_id}, Cell B={cell_b_id}")
    
    v.add_labels(pair_mask.astype(np.uint8), name=f"interscellar_volume_pair_{args.pair_id}", opacity=args.pair_opacity)
    
    v.add_labels(cell_a_mask.astype(np.uint8), name=f"cell_{cell_a_id}_only", opacity=args.cells_opacity)
    v.add_labels(cell_b_mask.astype(np.uint8), name=f"cell_{cell_b_id}_only", opacity=args.cells_opacity)
    
    if args.cells_opacity > 0:
        v.add_labels(cell_only_region.astype(np.uint32), name="all_cell_only_volumes", opacity=0.2)

    v.camera.center = (
        (union_bbox[2].start + union_bbox[2].stop) / 2,
        (union_bbox[1].start + union_bbox[1].stop) / 2,
    )
    v.camera.zoom = 0.8

    napari.run()

if __name__ == "__main__":
    main()

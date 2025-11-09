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
    p.add_argument("--mesh-zarr", default="melanoma_interscellar_volumes.zarr",
                   help="Path to interscellar mesh zarr (contains 'interscellar_meshes')")
    p.add_argument("--seg-zarr", default="data/3d_melanoma/melanoma_mask.zarr",
                   help="Path to segmentation labels zarr")
    p.add_argument("--db", default="melanoma_interscellar_volumes.db",
                   help="SQLite DB with interscellar_volumes table (to map pair->cell IDs)")
    p.add_argument("--pair-opacity", type=float, default=0.6, help="Opacity for the interscellar volume layer")
    p.add_argument("--cells-opacity", type=float, default=0.7, help="Opacity for the cell-only layers")
    p.add_argument("--halo-bboxes-pickle", default="melanoma_halo_bboxes_optimized.pkl",
                   help="Path to pickle file with halo bounding boxes from find_cell_neighbors_3d.py")
    args = p.parse_args()

    try:
        mesh_zarr_path = _find_file(args.mesh_zarr, "Mesh zarr", script_dir)
        seg_zarr_path = _find_file(args.seg_zarr, "Segmentation zarr", script_dir)
        db_path_str = _find_file(args.db, "Database", script_dir)
        halo_bboxes_path = _find_file(args.halo_bboxes_pickle, "Halo bboxes pickle", script_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Loading mesh zarr: {mesh_zarr_path}")
    mesh_store = zarr.open(mesh_zarr_path, mode="r")
    if "interscellar_meshes" not in mesh_store:
        print(f"Error: 'interscellar_meshes' dataset not found in {mesh_zarr_path}")
        sys.exit(1)
    mesh = mesh_store["interscellar_meshes"]

    print(f"Loading segmentation zarr: {seg_zarr_path}")
    labels = _load_segmentation_labels(seg_zarr_path)

    if tuple(labels.shape[-3:]) != tuple(mesh.shape[-3:]):
        print("Error: segmentation and interscellar mesh volumes have different shapes.")
        print(f"  seg shape:  {labels.shape}")
        print(f"  mesh shape: {mesh.shape}")
        sys.exit(1)

    print(f"Loading database: {db_path_str}")
    cell_a_id, cell_b_id = None, None
    
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
                else:
                    print(f"Pair {args.pair_id} not found in CSV either")
                    if len(df) > 0:
                        min_id = df['pair_id'].min()
                        max_id = df['pair_id'].max()
                        print(f"Valid pair_id range in CSV: [{min_id}, {max_id}]")
                        print(f"Total pairs in CSV: {len(df)}")
            else:
                print(f"CSV file not found: {csv_path}")
    finally:
        conn.close()
    
    if cell_a_id is None or cell_b_id is None:
        print(f"\nError: pair_id {args.pair_id} not found in database or CSV")
        sys.exit(1)
    
    print(f"Pair {args.pair_id}: Cell A={cell_a_id}, Cell B={cell_b_id}")

    print("Computing union bbox from halo bboxes for cells...")
    
    if not os.path.exists(halo_bboxes_path):
        print(f"Error: Halo bounding boxes pickle file not found: {halo_bboxes_path}")
        sys.exit(1)
    
    with open(halo_bboxes_path, "rb") as f:
        bbox_data = pickle.load(f)
    
    if 'all_bboxes_with_halo' in bbox_data:
        halo_bboxes = bbox_data['all_bboxes_with_halo']
    elif isinstance(bbox_data, dict):
        halo_bboxes = bbox_data
    else:
        print(f"Error: Invalid format in halo bounding boxes pickle file")
        sys.exit(1)
    
    if cell_a_id not in halo_bboxes or cell_b_id not in halo_bboxes:
        print(f"Error: Halo bboxes not found for cell A={cell_a_id} or cell B={cell_b_id}")
        sys.exit(1)
    
    bbox_a = halo_bboxes[cell_a_id]
    bbox_b = halo_bboxes[cell_b_id]
    
    z_start = min(bbox_a[0].start, bbox_b[0].start)
    z_stop = max(bbox_a[0].stop, bbox_b[0].stop)
    y_start = min(bbox_a[1].start, bbox_b[1].start)
    y_stop = max(bbox_a[1].stop, bbox_b[1].stop)
    x_start = min(bbox_a[2].start, bbox_b[2].start)
    x_stop = max(bbox_a[2].stop, bbox_b[2].stop)
    
    union_bbox = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Union bbox from halo bboxes: z=[{union_bbox[0].start}:{union_bbox[0].stop}], "
          f"y=[{union_bbox[1].start}:{union_bbox[1].stop}], "
          f"x=[{union_bbox[2].start}:{union_bbox[2].stop}]")
    
    print("Checking if pair_id exists in union bbox region...")
    mesh_region = np.asarray(mesh[union_bbox])
    pair_mask_in_union = (mesh_region == args.pair_id)
    pair_voxels_in_union = pair_mask_in_union.sum()
    
    if pair_voxels_in_union > 0:
        print(f"Found {pair_voxels_in_union} voxels for pair_id {args.pair_id} in union bbox")
    else:
        print(f"No voxels found for pair_id {args.pair_id} in union bbox")
        print(f"Searching for pair_id {args.pair_id} in entire mesh zarr...")
        
        pair_coords = []
        chunk_shape = mesh.chunks if hasattr(mesh, 'chunks') else (64, 64, 64)
        z_step, y_step, x_step = chunk_shape
        found_pairs = False
        
        for z_start in range(0, mesh.shape[0], z_step):
            z_end = min(z_start + z_step, mesh.shape[0])
            for y_start in range(0, mesh.shape[1], y_step):
                y_end = min(y_start + y_step, mesh.shape[1])
                for x_start in range(0, mesh.shape[2], x_step):
                    x_end = min(x_start + x_step, mesh.shape[2])
                    
                    chunk = np.asarray(mesh[z_start:z_end, y_start:y_end, x_start:x_end])
                    matching_coords = np.argwhere(chunk == args.pair_id)
                    
                    if matching_coords.size > 0:
                        found_pairs = True
                        matching_coords_global = matching_coords + np.array([z_start, y_start, x_start])
                        pair_coords.append(matching_coords_global)
        
        if found_pairs and len(pair_coords) > 0:
            all_pair_coords = np.vstack(pair_coords)
            z_min, y_min, x_min = all_pair_coords.min(axis=0)
            z_max, y_max, x_max = all_pair_coords.max(axis=0)
            
            interscellar_bbox = (
                slice(max(0, z_min - 5), min(mesh.shape[0], z_max + 6)),
                slice(max(0, y_min - 5), min(mesh.shape[1], y_max + 6)),
                slice(max(0, x_min - 5), min(mesh.shape[2], x_max + 6))
            )
            
            print(f"  Found {len(all_pair_coords)} voxels for pair_id {args.pair_id} at:")
            print(f"    z=[{interscellar_bbox[0].start}:{interscellar_bbox[0].stop}], "
                  f"y=[{interscellar_bbox[1].start}:{interscellar_bbox[1].stop}], "
                  f"x=[{interscellar_bbox[2].start}:{interscellar_bbox[2].stop}]")
            
            z_start = min(union_bbox[0].start, interscellar_bbox[0].start)
            z_stop = max(union_bbox[0].stop, interscellar_bbox[0].stop)
            y_start = min(union_bbox[1].start, interscellar_bbox[1].start)
            y_stop = max(union_bbox[1].stop, interscellar_bbox[1].stop)
            x_start = min(union_bbox[2].start, interscellar_bbox[2].start)
            x_stop = max(union_bbox[2].stop, interscellar_bbox[2].stop)
            
            union_bbox = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
            print(f"Expanded union bbox to include both cells and interscellar volume")
        else:
            print(f"Warning: pair_id {args.pair_id} not found anywhere in mesh zarr")
            print(f"Using union bbox from halo bboxes only (cells will be shown but interscellar volume may be missing)")
    
    print(f"Union bbox: z=[{union_bbox[0].start}:{union_bbox[0].stop}], "
          f"y=[{union_bbox[1].start}:{union_bbox[1].stop}], "
          f"x=[{union_bbox[2].start}:{union_bbox[2].stop}]")
    print(f"Region size: {union_bbox[0].stop - union_bbox[0].start} x "
          f"{union_bbox[1].stop - union_bbox[1].start} x "
          f"{union_bbox[2].stop - union_bbox[2].start} voxels")
    
    print("Loading union bbox region from zarr arrays...")
    mesh_union = np.asarray(mesh[union_bbox])
    
    if labels.ndim == 5:
        labels_union = np.asarray(labels[union_bbox[0], union_bbox[1], union_bbox[2]])
    else:
        labels_union = np.asarray(labels[union_bbox])
    
    if mesh_union.shape != labels_union.shape:
        print(f"Warning: Shape mismatch - mesh_union: {mesh_union.shape}, labels_union: {labels_union.shape}")

        if labels_union.ndim == 5:
            labels_union = labels_union[0, 0]

        if mesh_union.shape != labels_union.shape:
            print(f"Error: Could not align shapes - mesh_union: {mesh_union.shape}, labels_union: {labels_union.shape}")
            sys.exit(1)
    
    pair_mask = (mesh_union == args.pair_id)
    cell_a_mask = (labels_union == cell_a_id)
    cell_b_mask = (labels_union == cell_b_id)
    
    cell_a_only = cell_a_mask & (~pair_mask)
    cell_b_only = cell_b_mask & (~pair_mask)
    
    print(f"Interscellar volume voxels: {pair_mask.sum()}")
    print(f"Cell A only voxels: {cell_a_only.sum()}")
    print(f"Cell B only voxels: {cell_b_only.sum()}")
    
    if pair_mask.sum() == 0:
        print(f"Warning: No interscellar volume found in union bbox for pair_id {args.pair_id}")
        print(f"Try visualizing with a larger region or check if pair_id {args.pair_id} has interscellar volume")

    try:
        import napari
    except Exception as e:
        print("Error: napari import failed. Install with: pip install 'napari[all]'")
        raise

    v = napari.Viewer()
    
    v.add_labels(pair_mask.astype(np.uint8), name=f"interscellar_volume", opacity=args.pair_opacity)
    v.add_labels(cell_a_only.astype(np.uint8), name=f"cell_{cell_a_id}_only", opacity=args.cells_opacity)
    v.add_labels(cell_b_only.astype(np.uint8), name=f"cell_{cell_b_id}_only", opacity=args.cells_opacity)
    
    if args.cells_opacity > 0:
        v.add_labels(labels_union.astype(np.uint32), name="all_cells", opacity=0.2)

    v.camera.center = (
        (union_bbox[2].start + union_bbox[2].stop) / 2,
        (union_bbox[1].start + union_bbox[1].stop) / 2,
    )

    napari.run()

if __name__ == "__main__":
    main()

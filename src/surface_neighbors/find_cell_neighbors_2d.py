# Import

import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
import cv2
import csv
import sqlite3
import os
import pickle
import math
import json

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Cell surface precomputation

def build_global_mask_2d(polygon_mask: dict) -> Tuple[np.ndarray, Tuple[int, int], dict]:
    print("Building global 2D mask...")
    
    all_points = np.concatenate([np.array(pts) for pts in polygon_mask.values()])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    mask_shape = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)
    global_mask = np.zeros(mask_shape, dtype=np.int32)
    cell_id_mapping = {cid: i+1 for i, cid in enumerate(polygon_mask.keys())}
    
    for cid, pts in polygon_mask.items():
        mapped_id = cell_id_mapping[cid]
        pts = np.array(pts, dtype=np.float32)
        
        shifted_pts = pts - np.array([min_x, min_y])
        
        int_pts = shifted_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(global_mask, [int_pts], mapped_id)
    
    print(f"Global 2D mask generated: shape {global_mask.shape}, {len(polygon_mask)} cells")
    return global_mask, mask_shape, cell_id_mapping

def build_global_mask_2d_with_mapping(polygon_mask: dict, cell_id_mapping: dict) -> Tuple[np.ndarray, Tuple[int, int]]:
    print("Building global 2D mask...")
    
    all_points = np.concatenate([np.array(pts) for pts in polygon_mask.values()])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    mask_shape = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)
    global_mask = np.zeros(mask_shape, dtype=np.int32)
    
    for cid, pts in polygon_mask.items():
        if cid not in cell_id_mapping:
            continue
        mapped_id = cell_id_mapping[cid]
        pts = np.array(pts, dtype=np.float32)
        
        shifted_pts = pts - np.array([min_x, min_y])
        
        int_pts = shifted_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(global_mask, [int_pts], mapped_id)
    
    print(f"Global mask generated: shape {global_mask.shape}, {len(polygon_mask)} cells")
    return global_mask, mask_shape

def get_bounding_boxes_2d(global_mask: np.ndarray, unique_ids: set) -> dict:
    y, x = np.nonzero(global_mask)
    cell_ids = global_mask[y, x]
    df = pd.DataFrame({'cell_id': cell_ids, 'y': y, 'x': x})
    bbox = {}
    grouped = df.groupby('cell_id')
    for cell_id, group in grouped:
        if cell_id == 0:
            continue
        miny, maxy = group['y'].min(), group['y'].max() + 1
        minx, maxx = group['x'].min(), group['x'].max() + 1
        bbox[cell_id] = (slice(miny, maxy), slice(minx, maxx))
    return bbox

def compute_bounding_box_with_halo_2d(
    surface_a: np.ndarray,
    max_distance_um: float,
    pixel_size_um: float
) -> Tuple[slice, slice]:
    y_coords, x_coords = np.where(surface_a)
    
    if len(y_coords) == 0:
        return None
    
    min_y, max_y = y_coords.min(), y_coords.max() + 1
    min_x, max_x = x_coords.min(), x_coords.max() + 1
    
    pad_y = math.ceil(max_distance_um / pixel_size_um)
    pad_x = math.ceil(max_distance_um / pixel_size_um)
    
    min_y_pad = max(0, min_y - pad_y)
    max_y_pad = max_y + pad_y + 1
    min_x_pad = max(0, min_x - pad_x)
    max_x_pad = max_x + pad_x + 1
    
    return (slice(min_y_pad, max_y_pad), slice(min_x_pad, max_x_pad))

def global_surface_2d(global_mask: np.ndarray) -> np.ndarray:
    print("Computing global surface mask...")
    
    structure = generate_binary_structure(2, 2)
    binary_mask = (global_mask > 0).astype(bool)
    eroded = binary_erosion(binary_mask, structure=structure)    
    global_surface = binary_mask & ~eroded
    
    print(f"Global surface mask computed: {global_surface.sum()} surface pixels")
    return global_surface

def all_cell_bboxes_2d(global_mask: np.ndarray) -> Dict[int, Tuple[slice, slice]]:
    print("Computing bounding boxes for all cells in single sweep...")
    
    unique_ids = set(np.unique(global_mask))
    unique_ids.discard(0)
    
    bboxes = get_bounding_boxes_2d(global_mask, unique_ids)
    
    print(f"Bounding boxes computed for {len(bboxes)} cells")
    return bboxes

def precompute_global_surface_and_halo_bboxes_2d(
    global_mask: np.ndarray, 
    max_distance_um: float,
    pixel_size_um: float
) -> Tuple[np.ndarray, Dict[int, Tuple[slice, slice]]]:
    print("Pre-computing global surface and halo-extended bounding boxes...")
    
    global_surface = global_surface_2d(global_mask)

    all_bboxes = all_cell_bboxes_2d(global_mask)
    
    print("Pre-computing halo-extended bounding boxes...")
    all_bboxes_with_halo = {}
    
    pad_y = math.ceil(max_distance_um / pixel_size_um)
    pad_x = math.ceil(max_distance_um / pixel_size_um)
    
    for cell_id, bbox in all_bboxes.items():
        slice_y, slice_x = bbox
        
        y_start = max(0, slice_y.start - pad_y)
        y_stop = min(global_mask.shape[0], slice_y.stop + pad_y)
        x_start = max(0, slice_x.start - pad_x)
        x_stop = min(global_mask.shape[1], slice_x.stop + pad_x)
        
        all_bboxes_with_halo[cell_id] = (slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Pre-computed halo-extended bounding boxes for {len(all_bboxes_with_halo)} cells")
    
    return global_surface, all_bboxes_with_halo, all_bboxes

## Surface-based neighbor identification

def find_touching_neighbors_2d(global_mask: np.ndarray, all_bboxes: dict, n_jobs: int = 1) -> set:
    print("Finding touching neighbors...")
    
    labels = global_mask
    y_dim, x_dim = labels.shape
    
    touching_pairs: set = set()
    
    # Helper function: add pairs from two same-shaped arrays
    def add_pairs(a: np.ndarray, b: np.ndarray) -> None:
        diff_mask = (a != b)
        if not diff_mask.any():
            return
        a_nz = a[diff_mask]
        b_nz = b[diff_mask]

        nz_mask = (a_nz != 0) & (b_nz != 0)
        if not nz_mask.any():
            return
        a_nz = a_nz[nz_mask]
        b_nz = b_nz[nz_mask]

        minv = np.minimum(a_nz, b_nz)
        maxv = np.maximum(a_nz, b_nz)

        touching_pairs.update(zip(minv.astype(np.int64).tolist(), maxv.astype(np.int64).tolist()))
    
    # Y-axis face adjacency: compare row y with y+1
    for y in tqdm(range(y_dim - 1), desc="4-conn touching: Y faces", ncols=100):
        a = labels[y + 1, :]
        b = labels[y, :]
        add_pairs(a, b)
    
    # X-axis face adjacency: compare col x with x+1
    for y in tqdm(range(y_dim), desc="4-conn touching: X faces", ncols=100):
        a = labels[:, 1:]
        b = labels[:, :-1]
        add_pairs(a, b)
    
    print(f"Found {len(touching_pairs)} touching neighbor pairs")
    return touching_pairs

## Surface-to-surface distance computation

def compute_surface_to_surface_distance_2d(
    global_mask: np.ndarray, 
    cell_a_id: int, 
    cell_b_id: int, 
    pixel_size_um: float,
    max_distance_um: float = float('inf')
) -> float:
    mask_a = (global_mask == cell_a_id)
    mask_b = (global_mask == cell_b_id)
    
    if not mask_a.any() or not mask_b.any():
        return float('inf')
    
    structure = generate_binary_structure(2, 2)
    surface_a = mask_a & ~binary_erosion(mask_a, structure=structure)
    surface_b = mask_b & ~binary_erosion(mask_b, structure=structure)
    
    if not surface_a.any() or not surface_b.any():
        return float('inf')
    
    bbox_with_halo = compute_bounding_box_with_halo_2d(surface_a, max_distance_um, pixel_size_um)
    
    if bbox_with_halo is None:
        return float('inf')
    
    slice_y, slice_x = bbox_with_halo
    surface_a_crop = surface_a[slice_y, slice_x]
    surface_b_crop = surface_b[slice_y, slice_x]
    
    if not surface_b_crop.any():
        return float('inf')
    
    dist_transform_crop = distance_transform_edt(~surface_a_crop, sampling=pixel_size_um) # EDT from surface A
    
    min_distance = dist_transform_crop[surface_b_crop].min()
    
    return min_distance

def compute_surface_distances_batch_2d(
    global_surface: np.ndarray,
    cell_pairs: List[Tuple[int, int]],
    pixel_size_um: float,
    max_distance_um: float,
    cells_df: pd.DataFrame,
    global_mask: np.ndarray,
    all_bboxes_with_halo: Dict[int, Tuple[slice, slice]],
    n_jobs: int = 1
) -> List[Dict[str, Any]]:
    from collections import defaultdict
    from joblib import Parallel, delayed
    
    print("VECTORIZED APPROACH: Computing surface distances using global EDTs for unique crop regions...")
    print(f"Processing {len(cell_pairs)} cell pairs with max_distance_um = {max_distance_um}")
    
    print("Step 1: Identifying unique crop regions...")
    unique_crops = set()
    cell_to_crop_tuple = {}
    
    for cell_a_id, cell_b_id in cell_pairs:
        if cell_a_id in all_bboxes_with_halo:
            crop_slice = all_bboxes_with_halo[cell_a_id]
            crop_tuple = (crop_slice[0].start, crop_slice[0].stop, 
                         crop_slice[1].start, crop_slice[1].stop)
            unique_crops.add(crop_tuple)
            cell_to_crop_tuple[cell_a_id] = crop_tuple
    
    total_cells = len(set(pair[0] for pair in cell_pairs if pair[0] in all_bboxes_with_halo))
    print(f"Found {len(unique_crops)} unique crop regions")
    
    print(f"Step 2: Computing EDTs for {len(unique_crops)} unique crop regions...")
    crop_edts = {}
    
    def compute_crop_edt(crop_tuple):
        y_start, y_stop, x_start, x_stop = crop_tuple
        crop_slice = (slice(y_start, y_stop), slice(x_start, x_stop))
        
        mask_crop = global_mask[crop_slice]
        global_surface_crop = global_surface[crop_slice]

        return crop_tuple, None, mask_crop, global_surface_crop
    
    if n_jobs == 1:
        # Sequential crop data extraction
        pbar = tqdm(unique_crops, desc="Extracting crop regions", 
                   unit="crops", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        for crop_tuple in pbar:
            crop_tuple, _, mask_crop, global_surface_crop = compute_crop_edt(crop_tuple)
            crop_edts[crop_tuple] = {
                'mask_crop': mask_crop,
                'global_surface_crop': global_surface_crop
            }
    else:
        # Parallel crop data extraction
        print(f"Extracting crop regions with {n_jobs} parallel jobs...")
        pbar = tqdm(unique_crops, desc="Extracting crop regions in parallel", 
                   unit="crops", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(compute_crop_edt)(crop_tuple) 
            for crop_tuple in pbar
        )
        
        for crop_tuple, _, mask_crop, global_surface_crop in results_list:
            crop_edts[crop_tuple] = {
                'mask_crop': mask_crop,
                'global_surface_crop': global_surface_crop
            }
    
    print(f"Completed crop region extraction for all {len(crop_edts)} unique crop regions")
    
    print("Step 3: Computing surface-to-surface distances for all cell pairs...")
    results = []
    cell_type_map = dict(zip(cells_df['cell_id'], cells_df['cell_type']))
    
    pbar = tqdm(cell_pairs, desc="Computing surface-to-surface distances", 
               unit="pairs", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
    
    for cell_a_id, cell_b_id in pbar:
        if cell_a_id not in cell_to_crop_tuple:
            continue
            
        crop_tuple = cell_to_crop_tuple[cell_a_id]
        crop_data = crop_edts[crop_tuple]
        
        mask_crop = crop_data['mask_crop']
        global_surface_crop = crop_data['global_surface_crop']
        
        surface_a_indices = (mask_crop == cell_a_id) & global_surface_crop
        surface_b_indices = (mask_crop == cell_b_id) & global_surface_crop
        
        if surface_a_indices.any() and surface_b_indices.any():
            surface_a_mask = (mask_crop == cell_a_id)
            structure = generate_binary_structure(2, 2)  # 8-connectivity
            surface_a_only = surface_a_mask & ~binary_erosion(surface_a_mask, structure=structure)
            
            dist_transform = distance_transform_edt(~surface_a_only, sampling=pixel_size_um)
            
            min_distance = dist_transform[surface_b_indices].min()
            
            if min_distance <= max_distance_um:
                results.append({
                    'cell_id_a': cell_a_id,
                    'cell_id_b': cell_b_id,
                    'cell_type_a': cell_type_map.get(cell_a_id, 'Unknown'),
                    'cell_type_b': cell_type_map.get(cell_b_id, 'Unknown'),
                    'surface_distance_um': min_distance
                })
        
        if pbar.n % 1000 == 0:
            pbar.refresh()
    
    print(f"Found {len(results)} near-neighbor pairs within {max_distance_um} μm")
    return results

## Graph database


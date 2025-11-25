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

def create_graph_database(db_path: str = 'cell_neighbor_pair_graph.db') -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS neighbors")
    cursor.execute("DROP TABLE IF EXISTS cells")
    
    # Cell (Node) table
    cursor.execute("""
    CREATE TABLE cells (
        cell_id INTEGER PRIMARY KEY,
        cell_type TEXT,
        centroid_x REAL,
        centroid_y REAL
    )
    """)
    
    # Neighbor (Edge) table
    cursor.execute("""
    CREATE TABLE neighbors (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cell_id_a INTEGER,
        cell_id_b INTEGER,
        cell_type_a TEXT,
        cell_type_b TEXT,
        surface_distance_um REAL,
        FOREIGN KEY(cell_id_a) REFERENCES cells(cell_id),
        FOREIGN KEY(cell_id_b) REFERENCES cells(cell_id),
        UNIQUE(cell_id_a, cell_id_b)
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_cell_type ON cells(cell_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_centroid ON cells(centroid_x, centroid_y)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_a ON neighbors(cell_id_a)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_b ON neighbors(cell_id_b)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_types ON neighbors(cell_type_a, cell_type_b)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_unique_pair ON neighbors(cell_id_a, cell_id_b)")
    
    conn.commit()
    return conn

def populate_cells_table_2d(conn: sqlite3.Connection, metadata_df: pd.DataFrame, 
                        cell_id: str = 'cell_id', 
                        cell_type: str = 'subclass',
                        centroid_x: str = 'X',
                        centroid_y: str = 'Y') -> None:
    required_cols = [cell_id, cell_type, centroid_x, centroid_y]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    cells_data = metadata_df[[cell_id, cell_type, centroid_x, centroid_y]].copy()
    cells_data.columns = ['cell_id', 'cell_type', 'centroid_x', 'centroid_y']
    
    cells_data.to_sql('cells', conn, if_exists='replace', index=False)
    print(f"Populated cells table with {len(cells_data)} cells")

def get_cells_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM cells", conn)

def query_cell_type_pairs(conn: sqlite3.Connection, cell_type_a: str, cell_type_b: str) -> pd.DataFrame:
    query = f"""
    SELECT pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b, surface_distance_um
    FROM neighbors
    WHERE (cell_type_a = '{cell_type_a}' AND cell_type_b = '{cell_type_b}')
       OR (cell_type_a = '{cell_type_b}' AND cell_type_b = '{cell_type_a}')
    ORDER BY pair_id
    """
    
    return pd.read_sql_query(query, conn)

def export_graph_tables(conn: sqlite3.Connection, cells_file: str = 'cells_node_table.csv', 
                       neighbors_file: str = 'neighbors_edge_table.csv') -> None:
    df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
    df_cells.to_csv(cells_file, index=False)
    print(f"Cells table saved as '{cells_file}'")
    
    df_neighbors = pd.read_sql_query("SELECT * FROM neighbors", conn)
    df_neighbors.to_csv(neighbors_file, index=False)
    print(f"Neighbors table saved as '{neighbors_file}'")

def get_graph_statistics(conn: sqlite3.Connection) -> Dict[str, Any]:
    df_cells = pd.read_sql_query("SELECT COUNT(*) as cell_count FROM cells", conn)
    cell_count = df_cells.iloc[0]['cell_count']
    
    df_edges = pd.read_sql_query("SELECT COUNT(*) as edge_count FROM neighbors", conn)
    edge_count = df_edges.iloc[0]['edge_count']
    
    df_cell_types = pd.read_sql_query("SELECT DISTINCT cell_type FROM cells", conn)
    cell_type_count = len(df_cell_types)
    
    df_cell_type_dist = pd.read_sql_query(
        "SELECT cell_type, COUNT(*) as count FROM cells GROUP BY cell_type",
        conn
    )
    
    df_neighbor_types = pd.read_sql_query("""
        SELECT cell_type_a, cell_type_b, COUNT(*) as pair_count 
        FROM neighbors 
        GROUP BY cell_type_a, cell_type_b 
        ORDER BY pair_count DESC
    """, conn)
    
    return {
        'total_cells': cell_count,
        'total_edges': edge_count,
        'unique_cell_types': cell_type_count,
        'cell_type_distribution': df_cell_type_dist.to_dict('records'),
        'neighbor_type_pairs': df_neighbor_types.to_dict('records')
    }

def export_to_anndata(conn: sqlite3.Connection, output_file: str = 'cell_neighbor_graph.h5ad') -> Optional[ad.AnnData]:
    if not ANNDATA_AVAILABLE:
        print("Error: AnnData not available. Install with: pip install anndata")
        return None
    
    df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
    df_cells.set_index('cell_id', inplace=True)
    
    df_neighbors = pd.read_sql_query("SELECT * FROM neighbors", conn)
    
    n_cells = len(df_cells)
    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=int)
    
    cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(df_cells.index)}
    
    for _, row in df_neighbors.iterrows():
        idx_a = cell_id_to_idx.get(row['cell_id_a'])
        idx_b = cell_id_to_idx.get(row['cell_id_b'])
        if idx_a is not None and idx_b is not None:
            adjacency_matrix[idx_a, idx_b] = 1
            adjacency_matrix[idx_b, idx_a] = 1  # Undirected graph
    
    from scipy.sparse import csr_matrix
    sparse_adjacency = csr_matrix(adjacency_matrix)
    
    adata = ad.AnnData(
        X=sparse_adjacency,  # Adjacency matrix
        obs=df_cells,  # Cell metadata
        var=df_cells.copy(),  # Same metadata for variables
        obsp={'spatial_connectivities': sparse_adjacency}  # Store in obsp
    )
    
    adata.uns['neighbor_graph_info'] = {
        'total_cells': n_cells,
        'total_neighbor_pairs': len(df_neighbors),
        'graph_type': 'undirected',
        'distance_threshold_um': 'defined_during_construction',
        'construction_method': 'surface_distance_based_2d'
    }
    
    adata.uns['neighbor_pairs'] = df_neighbors.to_dict('records')
    
    try:
        adata.write(output_file)
        print(f"AnnData object saved to '{output_file}'")
        print(f"  - {n_cells} cells")
        print(f"  - {len(df_neighbors)} neighbor pairs")
        print(f"  - Adjacency matrix shape: {adjacency_matrix.shape}")
        return adata
    except Exception as e:
        print(f"Error saving AnnData file: {e}")
        return adata

def find_all_neighbors_by_surface_distance_2d(
    global_mask: np.ndarray,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 1.0,
    pixel_size_um: float = 0.1085,
    centroid_prefilter_radius_um: float = 75.0,
    n_jobs: int = 1,
    cell_type: str = 'subclass'
) -> pd.DataFrame:
    
    required_cols = ['cell_id', cell_type, 'X', 'Y']
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    cell_type_map = dict(zip(metadata_df['cell_id'], metadata_df[cell_type]))
    
    print(f"Step 1: Finding touching cells using 4-connectivity direct pixel adjacency...")
    touching_pairs = find_touching_neighbors_2d(global_mask, {}, n_jobs)
    
    neighbor_data = []
    for cell_a_id, cell_b_id in touching_pairs:
        cell_a_type = cell_type_map.get(cell_a_id, 'Unknown')
        cell_b_type = cell_type_map.get(cell_b_id, 'Unknown')
        
        neighbor_data.append({
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'cell_a_type': cell_a_type,
            'cell_b_type': cell_b_type,
            'surface_distance_um': 0.0  # Touching cells have distance = 0
        })
    
    touching_count = len(neighbor_data)
    print(f"Found {touching_count} touching neighbor pairs")
    
    if max_distance_um == 0.0:
        print(f"Returning only touching cells (max_distance_um = 0.0)")
        neighbor_df = pd.DataFrame(neighbor_data)
        return neighbor_df
    
    print(f"Step 2: Finding additional neighbors within {max_distance_um} μm...")
    print(f"Using centroid pre-filter radius: {centroid_prefilter_radius_um} μm")
    
    from joblib import Parallel, delayed
    
    centroids = metadata_df[['Y', 'X']].values  # Note: Y, X order for 2D
    scaled_centroids = centroids * pixel_size_um
    kdtree = cKDTree(scaled_centroids)
    cell_ids = metadata_df['cell_id'].values
    
    touching_pairs_set = touching_pairs
    
    def process_cell_pair(cell_a_idx):
        cell_a_id = cell_ids[cell_a_idx]
        cell_a_centroid = scaled_centroids[cell_a_idx]
        
        candidate_indices = kdtree.query_ball_point(
            cell_a_centroid, 
            r=centroid_prefilter_radius_um
        )
        
        neighbors = []
        for cell_b_idx in candidate_indices:
            if cell_b_idx <= cell_a_idx:  # Avoid duplicate pairs
                continue
                
            cell_b_id = cell_ids[cell_b_idx]
            
            if (cell_a_id, cell_b_id) in touching_pairs_set or (cell_b_id, cell_a_id) in touching_pairs_set:
                continue # Skip if pair is aleady touching
            
            try:
                distance = compute_surface_to_surface_distance_2d(
                    global_mask, cell_a_id, cell_b_id, pixel_size_um, max_distance_um
                )
                
                if distance <= max_distance_um:
                    neighbors.append({
                        'cell_a_id': cell_a_id,
                        'cell_b_id': cell_b_id,
                        'cell_a_type': metadata_df.iloc[cell_a_idx][cell_type],
                        'cell_b_type': metadata_df.iloc[cell_b_idx][cell_type],
                        'surface_distance_um': distance
                    })
            except Exception as e:
                print(f"Error computing optimized distance for pair ({cell_a_id}, {cell_b_id}): {e}")
                continue
        
        return neighbors
    
    # Optimization: parallel processing of all cells
    print(f"Processing {len(cell_ids)} cells for non-touching neighbors...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_pair)(i) for i in tqdm(range(len(cell_ids)), desc="Finding non-touching neighbors...")
    )
    
    for cell_neighbors in results:
        neighbor_data.extend(cell_neighbors)
    
    neighbor_df = pd.DataFrame(neighbor_data)
    
    total_count = len(neighbor_df)
    non_touching_count = total_count - touching_count
    
    print(f"Neighbor detection complete:")
    print(f"  - Touching neighbors: {touching_count}")
    print(f"  - Non-touching neighbors within {max_distance_um} μm: {non_touching_count}")
    print(f"  - Total neighbors: {total_count}")
    
    if non_touching_count > 0:
        non_touching_df = neighbor_df[neighbor_df['surface_distance_um'] > 0.0]
        print(f"Distance statistics for non-touching neighbors:")
        print(f"  Min distance: {non_touching_df['surface_distance_um'].min():.3f} μm")
        print(f"  Max distance: {non_touching_df['surface_distance_um'].max():.3f} μm")
        print(f"  Mean distance: {non_touching_df['surface_distance_um'].mean():.3f} μm")
    
    return neighbor_df

def build_cell_graph_database_2d(
    polygon_mask: dict,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 1.0,
    pixel_size_um: float = 0.1085,
    centroid_prefilter_radius_um: float = 75.0,
    db_path: str = 'cell_neighbor_pair_graph.db',
    cell_id: str = 'cell_id',
    cell_type: str = 'subclass',
    centroid_x: str = 'X',
    centroid_y: str = 'Y',
    n_jobs: int = 1,
    save_surfaces_pickle: str = None,
    load_surfaces_pickle: str = None,
    save_graph_state_pickle: str = None
) -> sqlite3.Connection:
    from joblib import Parallel, delayed
    
    print(f"Building optimized cell neighbor graph")
    print(f"Max distance threshold: {max_distance_um} μm")
    print(f"Centroid pre-filter radius: {centroid_prefilter_radius_um} μm")
    print(f"Using 8-neighborhood global surface + bounding box cropping for surface extraction")
    print(f"Using cropped EDT computation with A-bbox+halo and reuse for A's B's")
    
    print("Building global mask from polygons...")
    global_mask, mask_shape, cell_id_mapping = build_global_mask_2d(polygon_mask)
    
    if metadata_df[cell_id].dtype == object or metadata_df[cell_id].dtype.name == 'string':
        reverse_mapping = {v: k for k, v in cell_id_mapping.items()}
        metadata_df = metadata_df.copy()
        metadata_df['mapped_cell_id'] = metadata_df[cell_id].map(cell_id_mapping)
        metadata_df = metadata_df.dropna(subset=['mapped_cell_id'])
        metadata_df['mapped_cell_id'] = metadata_df['mapped_cell_id'].astype(int)
        cell_id_col = 'mapped_cell_id'
    else:
        cell_id_col = cell_id
    
    conn = create_graph_database(db_path)
    
    cells_data = metadata_df[[cell_id_col, cell_type, centroid_x, centroid_y]].copy()
    cells_data.columns = ['cell_id', 'cell_type', 'centroid_x', 'centroid_y']
    cells_data.to_sql('cells', conn, if_exists='replace', index=False)
    print(f"Populated cells table with {len(cells_data)} cells")
    
    cells_df = get_cells_dataframe(conn)
    cell_ids = cells_df['cell_id'].values.tolist()
    
    if load_surfaces_pickle and os.path.exists(load_surfaces_pickle):
        print(f"Loading pre-computed global surface from: {load_surfaces_pickle}")
        with open(load_surfaces_pickle, 'rb') as f:
            surface_data = pickle.load(f)
        global_surface = surface_data.get('global_surface', surface_data)

        all_bboxes = all_cell_bboxes_2d(global_mask)
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
    else:
        print("Pre-computing global surface and halo-extended bounding boxes...")
        global_surface, all_bboxes_with_halo, all_bboxes = precompute_global_surface_and_halo_bboxes_2d(
            global_mask, max_distance_um, pixel_size_um
        )
    
    centroids = cells_df[['centroid_y', 'centroid_x']].values  # Y, X order
    scaled_centroids = centroids * pixel_size_um
    kdtree = cKDTree(scaled_centroids)
    
    def find_candidate_pairs_for_cell(cell_idx):
        cell_id = cell_ids[cell_idx]
        cell_centroid = scaled_centroids[cell_idx]
        
        candidate_indices = kdtree.query_ball_point(
            cell_centroid, 
            r=centroid_prefilter_radius_um
        )
        
        pairs = []
        for candidate_idx in candidate_indices:
            candidate_id = cell_ids[candidate_idx]
            if candidate_idx > cell_idx:
                pairs.append((cell_id, candidate_id))
        
        return pairs
    
    print("Finding candidate pairs using centroid pre-filtering...")
    all_candidate_pairs = []
    
    if n_jobs == 1:
        # Sequential processing
        pbar = tqdm(range(len(cell_ids)), desc="Finding candidate pairs", 
                   unit="cells", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        for cell_idx in pbar:
            pairs = find_candidate_pairs_for_cell(cell_idx)
            all_candidate_pairs.extend(pairs)
            if pbar.n % 10 == 0:
                pbar.refresh()
    else:
        # Parallel processing
        pbar = tqdm(range(len(cell_ids)), desc="Finding candidate pairs", 
                   unit="cells", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        results = Parallel(n_jobs=n_jobs)(
            delayed(find_candidate_pairs_for_cell)(cell_idx) 
            for cell_idx in pbar
        )
        for pairs in results:
            all_candidate_pairs.extend(pairs)
    
    print(f"Found {len(all_candidate_pairs)} candidate pairs")
    
    # Hybrid approach: touching cells + near-neighbors
    print("Using hybrid approach: touching cells (4-connectivity) + near-neighbors...")
    
    print("Step 1: Finding touching cells...")
    touching_pairs = find_touching_neighbors_2d(global_mask, all_bboxes, n_jobs)
    
    touching_neighbor_data = []
    cell_type_map = dict(zip(cells_df['cell_id'], cells_df['cell_type']))
    
    for cell_a_id, cell_b_id in touching_pairs:
        cell_a_type = cell_type_map.get(cell_a_id, 'Unknown')
        cell_b_type = cell_type_map.get(cell_b_id, 'Unknown')
        
        touching_neighbor_data.append({
            'cell_id_a': cell_a_id,
            'cell_id_b': cell_b_id,
            'cell_type_a': cell_a_type,
            'cell_type_b': cell_b_type,
            'surface_distance_um': 0.0
        })
    
    touching_count = len(touching_neighbor_data)
    print(f"Found {touching_count} touching neighbor pairs")
    
    if max_distance_um > 0.0:
        print(f"Step 2: Finding additional neighbors within {max_distance_um} μm...")
        
        touching_pairs_set = touching_pairs
        
        non_touching_candidates = []
        for cell_a_id, cell_b_id in all_candidate_pairs:
            if (cell_a_id, cell_b_id) not in touching_pairs_set and (cell_b_id, cell_a_id) not in touching_pairs_set:
                non_touching_candidates.append((cell_a_id, cell_b_id))
        
        print(f"Processing {len(non_touching_candidates)} non-touching candidate pairs...")
        
        near_neighbor_pairs = compute_surface_distances_batch_2d(
            global_surface, non_touching_candidates, pixel_size_um, max_distance_um, cells_df, global_mask, all_bboxes_with_halo, n_jobs
        )
        
        neighbor_pairs = touching_neighbor_data + near_neighbor_pairs
        
        near_neighbor_count = len(near_neighbor_pairs)
        print(f"Found {near_neighbor_count} additional near-neighbor pairs")
        print(f"Total neighbor pairs: {touching_count} touching + {near_neighbor_count} near-neighbors = {len(neighbor_pairs)}")
    else:
        print("Returning only touching cells (max_distance_um = 0.0)")
        neighbor_pairs = touching_neighbor_data
    
    if neighbor_pairs:
        cursor = conn.cursor()
        neighbor_data = []
        for n in neighbor_pairs:
            neighbor_data.append((
                n['cell_id_a'], 
                n['cell_id_b'], 
                n['cell_type_a'], 
                n['cell_type_b'],
                n.get('surface_distance_um', 0.0)
            ))
        
        try:
            cursor.executemany(
                "INSERT INTO neighbors (cell_id_a, cell_id_b, cell_type_a, cell_type_b, surface_distance_um) VALUES (?, ?, ?, ?, ?)", 
                neighbor_data
            )
            conn.commit()
            print(f"Successfully inserted {len(neighbor_pairs)} neighbor pairs into database")
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                print(f"Warning: Some duplicate pairs were found and skipped due to UNIQUE constraint")
                inserted_count = 0
                for data in neighbor_data:
                    try:
                        cursor.execute(
                            "INSERT INTO neighbors (cell_id_a, cell_id_b, cell_type_a, cell_type_b, surface_distance_um) VALUES (?, ?, ?, ?, ?)", 
                            data
                        )
                        inserted_count += 1
                    except sqlite3.IntegrityError:
                        continue  # Skip duplicates
                conn.commit()
                print(f"Successfully inserted {inserted_count} unique neighbor pairs into database")
            else:
                raise e
    else:
        print("No neighbor pairs found within threshold distance")
    
    if save_surfaces_pickle:
        print(f"Saving global surface to: {save_surfaces_pickle}")
        surface_data = {
            'global_surface': global_surface,
            'metadata': {
                'shape': global_surface.shape,
                'total_surface_pixels': global_surface.sum(),
                'description': 'Global surface mask for all cells (2D)'
            }
        }
        with open(save_surfaces_pickle, 'wb') as f:
            pickle.dump(surface_data, f)
        
        halo_bboxes_pickle = save_surfaces_pickle.replace('global_surface', 'halo_bboxes')
        print(f"Saving halo bounding boxes to: {halo_bboxes_pickle}")
        halo_bboxes_data = {
            'all_bboxes_with_halo': all_bboxes_with_halo,
            'cell_id_mapping': cell_id_mapping,
            'metadata': {
                'total_cells': len(all_bboxes_with_halo),
                'max_distance_um': max_distance_um,
                'pixel_size_um': pixel_size_um,
                'description': 'Halo-extended bounding boxes for all cells (2D)'
            }
        }
        with open(halo_bboxes_pickle, 'wb') as f:
            pickle.dump(halo_bboxes_data, f)
    
    if save_graph_state_pickle:
        print(f"Saving complete graph state to: {save_graph_state_pickle}")

        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors", conn)
        neighbor_pairs = df_neighbors.to_dict('records')
        
        parameters = {
            'max_distance_um': max_distance_um,
            'pixel_size_um': pixel_size_um,
            'centroid_prefilter_radius_um': centroid_prefilter_radius_um,
            'n_jobs': n_jobs
        }
        
        graph_state = {
            'surfaces': {'global_surface': global_surface},
            'neighbor_pairs': neighbor_pairs,
            'metadata_df': metadata_df,
            'parameters': parameters,
            'timestamp': pd.Timestamp.now()
        }
        
        with open(save_graph_state_pickle, 'wb') as f:
            pickle.dump(graph_state, f)
        print(f"Complete graph state saved to: {save_graph_state_pickle}")
    
    return conn

def _load_polygons_from_file(polygon_file_path: str, cell_id_key: str = 'cell_id') -> Dict[str, list]:
    try:
        with open(polygon_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading polygon file: {e}")
    
    polygon_mask = {}
    
    if isinstance(data, dict) and data.get('type') == 'FeatureCollection' and 'features' in data:
        # GeoJSON input format
        print("Detected GeoJSON format")
        for feature in data['features']:
            if feature.get('type') != 'Feature':
                continue
            
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if cell_id_key in properties:
                cell_id = str(properties[cell_id_key])
            elif 'id' in properties:
                cell_id = str(properties['id'])
            elif 'cell_id' in properties:
                cell_id = str(properties['cell_id'])
            else:
                cell_id = str(len(polygon_mask))
            
            geom_type = geometry.get('type', '')
            coords = geometry.get('coordinates', [])
            
            if geom_type == 'Polygon':
                if len(coords) > 0 and len(coords[0]) > 0:
                    polygon_coords = [[float(x), float(y)] for x, y in coords[0]]
                    if len(polygon_coords) > 1 and polygon_coords[0] == polygon_coords[-1]:
                        polygon_coords = polygon_coords[:-1]
                    polygon_mask[cell_id] = polygon_coords
            elif geom_type == 'MultiPolygon':
                if len(coords) > 0:
                    largest_poly = max(coords, key=lambda p: len(p[0]) if len(p) > 0 and len(p[0]) > 0 else 0)
                    if len(largest_poly) > 0 and len(largest_poly[0]) > 0:
                        polygon_coords = [[float(x), float(y)] for x, y in largest_poly[0]]
                        if len(polygon_coords) > 1 and polygon_coords[0] == polygon_coords[-1]:
                            polygon_coords = polygon_coords[:-1]
                        polygon_mask[cell_id] = polygon_coords
    else:
        # JSON input format
        print("Detected JSON format")
        polygon_mask = data
    
    return polygon_mask


def create_neighbor_edge_table_database_2d(
    polygon_json_path: str,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 1.0,
    pixel_size_um: float = 0.1085,
    centroid_prefilter_radius_um: float = 75.0,
    db_path: str = 'cell_neighbor_pair_graph.db',
    cell_id: str = 'cell_id',
    cell_type: str = 'subclass',
    centroid_x: str = 'X',
    centroid_y: str = 'Y',
    output_csv: str = None,
    output_anndata: str = None,
    n_jobs: int = 1,
    save_surfaces_pickle: str = None,
    load_surfaces_pickle: str = None,
    save_graph_state_pickle: str = None
) -> sqlite3.Connection:
    print(f"Loading polygon mask from: {polygon_json_path}")
    
    try:
        polygon_mask = _load_polygons_from_file(polygon_json_path, cell_id_key=cell_id)
    except Exception as e:
        raise ValueError(f"Error loading polygon file: {e}")
    
    print(f"Loaded {len(polygon_mask)} polygons")
    
    valid_polygons = {}
    for poly_cell_id, polygon in polygon_mask.items():
        polygon = np.array(polygon)
        if polygon.shape[0] >= 3:
            unique_points = np.unique(polygon.round(decimals=6), axis=0)
            if unique_points.shape[0] >= 3:
                valid_polygons[poly_cell_id] = polygon.tolist()
    
    print(f"Valid polygons: {len(valid_polygons)}")
    
    required_cols = [cell_id, cell_type, centroid_x, centroid_y]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        print(f"Available columns in metadata: {list(metadata_df.columns)}")
        print(f"Required columns: {required_cols}")
        raise ValueError(f"Missing required columns in metadata: {missing_cols}. Available columns: {list(metadata_df.columns)}")

    metadata_df = metadata_df.copy()
    metadata_df[cell_id] = metadata_df[cell_id].astype(str)
    
    polygon_cell_ids = set(valid_polygons.keys())
    
    metadata_df = metadata_df[metadata_df[cell_id].isin(polygon_cell_ids)].copy()
        
    metadata_cell_ids = set(metadata_df[cell_id].astype(str))
    valid_polygons = {cid: poly for cid, poly in valid_polygons.items() if cid in metadata_cell_ids}
        
    if len(metadata_df) == 0:
        raise ValueError("No cells found that have both polygons and metadata. Check that cell IDs match between JSON and CSV files.")
    
    conn = build_cell_graph_database_2d(
        polygon_mask=valid_polygons,
        metadata_df=metadata_df,
        max_distance_um=max_distance_um,
        pixel_size_um=pixel_size_um,
        centroid_prefilter_radius_um=centroid_prefilter_radius_um,
        db_path=db_path,
        cell_id=cell_id,
        cell_type=cell_type,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        n_jobs=n_jobs,
        save_surfaces_pickle=save_surfaces_pickle,
        load_surfaces_pickle=load_surfaces_pickle,
        save_graph_state_pickle=save_graph_state_pickle
    )
    
    # Export to CSV
    if output_csv:
        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors", conn)
        df_neighbors.to_csv(output_csv, index=False)
        print(f"Optimized neighbor edge table saved to: {output_csv}")
    
    # Export to AnnData
    if output_anndata:
        adata = export_to_anndata(conn, output_anndata)
        if adata is not None:
            print(f"AnnData object created and saved to: {output_anndata}")
    
    return conn

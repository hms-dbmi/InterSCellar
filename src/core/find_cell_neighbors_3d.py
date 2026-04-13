# Import

import numpy as np
import pandas as pd
from itertools import product
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from skimage.measure import regionprops
import skimage.measure
sk_label = skimage.measure.label
from scipy.ndimage import label, generate_binary_structure, binary_erosion, distance_transform_edt, find_objects
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import cv2
import csv
import sqlite3
import os
import pickle
import traceback
import math
import zarr

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Cell surface precomputation

def get_bounding_boxes_3d(mask_3d: np.ndarray, unique_ids: set) -> dict:
    z, y, x = np.nonzero(mask_3d)
    cell_ids = mask_3d[z, y, x]
    df = pd.DataFrame({'cell_id': cell_ids, 'z': z, 'y': y, 'x': x})
    bbox = {}
    grouped = df.groupby('cell_id')
    for cell_id, group in grouped:
        if cell_id == 0:
            continue
        minz, maxz = group['z'].min(), group['z'].max() + 1
        miny, maxy = group['y'].min(), group['y'].max() + 1
        minx, maxx = group['x'].min(), group['x'].max() + 1
        bbox[cell_id] = (slice(minz, maxz), slice(miny, maxy), slice(minx, maxx))
    return bbox

def compute_bounding_box_with_halo(
    surface_a: np.ndarray,
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[slice, slice, slice]:
    z_coords, y_coords, x_coords = np.where(surface_a)
    
    if len(z_coords) == 0:
        return None
    
    min_z, max_z = z_coords.min(), z_coords.max() + 1
    min_y, max_y = y_coords.min(), y_coords.max() + 1
    min_x, max_x = x_coords.min(), x_coords.max() + 1
    
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    min_z_pad = max(0, min_z - pad_z)
    max_z_pad = max_z + pad_z + 1
    min_y_pad = max(0, min_y - pad_y)
    max_y_pad = max_y + pad_y + 1
    min_x_pad = max(0, min_x - pad_x)
    max_x_pad = max_x + pad_x + 1
    
    return (slice(min_z_pad, max_z_pad), 
            slice(min_y_pad, max_y_pad), 
            slice(min_x_pad, max_x_pad))

def global_surface_26n(mask_3d: np.ndarray) -> np.ndarray:
    print("Computing global surface mask...")
    
    structure = generate_binary_structure(3, 3)  # 26-connectivity
    binary_mask = (mask_3d > 0).astype(bool)
    eroded = binary_erosion(binary_mask, structure=structure)
    global_surface = binary_mask & ~eroded
    
    print(f"Global surface mask computed: {global_surface.sum()} surface voxels")
    return global_surface

def all_cell_bboxes(mask_3d: np.ndarray) -> Dict[int, Tuple[slice, slice, slice]]:
    print("Computing bounding boxes for all cells...")
    
    unique_ids = set(np.unique(mask_3d))
    unique_ids.discard(0)
    
    bboxes = get_bounding_boxes_3d(mask_3d, unique_ids)
    
    print(f"Bounding boxes computed for {len(bboxes)} cells")
    return bboxes

def precompute_global_surface_and_halo_bboxes(
    mask_3d: np.ndarray, 
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[np.ndarray, Dict[int, Tuple[slice, slice, slice]]]:
    print("Pre-computing global surface and halo-extended bounding boxes...")
    
    global_surface = global_surface_26n(mask_3d)
    
    all_bboxes = all_cell_bboxes(mask_3d)
    
    print("Pre-computing halo-extended bounding boxes...")
    all_bboxes_with_halo = {}
    
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    for cell_id, bbox in all_bboxes.items():
        slice_z, slice_y, slice_x = bbox
        
        z_start = max(0, slice_z.start - pad_z)
        z_stop = min(mask_3d.shape[0], slice_z.stop + pad_z)
        y_start = max(0, slice_y.start - pad_y)
        y_stop = min(mask_3d.shape[1], slice_y.stop + pad_y)
        x_start = max(0, slice_x.start - pad_x)
        x_stop = min(mask_3d.shape[2], slice_x.stop + pad_x)
        
        all_bboxes_with_halo[cell_id] = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Pre-computed halo-extended bounding boxes for {len(all_bboxes_with_halo)} cells")
    print(f"Using only halo-extended bboxes for all operations")
    
    return global_surface, all_bboxes_with_halo, all_bboxes

## Surface-based neighbor identification

def cell_neighbor_candidate_centroid_distance_kdtree(conn: sqlite3.Connection, 
                                                   cell_id: int,
                                                   radius_um: float = 75.0,
                                                   voxel_size_um: tuple = (0.56, 0.28, 0.28)) -> Set[int]:
    from scipy.spatial import cKDTree
    
    if isinstance(cell_id, np.ndarray):
        cell_id = int(cell_id.ravel()[0])
    else:
        cell_id = int(cell_id)

    metadata_df = get_cells_dataframe(conn)

    if cell_id not in metadata_df['cell_id'].values:
        raise ValueError(f"Cell ID {cell_id} not found in database")

    coords = metadata_df[['centroid_z', 'centroid_y', 'centroid_x']].values
    scaled_coords = coords * np.array(voxel_size_um)
    kdtree = cKDTree(scaled_coords)

    target_idx = metadata_df.index[metadata_df['cell_id'] == cell_id][0]
    target_point = scaled_coords[target_idx]
    indices = kdtree.query_ball_point(target_point, r=radius_um)
    candidate_neighbor_ids = set(metadata_df.iloc[i]['cell_id'] for i in indices if metadata_df.iloc[i]['cell_id'] != cell_id)

    return candidate_neighbor_ids

def find_touching_neighbors_direct_adjacency(mask_3d: np.ndarray, all_bboxes: dict, n_jobs: int = 1) -> set:
    import numpy as np
    from tqdm import tqdm

    print("Finding touching neighbors...")

    labels = mask_3d
    z_dim, y_dim, x_dim = labels.shape

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

    # Z-axis face adjacency: compare slice z with z+1
    for z in tqdm(range(z_dim - 1), desc="6-conn touching: Z faces", ncols=100):
        a = labels[z + 1, :, :]
        b = labels[z, :, :]
        add_pairs(a, b)

    # Y-axis face adjacency: within each z-slice, compare row y with y+1
    for z in tqdm(range(z_dim), desc="6-conn touching: Y faces", ncols=100):
        s = labels[z]
        a = s[1:, :]
        b = s[:-1, :]
        add_pairs(a, b)

    # X-axis face adjacency: within each z-slice, compare col x with x+1
    for z in tqdm(range(z_dim), desc="6-conn touching: X faces", ncols=100):
        s = labels[z]
        a = s[:, 1:]
        b = s[:, :-1]
        add_pairs(a, b)

    print(f"Identified {len(touching_pairs)} touching neighbor pairs.")
    return touching_pairs

def find_all_neighbors_by_surface_distance_3d(
    mask_3d: np.ndarray,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    n_jobs: int = 1
) -> pd.DataFrame:
    required_cols = ['CellID', 'phenotype', 'Z_centroid', 'Y_centroid', 'X_centroid']
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    cell_type_map = dict(zip(metadata_df['CellID'], metadata_df['phenotype']))
    
    print(f"Step 1: Finding touching cells...")
    touching_pairs = find_touching_neighbors_direct_adjacency(mask_3d, n_jobs)
    
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
    
    print(f"Step 2: Finding non-touching neighbors within {max_distance_um} μm...")
    print(f"Using centroid pre-filter radius: {centroid_prefilter_radius_um} μm")
    
    from joblib import Parallel, delayed
    
    centroids = metadata_df[['Z_centroid', 'Y_centroid', 'X_centroid']].values
    scaled_centroids = centroids * np.array(voxel_size_um)
    kdtree = cKDTree(scaled_centroids)
    cell_ids = metadata_df['CellID'].values
    
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
                continue # Skip pair if already touching
            
            try:
                distance = compute_surface_to_surface_distance_3d(
                    mask_3d, cell_a_id, cell_b_id, voxel_size_um, max_distance_um
                )
                
                if distance <= max_distance_um:
                    neighbors.append({
                        'cell_a_id': cell_a_id,
                        'cell_b_id': cell_b_id,
                        'cell_a_type': metadata_df.iloc[cell_a_idx]['phenotype'],
                        'cell_b_type': metadata_df.iloc[cell_b_idx]['phenotype'],
                        'surface_distance_um': distance
                    })
            except Exception as e:
                print(f"Error computing distance for pair ({cell_a_id}, {cell_b_id}): {e}")
                continue
        
        return neighbors
    
    # Optimization: parallel processing
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

## Surface-to-surface distance computation

def compute_surface_to_surface_distance_3d(
    mask_3d: np.ndarray, 
    cell_a_id: int, 
    cell_b_id: int, 
    voxel_size_um: tuple,
    max_distance_um: float = float('inf')
) -> float:
    mask_a = (mask_3d == cell_a_id)
    mask_b = (mask_3d == cell_b_id)
    
    if not mask_a.any() or not mask_b.any():
        return float('inf')
    
    surface_a = mask_a & ~binary_erosion(mask_a)
    surface_b = mask_b & ~binary_erosion(mask_b)
    
    if not surface_a.any() or not surface_b.any():
        return float('inf')
    
    bbox_with_halo = compute_bounding_box_with_halo(surface_a, max_distance_um, voxel_size_um)
    
    if bbox_with_halo is None:
        return float('inf')
    
    slice_z, slice_y, slice_x = bbox_with_halo
    surface_a_crop = surface_a[slice_z, slice_y, slice_x]
    surface_b_crop = surface_b[slice_z, slice_y, slice_x]
    
    if not surface_b_crop.any():
        return float('inf')
    
    dist_transform_crop = distance_transform_edt(~surface_a_crop, sampling=voxel_size_um)
    
    min_distance = dist_transform_crop[surface_b_crop].min()
    return min_distance

def compute_surface_distances_batch_3d(
    global_surface: np.ndarray,
    cell_pairs: List[Tuple[int, int]],
    voxel_size_um: tuple,
    max_distance_um: float,
    cells_df: pd.DataFrame,
    mask_3d: np.ndarray,
    all_bboxes_with_halo: Dict[int, Tuple[slice, slice, slice]],
    n_jobs: int = 1
) -> List[Dict[str, Any]]:
    from collections import defaultdict
    from joblib import Parallel, delayed
    
    print("Computing surface distances for unique crop regions...")
    print(f"Processing {len(cell_pairs)} cell pairs with max_distance_um = {max_distance_um}")
    
    print("Step 1: Identifying unique crop regions...")
    unique_crops = set()
    cell_to_crop_tuple = {}
    
    for cell_a_id, cell_b_id in cell_pairs:
        if cell_a_id in all_bboxes_with_halo:
            crop_slice = all_bboxes_with_halo[cell_a_id]
            crop_tuple = (crop_slice[0].start, crop_slice[0].stop, 
                         crop_slice[1].start, crop_slice[1].stop,
                         crop_slice[2].start, crop_slice[2].stop)
            unique_crops.add(crop_tuple)
            cell_to_crop_tuple[cell_a_id] = crop_tuple
    
    total_cells = len(set(pair[0] for pair in cell_pairs if pair[0] in all_bboxes_with_halo))
    print(f"Found {len(unique_crops)} unique crop regions")
    
    print(f"Step 2: Computing EDTs for {len(unique_crops)} unique crop regions...")
    crop_edts = {}
    
    def compute_crop_edt(crop_tuple):
        z_start, z_stop, y_start, y_stop, x_start, x_stop = crop_tuple
        crop_slice = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
        
        mask_crop = mask_3d[crop_slice]
        global_surface_crop = global_surface[crop_slice]
        

        dist_transform = distance_transform_edt(~global_surface_crop, sampling=voxel_size_um)
        return crop_tuple, dist_transform, mask_crop, global_surface_crop
    
    if n_jobs == 1:
        # Sequential EDT computation
        pbar = tqdm(unique_crops, desc="Computing EDTs for unique crop regions", 
                   unit="crops", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        for crop_tuple in pbar:
            crop_tuple, dist_transform, mask_crop, global_surface_crop = compute_crop_edt(crop_tuple)
            crop_edts[crop_tuple] = {
                'dist_transform': dist_transform,
                'mask_crop': mask_crop,
                'global_surface_crop': global_surface_crop
            }
    else:
        # Parallel EDT computation
        print(f"Computing EDTs with {n_jobs} parallel jobs...")
        pbar = tqdm(unique_crops, desc="Computing EDTs for unique crop regions in parallel", 
                   unit="crops", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(compute_crop_edt)(crop_tuple) 
            for crop_tuple in pbar
        )
        
        for crop_tuple, dist_transform, mask_crop, global_surface_crop in results_list:
            crop_edts[crop_tuple] = {
                'dist_transform': dist_transform,
                'mask_crop': mask_crop,
                'global_surface_crop': global_surface_crop
            }
    
    print(f"Completed EDT computations for all {len(crop_edts)} unique crop regions")
    
    print("Step 3: Extracting surface-surface distances for all cell pairs...")
    results = []
    cell_type_map = dict(zip(cells_df['cell_id'], cells_df['cell_type']))
    
    pbar = tqdm(cell_pairs, desc="Extracting distances from pre-computed EDTs", 
               unit="pairs", ncols=100, leave=True, mininterval=0.1, maxinterval=1.0)
    
    for cell_a_id, cell_b_id in pbar:
        if cell_a_id not in cell_to_crop_tuple:
            continue
            
        crop_tuple = cell_to_crop_tuple[cell_a_id]
        crop_data = crop_edts[crop_tuple]
        
        dist_transform = crop_data['dist_transform']
        mask_crop = crop_data['mask_crop']
        global_surface_crop = crop_data['global_surface_crop']
        
        surface_a_indices = (mask_crop == cell_a_id) & global_surface_crop
        surface_b_indices = (mask_crop == cell_b_id) & global_surface_crop
        
        if surface_a_indices.any() and surface_b_indices.any():
            min_distance = dist_transform[surface_b_indices].min()
            
            if min_distance <= max_distance_um:
                results.append({
                    'cell_id_a': cell_a_id,
                    'cell_id_b': cell_b_id,
                    'cell_type_a': cell_type_map.get(cell_a_id, 'Unknown'),
                    'cell_type_b': cell_type_map.get(cell_b_id, 'Unknown'),
                    'surface_distance_um': min_distance
                })
        
        if pbar.n % 1000 == 0:  # Update every 1000 iterations
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
        centroid_y REAL,
        centroid_z REAL
    )
    """)
    
    # Neighbor (Edge) table
    cursor.execute("""
    CREATE TABLE neighbors (
        pair_id INTEGER PRIMARY KEY,
        cell_id_a INTEGER,
        cell_id_b INTEGER,
        cell_type_a TEXT,
        cell_type_b TEXT,
        FOREIGN KEY(cell_id_a) REFERENCES cells(cell_id),
        FOREIGN KEY(cell_id_b) REFERENCES cells(cell_id),
        UNIQUE(cell_id_a, cell_id_b)
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_cell_type ON cells(cell_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_centroid ON cells(centroid_x, centroid_y, centroid_z)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_a ON neighbors(cell_id_a)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_b ON neighbors(cell_id_b)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_cell_types ON neighbors(cell_type_a, cell_type_b)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_neighbors_unique_pair ON neighbors(cell_id_a, cell_id_b)")
    
    conn.commit()
    return conn


def renumber_neighbors_pair_ids_sequential(conn: sqlite3.Connection) -> None:
    df = pd.read_sql_query(
        "SELECT cell_id_a, cell_id_b, cell_type_a, cell_type_b FROM neighbors ORDER BY pair_id",
        conn,
    )
    if len(df) == 0:
        return
    rows = list(
        zip(
            range(1, len(df) + 1),
            df["cell_id_a"].astype(int),
            df["cell_id_b"].astype(int),
            df["cell_type_a"],
            df["cell_type_b"],
        )
    )
    cursor = conn.cursor()
    cursor.execute("DELETE FROM neighbors")
    cursor.executemany(
        "INSERT INTO neighbors (pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"Renumbered neighbors.pair_id to contiguous 1..{len(df)}")

def populate_cells_table(conn: sqlite3.Connection, metadata_df: pd.DataFrame, 
                        cell_id: str = 'CellID', 
                        cell_type: str = 'phenotype',
                        centroid_x: str = 'X_centroid',
                        centroid_y: str = 'Y_centroid', 
                        centroid_z: str = 'Z_centroid') -> None:
    required_cols = [cell_id, cell_type, centroid_x, centroid_y, centroid_z]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    cells_data = metadata_df[[cell_id, cell_type, centroid_x, centroid_y, centroid_z]].copy()    
    cells_data.columns = ['cell_id', 'cell_type', 'centroid_x', 'centroid_y', 'centroid_z']

    cells_data.to_sql('cells', conn, if_exists='replace', index=False)
    print(f"Populated cells table with {len(cells_data)} cells using unified column names")

def get_cells_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM cells", conn)

def query_cell_type_pairs(conn: sqlite3.Connection, cell_type_a: str, cell_type_b: str) -> pd.DataFrame:
    query = f"""
    SELECT pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b
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

    df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
    df_neighbors.to_csv(neighbors_file, index=False)
    print(f"Neighbors table saved as '{neighbors_file}'")

def export_to_anndata(conn: sqlite3.Connection, output_file: str = 'cell_neighbor_graph.h5ad') -> Optional[ad.AnnData]:
    if not ANNDATA_AVAILABLE:
        print("Error: AnnData not available. Install with: pip install anndata")
        return None
    try:
        df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
        df_cells["cell_id"] = df_cells["cell_id"].astype(str)
        df_cells.set_index("cell_id", inplace=True)

        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)

        n_cells = len(df_cells)
        adjacency_matrix = np.zeros((n_cells, n_cells), dtype=int)

        cell_id_to_idx = {str(cid): idx for idx, cid in enumerate(df_cells.index)}

        for _, row in df_neighbors.iterrows():
            idx_a = cell_id_to_idx.get(str(row["cell_id_a"]))
            idx_b = cell_id_to_idx.get(str(row["cell_id_b"]))
            if idx_a is not None and idx_b is not None:
                adjacency_matrix[idx_a, idx_b] = 1
                adjacency_matrix[idx_b, idx_a] = 1  # Undirected graph

        from scipy.sparse import csr_matrix
        sparse_adjacency = csr_matrix(adjacency_matrix)

        adata = ad.AnnData(
            X=sparse_adjacency,
            obs=df_cells,
            var=df_cells.copy(),
            obsp={'spatial_connectivities': sparse_adjacency}
        )

        adata.uns['neighbor_graph_info'] = {
            'total_cells': n_cells,
            'total_neighbor_pairs': len(df_neighbors),
            'graph_type': 'undirected',
            'distance_threshold_um': 'defined_during_construction',
            'construction_method': 'surface_distance_based'
        }

        adata.uns["neighbor_pairs"] = df_neighbors.reset_index(drop=True).copy()

        adata.write(output_file)
        print(f"AnnData object saved to '{output_file}'")
        print(f"  - {n_cells} cells")
        print(f"  - {len(df_neighbors)} neighbor pairs")
        print(f"  - Adjacency matrix shape: {adjacency_matrix.shape}")
        return adata
    except Exception as e:
        print(
            f"Warning: AnnData export failed (non-fatal; SQLite/CSV/pickle outputs are unchanged): {e}"
        )
        traceback.print_exc()
        return None

def get_anndata_from_database(conn: sqlite3.Connection) -> Optional[ad.AnnData]:
    if not ANNDATA_AVAILABLE:
        print("Error: AnnData not available. Install with: pip install anndata")
        return None
    
    df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
    df_cells["cell_id"] = df_cells["cell_id"].astype(str)
    df_cells.set_index("cell_id", inplace=True)
    
    df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
    
    n_cells = len(df_cells)
    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=int)
    
    cell_id_to_idx = {str(cid): idx for idx, cid in enumerate(df_cells.index)}

    for _, row in df_neighbors.iterrows():
        idx_a = cell_id_to_idx.get(str(row["cell_id_a"]))
        idx_b = cell_id_to_idx.get(str(row["cell_id_b"]))
        if idx_a is not None and idx_b is not None:
            adjacency_matrix[idx_a, idx_b] = 1
            adjacency_matrix[idx_b, idx_a] = 1  # Undirected graph
    
    from scipy.sparse import csr_matrix
    sparse_adjacency = csr_matrix(adjacency_matrix)
    
    adata = ad.AnnData(
        X=sparse_adjacency,
        obs=df_cells,
        var=df_cells.copy(),
        obsp={'spatial_connectivities': sparse_adjacency}
    )
    
    adata.uns['neighbor_graph_info'] = {
        'total_cells': n_cells,
        'total_neighbor_pairs': len(df_neighbors),
        'graph_type': 'undirected',
        'construction_method': 'surface_distance_based'
    }
    
    adata.obsp['neighbor_pairs'] = df_neighbors
    
    return adata

def save_edges_to_pickle(all_edges: Set[Tuple[int, int]], filepath: str = "all_edges.pkl") -> None:
    with open(filepath, "wb") as f:
        pickle.dump(all_edges, f)
    print(f"Edges saved to pickle file: {filepath}")

def load_edges_from_pickle(filepath: str = "all_edges.pkl") -> Set[Tuple[int, int]]:
    with open(filepath, "rb") as f:
        edges = pickle.load(f)
    print(f"Loaded {len(edges)} edges from pickle file: {filepath}")
    return edges

def save_neighbor_pairs_to_pickle(neighbor_pairs: List[Dict[str, Any]], filepath: str = "neighbor_pairs.pkl") -> None:
    with open(filepath, "wb") as f:
        pickle.dump(neighbor_pairs, f)
    print(f"Neighbor pairs saved to pickle file: {filepath}")

def load_neighbor_pairs_from_pickle(filepath: str = "neighbor_pairs.pkl") -> List[Dict[str, Any]]:
    with open(filepath, "rb") as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} neighbor pairs from pickle file: {filepath}")
    return pairs

def save_surfaces_to_pickle(surfaces: Dict[int, np.ndarray], filepath: str = "cell_surfaces.pkl") -> None:
    with open(filepath, "wb") as f:
        pickle.dump(surfaces, f)
    print(f"Cell surfaces saved to pickle file: {filepath}")

def load_surfaces_from_pickle(filepath: str = "cell_surfaces.pkl") -> Dict[int, np.ndarray]:
    with open(filepath, "rb") as f:
        surfaces = pickle.load(f)
    print(f"Loaded surfaces for {len(surfaces)} cells from pickle file: {filepath}")
    return surfaces

def save_graph_state_to_pickle(
    surfaces: Dict[int, np.ndarray],
    neighbor_pairs: List[Dict[str, Any]],
    metadata_df: pd.DataFrame,
    parameters: Dict[str, Any],
    filepath: str = "graph_state.pkl"
) -> None:
    graph_state = {
        'surfaces': surfaces,
        'neighbor_pairs': neighbor_pairs,
        'metadata_df': metadata_df,
        'parameters': parameters,
        'timestamp': pd.Timestamp.now()
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(graph_state, f)
    print(f"Complete graph state saved to pickle file: {filepath}")
    print(f"  - {len(surfaces)} cell surfaces")
    print(f"  - {len(neighbor_pairs)} neighbor pairs")
    print(f"  - {len(metadata_df)} cells in metadata")

def load_graph_state_from_pickle(filepath: str = "graph_state.pkl") -> Dict[str, Any]:
    with open(filepath, "rb") as f:
        graph_state = pickle.load(f)
    
    print(f"Loaded complete graph state from pickle file: {filepath}")
    print(f"  - {len(graph_state['surfaces'])} cell surfaces")
    print(f"  - {len(graph_state['neighbor_pairs'])} neighbor pairs")
    print(f"  - {len(graph_state['metadata_df'])} cells in metadata")
    print(f"  - Saved on: {graph_state['timestamp']}")
    
    return graph_state

def export_to_duckdb(conn: sqlite3.Connection, output_file: str = 'cell_neighbor_graph.duckdb') -> None:
    try:
        import duckdb
    except ImportError:
        print("Error: DuckDB not available. Install with: pip install duckdb")
        return

    duckdb_conn = None
    try:
        print(f"Exporting cell neighbor graph to DuckDB: {output_file}")

        df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
        print(f"  - {len(df_cells)} cells (nodes)")

        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
        print(f"  - {len(df_neighbors)} neighbor pairs (edges)")

        duckdb_conn = duckdb.connect(output_file)
        for stmt in (
            "DROP VIEW IF EXISTS cell_connectivity",
            "DROP VIEW IF EXISTS cell_type_interactions",
            "DROP VIEW IF EXISTS spatial_analysis",
            "DROP TABLE IF EXISTS neighbors",
            "DROP TABLE IF EXISTS cells",
            "DROP TABLE IF EXISTS metadata",
        ):
            duckdb_conn.execute(stmt)

        duckdb_conn.execute("""
            CREATE TABLE cells (
                cell_id INTEGER PRIMARY KEY,
                cell_type VARCHAR,
                centroid_x DOUBLE,
                centroid_y DOUBLE,
                centroid_z DOUBLE
            )
        """)

        duckdb_conn.register("df_cells", df_cells)
        duckdb_conn.execute("INSERT INTO cells SELECT * FROM df_cells")
        duckdb_conn.execute("""
            CREATE TABLE neighbors (
                pair_id INTEGER PRIMARY KEY,
                cell_id_a INTEGER,
                cell_id_b INTEGER,
                cell_type_a VARCHAR,
                cell_type_b VARCHAR,
                FOREIGN KEY (cell_id_a) REFERENCES cells(cell_id),
                FOREIGN KEY (cell_id_b) REFERENCES cells(cell_id)
            )
        """)

        duckdb_conn.register("df_neighbors", df_neighbors)
        duckdb_conn.execute("INSERT INTO neighbors SELECT * FROM df_neighbors")
        print("Creating analytical views for graph queries...")

        duckdb_conn.execute("""
            CREATE VIEW cell_connectivity AS
            SELECT 
                c.cell_id,
                c.cell_type,
                c.centroid_x,
                c.centroid_y,
                c.centroid_z,
                COUNT(n1.cell_id_b) + COUNT(n2.cell_id_a) as total_neighbors,
                COUNT(DISTINCT n1.cell_id_b) + COUNT(DISTINCT n2.cell_id_a) as unique_neighbors
            FROM cells c
            LEFT JOIN neighbors n1 ON c.cell_id = n1.cell_id_a
            LEFT JOIN neighbors n2 ON c.cell_id = n2.cell_id_b
            GROUP BY c.cell_id, c.cell_type, c.centroid_x, c.centroid_y, c.centroid_z
        """)

        duckdb_conn.execute("""
            CREATE VIEW cell_type_interactions AS
            SELECT 
                cell_type_a,
                cell_type_b,
                COUNT(*) as interaction_count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM neighbors) as percentage
            FROM neighbors
            GROUP BY cell_type_a, cell_type_b
            ORDER BY interaction_count DESC
        """)

        duckdb_conn.execute("""
            CREATE VIEW spatial_analysis AS
            SELECT 
                n.pair_id,
                n.cell_id_a,
                n.cell_id_b,
                n.cell_type_a,
                n.cell_type_b,
                c1.centroid_x as x_a,
                c1.centroid_y as y_a,
                c1.centroid_z as z_a,
                c2.centroid_x as x_b,
                c2.centroid_y as y_b,
                c2.centroid_z as z_b,
                SQRT(
                    POWER(c1.centroid_x - c2.centroid_x, 2) +
                    POWER(c1.centroid_y - c2.centroid_y, 2) +
                    POWER(c1.centroid_z - c2.centroid_z, 2)
                ) as euclidean_distance
            FROM neighbors n
            JOIN cells c1 ON n.cell_id_a = c1.cell_id
            JOIN cells c2 ON n.cell_id_b = c2.cell_id
        """)

        duckdb_conn.execute("""
            CREATE TABLE metadata (
                key VARCHAR,
                value VARCHAR
            )
        """)

        metadata = [
            ('total_cells', str(len(df_cells))),
            ('total_edges', str(len(df_neighbors))),
            ('unique_cell_types', str(df_cells['cell_type'].nunique())),
            ('export_timestamp', pd.Timestamp.now().isoformat()),
            ('database_type', 'cell_neighbor_graph'),
            ('format', 'duckdb')
        ]

        for key, value in metadata:
            duckdb_conn.execute("INSERT INTO metadata VALUES (?, ?)", [key, value])

        print(f"DuckDB export completed: {output_file}")
        print(f"  - {len(df_cells)} cells (nodes)")
        print(f"  - {len(df_neighbors)} neighbor pairs (edges)")
        print(f"  - Analytical views created for graph analysis")
        print(f"  - Metadata table populated")

        print("\nExample DuckDB queries:")
        print("1. Cell connectivity: SELECT * FROM cell_connectivity ORDER BY total_neighbors DESC LIMIT 10")
        print("2. Cell type interactions: SELECT * FROM cell_type_interactions")
        print("3. Spatial analysis: SELECT * FROM spatial_analysis WHERE euclidean_distance < 10 LIMIT 10")
        print("4. Graph statistics: SELECT * FROM metadata")
    except Exception as e:
        print(
            f"Warning: DuckDB export failed (non-fatal; SQLite/CSV/pickle outputs are unchanged): {e}"
        )
        traceback.print_exc()
    finally:
        if duckdb_conn is not None:
            try:
                duckdb_conn.close()
            except Exception:
                pass

def get_graph_statistics(conn: sqlite3.Connection) -> Dict[str, Any]:
    # Count nodes (cells)
    df_cells = pd.read_sql_query("SELECT COUNT(*) as cell_count FROM cells", conn)
    cell_count = df_cells.iloc[0]['cell_count']
    
    # Count edges (neighbors)
    df_edges = pd.read_sql_query("SELECT COUNT(*) as edge_count FROM neighbors", conn)
    edge_count = df_edges.iloc[0]['edge_count']
    
    # Count unique cell types
    df_cell_types = pd.read_sql_query("SELECT DISTINCT cell_type FROM cells", conn)
    cell_type_count = len(df_cell_types)
    
    # Get cell type distribution
    df_cell_type_dist = pd.read_sql_query(
        "SELECT cell_type, COUNT(*) as count FROM cells GROUP BY cell_type",
        conn
    )
    
    # Get neighbor pair statistics by cell type combinations
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

def build_cell_graph_database_3d(
    mask_3d: np.ndarray,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    db_path: str = 'cell_neighbor_pair_graph.db',
    cell_id: str = 'CellID',
    cell_type: str = 'phenotype',
    centroid_x: str = 'X_centroid',
    centroid_y: str = 'Y_centroid',
    centroid_z: str = 'Z_centroid',
    n_jobs: int = 1,
    save_surfaces_pickle: str = None,
    load_surfaces_pickle: str = None,
    save_graph_state_pickle: str = None
) -> sqlite3.Connection:
    from joblib import Parallel, delayed
    
    print(f"Building cell neighbor graph – use bounding box optimization")
    print(f"Max distance threshold: {max_distance_um} μm")
    print(f"Centroid pre-filter radius: {centroid_prefilter_radius_um} μm")
    print(f"Using 26-neighborhood global surface + bounding box cropping for surface extraction")
    print(f"Using cropped EDT computation with A-bbox+halo and reuse for A's B's")
    
    conn = create_graph_database(db_path)
    populate_cells_table(conn, metadata_df, 
                        cell_id=cell_id,
                        cell_type=cell_type,
                        centroid_x=centroid_x,
                        centroid_y=centroid_y,
                        centroid_z=centroid_z)
    
    cells_df = get_cells_dataframe(conn)
    cell_ids = cells_df['cell_id'].values.tolist()
    
    if load_surfaces_pickle and os.path.exists(load_surfaces_pickle):
        print(f"Loading pre-computed global surface from: {load_surfaces_pickle}")
        global_surface = load_surfaces_from_pickle(load_surfaces_pickle)
        all_bboxes = all_cell_bboxes(mask_3d)
        all_bboxes_with_halo = {}
        pad_z = math.ceil(max_distance_um / voxel_size_um[0])
        pad_y = math.ceil(max_distance_um / voxel_size_um[1])
        pad_x = math.ceil(max_distance_um / voxel_size_um[2])
        for cell_id, bbox in all_bboxes.items():
            slice_z, slice_y, slice_x = bbox
            z_start = max(0, slice_z.start - pad_z)
            z_stop = min(mask_3d.shape[0], slice_z.stop + pad_z)
            y_start = max(0, slice_y.start - pad_y)
            y_stop = min(mask_3d.shape[1], slice_y.stop + pad_y)
            x_start = max(0, slice_x.start - pad_x)
            x_stop = min(mask_3d.shape[2], slice_x.stop + pad_x)
            all_bboxes_with_halo[cell_id] = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    else:
        print("Pre-computing global surface and halo-extended bounding boxes...")
        global_surface, all_bboxes_with_halo, all_bboxes = precompute_global_surface_and_halo_bboxes(mask_3d, max_distance_um, voxel_size_um)
    
    centroids = cells_df[['centroid_z', 'centroid_y', 'centroid_x']].values
    scaled_centroids = centroids * np.array(voxel_size_um)
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
            if pbar.n % 10 == 0:  # Update every 10 iterations
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
    print("Hybrid approach: touching cells (6-connectivity) + near-neighbors...")
    
    print("Step 1: Finding touching cells...")
    touching_pairs = find_touching_neighbors_direct_adjacency(mask_3d, all_bboxes, n_jobs)
    touching_neighbor_data = []
    cell_type_map = dict(zip(cells_df['cell_id'], cells_df['cell_type']))
    
    for cell_a_id, cell_b_id in touching_pairs:
        cell_a_type = cell_type_map.get(cell_a_id, 'Unknown')
        cell_b_type = cell_type_map.get(cell_b_id, 'Unknown')
        
        touching_neighbor_data.append({
            'cell_id_a': cell_a_id,
            'cell_id_b': cell_b_id,
            'cell_type_a': cell_a_type,
            'cell_type_b': cell_b_type
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
        
        near_neighbor_pairs = compute_surface_distances_batch_3d(
            global_surface, non_touching_candidates, voxel_size_um, max_distance_um, cells_df, mask_3d, all_bboxes_with_halo, n_jobs
        )
        
        neighbor_pairs = touching_neighbor_data + near_neighbor_pairs
        
        near_neighbor_count = len(near_neighbor_pairs)
        print(f"Found {near_neighbor_count} additional near-neighbor pairs")
        print(f"Total neighbor pairs: {touching_count} touching + {near_neighbor_count} near-neighbors = {len(neighbor_pairs)}")
    else:
        print("Returning only touching cells (max_distance_um = 0.0)")
        neighbor_pairs = touching_neighbor_data
    
    if neighbor_pairs:
        seen_pairs = set()
        deduped = []
        for n in neighbor_pairs:
            key = (int(n["cell_id_a"]), int(n["cell_id_b"]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            deduped.append(n)
        neighbor_pairs = deduped

        cursor = conn.cursor()
        neighbor_data = [
            (
                pair_id,
                n["cell_id_a"],
                n["cell_id_b"],
                n["cell_type_a"],
                n["cell_type_b"],
            )
            for pair_id, n in enumerate(neighbor_pairs, start=1)
        ]

        try:
            cursor.executemany(
                "INSERT INTO neighbors (pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b) VALUES (?, ?, ?, ?, ?)",
                neighbor_data,
            )
            conn.commit()
            print(f"Successfully inserted {len(neighbor_pairs)} neighbor pairs into database")
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                print(
                    "Warning: UNIQUE constraint on insert; inserting sequentially and skipping duplicate edges"
                )
                next_id = 1
                inserted_count = 0
                for n in neighbor_pairs:
                    try:
                        cursor.execute(
                            "INSERT INTO neighbors (pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b) VALUES (?, ?, ?, ?, ?)",
                            (
                                next_id,
                                n["cell_id_a"],
                                n["cell_id_b"],
                                n["cell_type_a"],
                                n["cell_type_b"],
                            ),
                        )
                        next_id += 1
                        inserted_count += 1
                    except sqlite3.IntegrityError:
                        continue
                conn.commit()
                print(f"Successfully inserted {inserted_count} unique neighbor pairs into database")
            else:
                raise e
    else:
        print("No neighbor pairs found within maximum distance threshold")
    
    if save_surfaces_pickle:
        print(f"Saving global surface to: {save_surfaces_pickle}")
        surface_data = {
            'global_surface': global_surface,
            'metadata': {
                'shape': global_surface.shape,
                'total_surface_voxels': global_surface.sum(),
                'description': 'Global surface mask for all cells'
            }
        }
        save_surfaces_to_pickle(surface_data, save_surfaces_pickle)
        
        if 'surfaces' in save_surfaces_pickle:
            halo_bboxes_pickle = save_surfaces_pickle.replace('surfaces', 'halo_bboxes')
        elif save_surfaces_pickle.endswith('.pkl'):
            halo_bboxes_pickle = save_surfaces_pickle.replace('.pkl', '_halo_bboxes.pkl')
        else:
            halo_bboxes_pickle = save_surfaces_pickle + '_halo_bboxes.pkl'
        print(f"Saving halo bounding boxes to: {halo_bboxes_pickle}")
        halo_bboxes_data = {
            'all_bboxes_with_halo': all_bboxes_with_halo,
            'metadata': {
                'total_cells': len(all_bboxes_with_halo),
                'max_distance_um': max_distance_um,
                'voxel_size_um': voxel_size_um,
                'description': 'Halo-extended bounding boxes for all cells'
            }
        }
        save_surfaces_to_pickle(halo_bboxes_data, halo_bboxes_pickle)
    
    if save_graph_state_pickle:
        print(f"Saving complete graph state to: {save_graph_state_pickle}")
        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
        neighbor_pairs = df_neighbors.to_dict('records')
        
        parameters = {
            'max_distance_um': max_distance_um,
            'voxel_size_um': voxel_size_um,
            'centroid_prefilter_radius_um': centroid_prefilter_radius_um,
            'n_jobs': n_jobs
        }
        
        save_graph_state_to_pickle(
            surfaces={'global_surface': global_surface},
            neighbor_pairs=neighbor_pairs,
            metadata_df=metadata_df,
            parameters=parameters,
            filepath=save_graph_state_pickle
        )
    
    return conn

def create_neighbor_edge_table_3d(
    ome_zarr_path: str,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    output_csv: str = None,
    n_jobs: int = 1
) -> pd.DataFrame:
    import zarr
    
    print(f"Loading segmentation mask from: {ome_zarr_path}")
    
    try:
        zarr_group = zarr.open(ome_zarr_path, mode='r')
        if 'labels' in zarr_group:
            mask_3d = zarr_group['labels'][0, 0]  # Common ome-zarr structure
        elif '0' in zarr_group and '0' in zarr_group['0']:
            mask_3d = zarr_group['0']['0'][0, 0]  # Alternative structure
        else:
            mask_3d = None
            for key in zarr_group.keys():
                if hasattr(zarr_group[key], 'shape') and len(zarr_group[key].shape) >= 3:
                    mask_3d = zarr_group[key]
                    break
            if mask_3d is None:
                raise ValueError("Could not find 3D segmentation mask in ome-zarr file")
    except Exception as e:
        raise ValueError(f"Error loading ome-zarr file: {e}")
    
    print(f"Mask shape: {mask_3d.shape}, dtype: {mask_3d.dtype}")
    
    if mask_3d.dtype.byteorder == '>': 
        print(f"Converting Big-endian data to native byte order for compatibility...")
        mask_3d = mask_3d.astype(mask_3d.dtype.newbyteorder('='))
        print(f"Converted to native byte order: {mask_3d.dtype}")
    
    required_cols = ['CellID', 'phenotype', 'Z_centroid', 'Y_centroid', 'X_centroid']
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    neighbor_df = find_all_neighbors_by_surface_distance_3d(
        mask_3d=mask_3d,
        metadata_df=metadata_df,
        max_distance_um=max_distance_um,
        voxel_size_um=voxel_size_um,
        centroid_prefilter_radius_um=centroid_prefilter_radius_um,
        n_jobs=n_jobs
    )
    
    if output_csv:
        neighbor_df.to_csv(output_csv, index=False)
        print(f"Neighbor edge table saved to: {output_csv}")
    
    return neighbor_df

def create_neighbor_edge_table_database_3d(
    ome_zarr_path: str,
    metadata_df: pd.DataFrame,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    db_path: str = 'cell_neighbor_pair_graph.db',
    cell_id: str = 'CellID',
    cell_type: str = 'phenotype',
    centroid_x: str = 'X_centroid',
    centroid_y: str = 'Y_centroid',
    centroid_z: str = 'Z_centroid',
    output_csv: str = None,
    output_anndata: str = None,
    n_jobs: int = 1,
    save_surfaces_pickle: str = None,
    load_surfaces_pickle: str = None,
    save_graph_state_pickle: str = None
) -> sqlite3.Connection:
    import zarr
    
    print(f"Loading segmentation mask from: {ome_zarr_path}")
    
    try:
        zarr_group = zarr.open(ome_zarr_path, mode='r')
        if 'labels' in zarr_group:
            mask_3d = zarr_group['labels'][0, 0]
        elif '0' in zarr_group and '0' in zarr_group['0']:
            mask_3d = zarr_group['0']['0'][0, 0]
        else:
            mask_3d = None
            for key in zarr_group.keys():
                if hasattr(zarr_group[key], 'shape') and len(zarr_group[key].shape) >= 3:
                    mask_3d = zarr_group[key]
                    break
            if mask_3d is None:
                raise ValueError("Could not find 3D segmentation mask in ome-zarr file")
    except Exception as e:
        raise ValueError(f"Error loading ome-zarr file: {e}")
    
    print(f"Mask shape: {mask_3d.shape}, dtype: {mask_3d.dtype}")
    
    if mask_3d.dtype.byteorder == '>': 
        print(f"Converting Big-endian data to native byte order for compatibility...")
        mask_3d = mask_3d.astype(mask_3d.dtype.newbyteorder('='))
        print(f"Converted to native byte order: {mask_3d.dtype}")
    
    required_cols = [cell_id, cell_type, centroid_x, centroid_y, centroid_z]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    conn = build_cell_graph_database_3d(
        mask_3d=mask_3d,
        metadata_df=metadata_df,
        max_distance_um=max_distance_um,
        voxel_size_um=voxel_size_um,
        centroid_prefilter_radius_um=centroid_prefilter_radius_um,
        db_path=db_path,
        cell_id=cell_id,
        cell_type=cell_type,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        centroid_z=centroid_z,
        n_jobs=n_jobs,
        save_surfaces_pickle=save_surfaces_pickle,
        load_surfaces_pickle=load_surfaces_pickle,
        save_graph_state_pickle=save_graph_state_pickle
    )
    
    # Export to CSV
    if output_csv:
        df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
        df_neighbors.to_csv(output_csv, index=False)
        print(f"Neighbor edge table saved to: {output_csv}")
    
    # Optional AnnData / DuckDB (failures are non-fatal; SQLite + CSV + pickles already written)
    if output_anndata:
        export_to_anndata(conn, output_anndata)

    duckdb_output = db_path.replace('.db', '.duckdb')
    export_to_duckdb(conn, duckdb_output)

    return conn

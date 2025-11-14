# Import

import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from scipy.ndimage import label, distance_transform_edt
import sqlite3
import os
import pickle
import math
import zarr

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Data loading

def load_neighbor_pairs_from_db(db_path: str) -> pd.DataFrame:
    """Load neighbor pairs from SQLite database (more efficient than CSV)."""
    print(f"Loading neighbor pairs from database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neighbors'")
    if not cursor.fetchone():
        conn.close()
        raise ValueError(f"Database {db_path} does not contain 'neighbors' table")
    
    query = """
    SELECT 
        cell_id_a as cell_a_id,
        cell_id_b as cell_b_id,
        cell_type_a as cell_a_type,
        cell_type_b as cell_b_type,
        surface_distance_um,
        euclidean_distance_um,
        pair_id
    FROM neighbors
    ORDER BY pair_id
    """
    
    neighbor_df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(neighbor_df)} neighbor pairs from database")
    return neighbor_df

def load_neighbor_pairs_from_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading neighbor pairs from: {csv_path}")
    neighbor_df = pd.read_csv(csv_path)
    
    if 'cell_a_id' in neighbor_df.columns and 'cell_b_id' in neighbor_df.columns:
        pass
    elif 'cell_id_a' in neighbor_df.columns and 'cell_id_b' in neighbor_df.columns:
        neighbor_df = neighbor_df.rename(columns={
            'cell_id_a': 'cell_a_id',
            'cell_id_b': 'cell_b_id'
        })
    else:
        raise ValueError(f"Missing required columns. Found: {list(neighbor_df.columns)}")
    
    if 'cell_a_type' in neighbor_df.columns and 'cell_b_type' in neighbor_df.columns:
        pass
    elif 'cell_type_a' in neighbor_df.columns and 'cell_type_b' in neighbor_df.columns:
        neighbor_df = neighbor_df.rename(columns={
            'cell_type_a': 'cell_a_type',
            'cell_type_b': 'cell_b_type'
        })
    else:
        raise ValueError(f"Missing cell type columns. Found: {list(neighbor_df.columns)}")
    
    print(f"Loaded {len(neighbor_df)} neighbor pairs")
    return neighbor_df

def load_halo_bboxes_from_pickle(pickle_path: str) -> Dict[int, Tuple[slice, slice, slice]]:
    print(f"Loading halo bounding boxes from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Halo bounding boxes pickle file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        bbox_data = pickle.load(f)
    
    if 'all_bboxes_with_halo' in bbox_data:
        halo_bboxes = bbox_data['all_bboxes_with_halo']
    elif isinstance(bbox_data, dict):
        halo_bboxes = bbox_data
    else:
        raise ValueError("Invalid format in halo bounding boxes pickle file")
    
    print(f"Loaded halo bounding boxes for {len(halo_bboxes)} cells")
    return halo_bboxes

def load_global_surface_from_pickle(pickle_path: str) -> np.ndarray:
    print(f"Loading global surface from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        graph_state_path = pickle_path.replace('surfaces.pkl', 'graph_state.pkl').replace('_surfaces.pkl', '_graph_state.pkl')
        if os.path.exists(graph_state_path):
            print(f"Global surface file not found. Trying to load from: {graph_state_path}")
            try:
                from ..core.find_cell_neighbors_3d import load_graph_state_from_pickle
                graph_state = load_graph_state_from_pickle(graph_state_path)
                if 'surfaces' in graph_state and 'global_surface' in graph_state['surfaces']:
                    global_surface = graph_state['surfaces']['global_surface']
                    print(f"Successfully loaded global surface")
                    print(f"Loaded global surface with shape {global_surface.shape} and {global_surface.sum()} surface voxels")
                    return global_surface
            except Exception as e:
                print(f"Could not load from graph_state file: {e}")
        
        raise FileNotFoundError(f"Global surface pickle file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        surface_data = pickle.load(f)
    
    if isinstance(surface_data, dict):
        if 'global_surface' in surface_data:
            global_surface = surface_data['global_surface']
        elif 'all_bboxes_with_halo' in surface_data:
            possible_graph_state_paths = [
                pickle_path.replace('surfaces.pkl', 'graph_state.pkl'),
                pickle_path.replace('_surfaces.pkl', '_graph_state.pkl'),
                pickle_path.replace('_graph_surfaces.pkl', '_graph_state.pkl'),
                os.path.join(os.path.dirname(pickle_path), os.path.basename(pickle_path).replace('surfaces.pkl', 'graph_state.pkl').replace('_surfaces.pkl', '_graph_state.pkl')),
            ]
            graph_state_path = None
            for path in possible_graph_state_paths:
                if os.path.exists(path):
                    graph_state_path = path
                    break
            
            if graph_state_path and os.path.exists(graph_state_path):
                print(f"Pickle file contains halo bounding boxes. Trying to load from graph_state file: {graph_state_path}")
                try:
                    from ..core.find_cell_neighbors_3d import load_graph_state_from_pickle
                    graph_state = load_graph_state_from_pickle(graph_state_path)
                    if 'surfaces' in graph_state and 'global_surface' in graph_state['surfaces']:
                        global_surface = graph_state['surfaces']['global_surface']
                        print(f"Successfully loaded global surface")
                        print(f"Loaded global surface with shape {global_surface.shape} and {global_surface.sum()} surface voxels")
                        return global_surface
                except Exception as e:
                    print(f"Could not load from graph_state file: {e}")
            
            raise ValueError(
                f"Pickle file contains halo bounding boxes, not global surface. "
                f"Expected file with 'global_surface' key, but found 'all_bboxes_with_halo'. "
            )
        else:
            for key, value in surface_data.items():
                if isinstance(value, np.ndarray):
                    print(f"Warning: Using '{key}' as global surface (expected 'global_surface')")
                    global_surface = value
                    break
            else:
                raise ValueError(
                    f"Invalid format in global surface pickle file. "
                    f"Expected dict with 'global_surface' key or numpy array. "
                    f"Found dict with keys: {list(surface_data.keys())}"
                )
    elif isinstance(surface_data, np.ndarray):
        global_surface = surface_data
    else:
        raise ValueError(
            f"Invalid format in global surface pickle file. "
            f"Expected dict or numpy array, got {type(surface_data)}"
        )
    
    if not isinstance(global_surface, np.ndarray):
        raise ValueError(f"Global surface is not a numpy array, got {type(global_surface)}")
    
    print(f"Loaded global surface with shape {global_surface.shape} and {global_surface.sum()} surface voxels")
    return global_surface

### Interscellar volume computation

def precompute_per_cell_edts(
    mask_3d: np.ndarray,
    cell_ids: List[int],
    neighbor_pairs_df: pd.DataFrame,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    voxel_size_um: tuple,
    max_distance_um: float,
    n_jobs: int = 1
) -> Dict[int, Dict[str, Any]]:
    from joblib import Parallel, delayed
    
    cells_with_neighbors = set(neighbor_pairs_df['cell_a_id'].unique()) | set(neighbor_pairs_df['cell_b_id'].unique())
    cells_to_process = [cid for cid in cell_ids if cid in cells_with_neighbors]
    skipped_count = len(cell_ids) - len(cells_to_process)
    
    print(f"EDT precomputation:")
    print(f"  - Total cells: {len(cell_ids)}")
    print(f"  - Cells with neighbors: {len(cells_to_process)}")
    print(f"  - Cells skipped (no neighbors): {skipped_count}")
    
    def compute_single_cell_edt_optimized(cell_id):
        if cell_id not in halo_bboxes:
            return cell_id, None
        
        halo_bbox = halo_bboxes[cell_id]
        
        mask_crop = mask_3d[halo_bbox]
        global_surface_crop = global_surface[halo_bbox]
        
        cell_mask = (mask_crop == cell_id)
        if not cell_mask.any():
            return cell_id, None
        
        surface = cell_mask & global_surface_crop
        if not surface.any():
            return cell_id, None
        
        from scipy.ndimage import binary_dilation, generate_binary_structure
        
        max_distance_voxels = int(np.ceil(max_distance_um / min(voxel_size_um)))
        struct_elem = generate_binary_structure(3, 1)
        
        dilated_region = binary_dilation(
            cell_mask, 
            structure=struct_elem, 
            iterations=max_distance_voxels
        )
        
        if not dilated_region.any():
            return cell_id, None
        
        edt_full = distance_transform_edt(~surface, sampling=voxel_size_um)
        
        edt_masked = np.where(dilated_region, edt_full, np.inf)
        
        return cell_id, {
            'edt': edt_masked,
            'surface': surface,
            'cell_mask': cell_mask,
            'dilated_region': dilated_region,
            'halo_bbox': halo_bbox
        }
    
    if n_jobs == 1:
        results = []
        for cell_id in tqdm(cells_to_process, desc="Computing optimized EDTs", unit="cells"):
            results.append(compute_single_cell_edt_optimized(cell_id))
    else:
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(compute_single_cell_edt_optimized)(cell_id) 
            for cell_id in tqdm(cells_to_process, desc="Computing optimized EDTs", unit="cells")
        )
    
    edt_cache = {}
    for cell_id, data in results:
        if data is not None:
            edt_cache[cell_id] = data
    
    print(f"Cached EDTs for {len(edt_cache)} cells (skipped {skipped_count} cells without neighbors)")
    return edt_cache

def compute_intercellular_volume_with_cached_edts(
    mask_a_crop: np.ndarray,
    mask_b_crop: np.ndarray,
    voxel_size_um: tuple,
    max_distance_um: float,
    edt_a_crop: np.ndarray = None,
    edt_b_crop: np.ndarray = None,
    dilated_a_crop: np.ndarray = None,
    dilated_b_crop: np.ndarray = None,
    global_surface_crop: np.ndarray = None
) -> tuple:
    from scipy.ndimage import binary_dilation, generate_binary_structure
    
    if edt_a_crop is None or edt_b_crop is None:
        return compute_intercellular_volume_constrained_edt(
            mask_a_crop, mask_b_crop, voxel_size_um, max_distance_um, global_surface_crop
        )
    
    if dilated_a_crop is not None and dilated_b_crop is not None:
        candidate_region = dilated_a_crop & dilated_b_crop 
        candidate_background = candidate_region & ~(mask_a_crop | mask_b_crop)
    else:
        max_distance_voxels = int(np.ceil(max_distance_um / min(voxel_size_um)))
        struct_elem = generate_binary_structure(3, 1)
        candidate_region_a = binary_dilation(mask_a_crop, structure=struct_elem, iterations=max_distance_voxels)
        candidate_region_b = binary_dilation(mask_b_crop, structure=struct_elem, iterations=max_distance_voxels)
        candidate_region = candidate_region_a & candidate_region_b 
        candidate_background = candidate_region & ~(mask_a_crop | mask_b_crop) 
    
    dist_to_surface_a_um = np.where(candidate_region, edt_a_crop, np.inf)
    dist_to_surface_b_um = np.where(candidate_region, edt_b_crop, np.inf)
    
    total_distance_um = dist_to_surface_a_um + dist_to_surface_b_um
    
    # gap = (~(A|B)) & (dA + dB <= max_distance_threshold)
    #   - (~(A|B)) = candidate_background (not in either cell)
    #   - dA = dist_to_surface_a_um
    #   - dB = dist_to_surface_b_um
    intercellular_mask = (
        candidate_background &
        (total_distance_um <= max_distance_um)
    )
    
    volume_voxels = intercellular_mask.sum() 
    voxel_volume_um3 = np.prod(voxel_size_um) 
    volume_um3 = volume_voxels * voxel_volume_um3
    
    if volume_voxels > 0:
        distances_um = total_distance_um[intercellular_mask]
        mean_distance_um = distances_um.mean()
        max_distance_um_actual = distances_um.max()
    else:
        mean_distance_um = 0.0
        max_distance_um_actual = 0.0
    
    return intercellular_mask, volume_voxels, volume_um3, mean_distance_um, max_distance_um_actual

def compute_intercellular_volume_constrained_edt(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, 
                                                voxel_size_um: tuple, max_distance_um: float = 3.0,
                                                global_surface_crop: np.ndarray = None) -> tuple:
    if global_surface_crop is not None:
        surface_a = mask_a_crop & global_surface_crop
        surface_b = mask_b_crop & global_surface_crop
    else:
        surface_a = mask_a_crop & ~binary_erosion(mask_a_crop)
        surface_b = mask_b_crop & ~binary_erosion(mask_b_crop)
    
    if not surface_a.any() or not surface_b.any():
        return np.zeros_like(mask_a_crop, dtype=bool), 0, 0.0, 0.0, 0.0
    
    max_distance_voxels = int(np.ceil(max_distance_um / min(voxel_size_um)))
    struct_elem = generate_binary_structure(3, 1)
    
    candidate_region_a = binary_dilation(mask_a_crop, structure=struct_elem, iterations=max_distance_voxels)
    candidate_region_b = binary_dilation(mask_b_crop, structure=struct_elem, iterations=max_distance_voxels)
    candidate_region = candidate_region_a & candidate_region_b
    candidate_background = candidate_region & ~(mask_a_crop | mask_b_crop)
    
    if not candidate_region.any():
        return np.zeros_like(mask_a_crop, dtype=bool), 0, 0.0, 0.0, 0.0
    
    coords = np.argwhere(candidate_region)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    
    padding = 2
    z_min = max(0, z_min - padding)
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    z_max = min(mask_a_crop.shape[0], z_max + padding + 1)
    y_max = min(mask_a_crop.shape[1], y_max + padding + 1)
    x_max = min(mask_a_crop.shape[2], x_max + padding + 1)
    
    bbox_slice = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    surface_a_tight = surface_a[bbox_slice]
    surface_b_tight = surface_b[bbox_slice]
    candidate_region_tight = candidate_region[bbox_slice]
    candidate_background_tight = candidate_background[bbox_slice]
    
    dist_to_surface_a_voxels = distance_transform_edt(~surface_a_tight)
    dist_to_surface_b_voxels = distance_transform_edt(~surface_b_tight)
    
    dist_to_surface_a_voxels = np.where(candidate_region_tight, dist_to_surface_a_voxels, np.inf)
    dist_to_surface_b_voxels = np.where(candidate_region_tight, dist_to_surface_b_voxels, np.inf)
    
    total_distance_voxels = dist_to_surface_a_voxels + dist_to_surface_b_voxels
    
    intercellular_mask_tight = (
        candidate_background_tight &
        (total_distance_voxels <= max_distance_voxels)
    )
    
    intercellular_mask = np.zeros_like(mask_a_crop, dtype=bool)
    intercellular_mask[bbox_slice] = intercellular_mask_tight
    
    volume_voxels = intercellular_mask.sum()
    voxel_volume_um3 = np.prod(voxel_size_um)
    volume_um3 = volume_voxels * voxel_volume_um3
    
    if volume_voxels > 0:
        distances_voxels = total_distance_voxels[intercellular_mask_tight]
        mean_distance_um = distances_voxels.mean() * min(voxel_size_um)
        max_distance_um_actual = distances_voxels.max() * min(voxel_size_um)
    else:
        mean_distance_um = 0.0
        max_distance_um_actual = 0.0
    
    return intercellular_mask, volume_voxels, volume_um3, mean_distance_um, max_distance_um_actual

def compute_intercellular_volume(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, 
                                voxel_size_um: tuple, max_distance_um: float = 3.0,
                                global_surface_crop: np.ndarray = None) -> tuple:
    from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation, generate_binary_structure
    
    if global_surface_crop is not None:
        surface_a = mask_a_crop & global_surface_crop
        surface_b = mask_b_crop & global_surface_crop
    else:
        surface_a = mask_a_crop & ~binary_erosion(mask_a_crop)
        surface_b = mask_b_crop & ~binary_erosion(mask_b_crop)
    
    if not surface_a.any() or not surface_b.any():
        return np.zeros_like(mask_a_crop, dtype=bool), 0, 0.0, 0.0, 0.0
    
    max_distance_voxels = int(np.ceil(max_distance_um / min(voxel_size_um)))
    
    struct_elem = generate_binary_structure(3, 1)
    
    candidate_region_a = binary_dilation(mask_a_crop, structure=struct_elem, iterations=max_distance_voxels)
    candidate_region_b = binary_dilation(mask_b_crop, structure=struct_elem, iterations=max_distance_voxels)
    
    candidate_region = candidate_region_a & candidate_region_b
    candidate_background = candidate_region & ~(mask_a_crop | mask_b_crop)
    
    surface_a_masked = surface_a.copy()
    surface_b_masked = surface_b.copy()
    
    dist_to_surface_a_voxels = distance_transform_edt(~surface_a)
    dist_to_surface_b_voxels = distance_transform_edt(~surface_b)
    
    dist_to_surface_a_voxels = np.where(candidate_region, dist_to_surface_a_voxels, np.inf)
    dist_to_surface_b_voxels = np.where(candidate_region, dist_to_surface_b_voxels, np.inf)
    
    total_communication_distance_voxels = dist_to_surface_a_voxels + dist_to_surface_b_voxels
    
    intercellular_mask = (
        candidate_background & 
        (total_communication_distance_voxels <= max_distance_voxels) 
    )
    
    volume_voxels = intercellular_mask.sum()
    voxel_volume_um3 = np.prod(voxel_size_um)
    volume_um3 = volume_voxels * voxel_volume_um3
    
    if volume_voxels > 0:
        distances_voxels = total_communication_distance_voxels[intercellular_mask]
        mean_distance_um = distances_voxels.mean() * min(voxel_size_um)
        max_distance_um_actual = distances_voxels.max() * min(voxel_size_um)
    else:
        mean_distance_um = 0.0
        max_distance_um_actual = 0.0
    
    return intercellular_mask, volume_voxels, volume_um3, mean_distance_um, max_distance_um_actual

def compute_touching_surface_volume(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, voxel_size_um: tuple) -> tuple:
    from scipy.ndimage import binary_dilation
    
    touching_a = mask_a_crop & binary_dilation(mask_b_crop, iterations=1)
    touching_b = mask_b_crop & binary_dilation(mask_a_crop, iterations=1)
    touching_surface = touching_a | touching_b
    
    volume_voxels = touching_surface.sum()
    
    voxel_volume_um3 = np.prod(voxel_size_um)
    volume_um3 = volume_voxels * voxel_volume_um3
    
    return touching_surface, volume_voxels, volume_um3

def compute_intracellular_volume(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, 
                                voxel_size_um: tuple, intracellular_threshold_um: float = 1.0,
                                intercellular_mask: np.ndarray = None) -> tuple:
    from scipy.ndimage import distance_transform_edt
    
    if intercellular_mask is None or not np.any(intercellular_mask):
        empty = np.zeros_like(mask_a_crop, dtype=bool)
        return empty, 0, 0.0
    
    dist_to_gap_um = distance_transform_edt(~intercellular_mask, sampling=voxel_size_um)
    
    intra_a = (mask_a_crop) & (dist_to_gap_um <= intracellular_threshold_um)
    intra_b = (mask_b_crop) & (dist_to_gap_um <= intracellular_threshold_um)
    
    intracellular_combined = intra_a | intra_b
    
    volume_voxels = int(intracellular_combined.sum())
    voxel_volume_um3 = np.prod(voxel_size_um)
    volume_um3 = volume_voxels * voxel_volume_um3
    
    return intracellular_combined, volume_voxels, volume_um3

def compute_interscellar_volume(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, 
                                             voxel_size_um: tuple, max_distance_um: float = 3.0,
                                             intracellular_threshold_um: float = 1.0,
                                             full_mask_crop: np.ndarray = None,
                                             global_surface_crop: np.ndarray = None,
                                             edt_a_crop: np.ndarray = None,
                                             edt_b_crop: np.ndarray = None,
                                             dilated_a_crop: np.ndarray = None,
                                             dilated_b_crop: np.ndarray = None) -> dict:
    # 1. Intercellular volume
    if edt_a_crop is not None and edt_b_crop is not None:
        intercellular_mask, intercellular_voxels, intercellular_um3, mean_dist_um, max_dist_um = compute_intercellular_volume_with_cached_edts(
            mask_a_crop, mask_b_crop, voxel_size_um, max_distance_um, 
            edt_a_crop, edt_b_crop, dilated_a_crop, dilated_b_crop, global_surface_crop
        )
    else:
        intercellular_mask, intercellular_voxels, intercellular_um3, mean_dist_um, max_dist_um = compute_intercellular_volume_constrained_edt(
            mask_a_crop, mask_b_crop, voxel_size_um, max_distance_um, global_surface_crop
        )
    
    # 2. Touching surface volume
    touching_surface, touching_voxels, touching_um3 = compute_touching_surface_volume(
        mask_a_crop, mask_b_crop, voxel_size_um
    )
    
    # 3. Intracellular volume
    intracellular_combined, intracellular_voxels, intracellular_um3 = compute_intracellular_volume(
        mask_a_crop, mask_b_crop, voxel_size_um, intracellular_threshold_um, intercellular_mask
    )
    
    # 4. Exclude third-party cells
    if full_mask_crop is not None:
        other_cells_mask = (full_mask_crop != 0) & ~mask_a_crop & ~mask_b_crop
        
        intercellular_mask = intercellular_mask & ~other_cells_mask
        intracellular_combined = intracellular_combined & ~other_cells_mask
        touching_surface = touching_surface & ~other_cells_mask
        
        intercellular_voxels = intercellular_mask.sum()
        intercellular_um3 = intercellular_voxels * np.prod(voxel_size_um)
        
        intracellular_voxels = intracellular_combined.sum()
        intracellular_um3 = intracellular_voxels * np.prod(voxel_size_um)
        
        touching_voxels = touching_surface.sum()
        touching_um3 = touching_voxels * np.prod(voxel_size_um)
    
    interscellar_voxels = intercellular_voxels + intracellular_voxels + touching_voxels
    interscellar_volume_um3 = intercellular_um3 + intracellular_um3 + touching_um3
    interscellar_mask = intercellular_mask | intracellular_combined | touching_surface
    
    labeled_intercellular, num_components = label(intercellular_mask)
    largest_component_size = 0
    if num_components > 0:
        component_sizes = [np.sum(labeled_intercellular == i) for i in range(1, num_components + 1)]
        largest_component_size = max(component_sizes) * np.prod(voxel_size_um)
    
    return {
        # Intercellular volume
        'intercellular_mask': intercellular_mask,
        'edt_volume_voxels': intercellular_voxels,
        'edt_volume_um3': intercellular_um3, 
        'intercellular_voxels': intercellular_voxels, 
        'intercellular_um3': intercellular_um3, 
        'mean_distance_um': mean_dist_um,
        'max_distance_um': max_dist_um,
        
        # Touching surface
        'touching_surface_mask': touching_surface,
        'touching_surface_area_voxels': touching_voxels,
        'touching_surface_area_um2': touching_um3, 
        'touching_surface_voxels': touching_voxels, 
        'touching_surface_um3': touching_um3,  
        
        # Intracellular volume 
        'intracellular_mask': intracellular_combined,
        'intracellular_volume_voxels': intracellular_voxels, 
        'intracellular_volume_um3': intracellular_um3, 
        'intracellular_voxels': intracellular_voxels, 
        'intracellular_um3': intracellular_um3,
        
        # Interscellar volume (combined total)
        'interscellar_mask': interscellar_mask,
        'interscellar_voxels': interscellar_voxels,
        'interscellar_volume_um3': interscellar_volume_um3,
        'total_interscellar_volume_um3': interscellar_volume_um3,
        'total_interscellar_volume_voxels': interscellar_voxels,
        
        # Connectivity statistics
        'num_components': num_components,
        'largest_component_volume_um3': largest_component_size,
        
        # Parameters used
        'max_distance_threshold_um': max_distance_um,
        'intracellular_threshold_um': intracellular_threshold_um,
        'voxel_volume_um3': np.prod(voxel_size_um)
    }

def compute_interscellar_volumes_for_all_pairs(
    mask_3d: np.ndarray,
    neighbor_pairs_df: pd.DataFrame,
    voxel_size_um: tuple,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    max_distance_um: float = 3.0,
    intracellular_threshold_um: float = 1.0,
    n_jobs: int = 4,
    intermediate_results_dir: str = "intermediate_interscellar_results",
    output_mesh_zarr: str = None 
) -> List[Dict[str, Any]]:
    print(f"Computing interscellar volumes for {len(neighbor_pairs_df)} neighbor pairs...")
    print(f"Using {n_jobs} parallel jobs")
    
    import os
    import zarr
    existing_results = _load_intermediate_results(intermediate_results_dir)
    zarr_exists = output_mesh_zarr and os.path.exists(output_mesh_zarr) and os.path.isdir(output_mesh_zarr)
    
    if existing_results:
        print(f"Found {len(existing_results)} existing intermediate results")
        if zarr_exists:
            print(f"Found existing mesh zarr: {output_mesh_zarr}")
            zarr_group = zarr.open(output_mesh_zarr, mode='r')
            existing_pairs = zarr_group.attrs.get('num_pairs', 0)
            print(f"Existing zarr contains {existing_pairs} pairs")
        
        user_input = input("Resume from intermediate results? (y/n): ").lower().strip()
        if user_input == 'y':
            print("Resuming from intermediate results...")
            computed_pairs = set()
            for result in existing_results:
                cell_a = result.get('cell_a_id')
                cell_b = result.get('cell_b_id')
                if cell_a and cell_b:
                    pair_key = tuple(sorted([cell_a, cell_b]))
                    computed_pairs.add(pair_key)
            
            print(f"Found {len(computed_pairs)} already-computed pairs")
            
            def pair_in_computed(row):
                pair_key = tuple(sorted([row['cell_a_id'], row['cell_b_id']]))
                return pair_key in computed_pairs
            
            remaining_pairs = neighbor_pairs_df[~neighbor_pairs_df.apply(pair_in_computed, axis=1)].copy()
            
            if len(remaining_pairs) == 0:
                print(f"All pairs already computed! Returning existing results.")
                if zarr_exists:
                    print(f"Mesh zarr already contains all pairs")
                return existing_results
            else:
                print(f"Continuing with {len(remaining_pairs)} remaining pairs to compute")
                neighbor_pairs_df = remaining_pairs
                if zarr_exists:
                    print(f"Mesh zarr will be updated incrementally with remaining pairs")
        else:
            print("Starting fresh computation...")
            _cleanup_intermediate_results(intermediate_results_dir)
            if zarr_exists:
                import shutil
                shutil.rmtree(output_mesh_zarr)
                print(f"Deleted existing mesh zarr to restart")
            existing_results = []
    
    if output_mesh_zarr:
        if not zarr_exists:
            _write_chunk_to_mesh_zarr([], mask_3d, output_mesh_zarr, voxel_size_um, initialize=True)
            print(f"Initialized incremental mesh zarr: {output_mesh_zarr}")
    
    cell_type_groups = neighbor_pairs_df.groupby(['cell_a_type', 'cell_b_type'])
    print(f"Found {len(cell_type_groups)} unique cell type combinations")
    
    all_results = []
    
    for (cell_type_a, cell_type_b), group_df in cell_type_groups:
        print(f"Processing {len(group_df)} pairs of {cell_type_a}-{cell_type_b}")
        
        group_results = _process_cell_type_group(
            mask_3d, group_df, voxel_size_um, global_surface, halo_bboxes,
            max_distance_um, intracellular_threshold_um, n_jobs, intermediate_results_dir,
            output_mesh_zarr=output_mesh_zarr
        )
        
        all_results.extend(group_results)
    
    if existing_results and len(all_results) > 0:
        print(f"Combining existing results ({len(existing_results)} pairs) with new results ({len(all_results)} pairs)")
        all_results = existing_results + all_results
        print(f"Total results: {len(all_results)} pairs")
    elif existing_results and len(all_results) == 0:
        all_results = existing_results
    
    print(f"Computed interscellar volumes for {len(all_results)} total cell neighbor pairs")
    
    # Note: Cleanup is deferred until after all steps complete (handled in wrapper)
    # This allows resuming if the job fails later
    
    return all_results

def _process_cell_type_group(
    mask_3d: np.ndarray,
    group_df: pd.DataFrame,
    voxel_size_um: tuple,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    max_distance_um: float,
    intracellular_threshold_um: float,
    n_jobs: int,
    intermediate_results_dir: str = "intermediate_interscellar_results",
    output_mesh_zarr: str = None 
) -> List[Dict[str, Any]]:
    from functools import partial
    
    unique_cell_ids = set()
    unique_cell_ids.update(group_df['cell_a_id'].unique())
    unique_cell_ids.update(group_df['cell_b_id'].unique())
    unique_cell_ids = list(unique_cell_ids)
    
    print(f"Precomputing EDTs for {len(unique_cell_ids)} unique cells in this group...")
    
    edt_cache = precompute_per_cell_edts(
        mask_3d, unique_cell_ids, group_df, global_surface, halo_bboxes, 
        voxel_size_um, max_distance_um, n_jobs
    )
    
    process_pair = partial(
        _process_single_pair,
        mask_3d=mask_3d,
        voxel_size_um=voxel_size_um,
        global_surface=global_surface,
        halo_bboxes=halo_bboxes,
        max_distance_um=max_distance_um,
        intracellular_threshold_um=intracellular_threshold_um,
        edt_cache=edt_cache
    )
    
    if 'pair_id' in group_df.columns:
        pair_data = [(row['cell_a_id'], row['cell_b_id'], row['cell_a_type'], row['cell_b_type'], row['pair_id']) 
                     for idx, row in group_df.iterrows()]
    else:
        pair_data = [(row['cell_a_id'], row['cell_b_id'], row['cell_a_type'], row['cell_b_type'], idx) 
                     for idx, row in group_df.iterrows()]
    
    chunk_size = 25 # Memory-efficient chunking
    total_chunks = (len(pair_data) - 1) // chunk_size + 1
    
    print(f"Processing {len(pair_data)} pairs in {total_chunks} chunks of size {chunk_size}...")
    
    if n_jobs == 1:
        # Sequential processing
        results = []
        for i in range(0, len(pair_data), chunk_size):
            chunk = pair_data[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} pairs)...")
            chunk_results = []
            
            for pair_info in tqdm(chunk, desc=f"Chunk {chunk_num}/{total_chunks}"):
                result = process_pair(pair_info)
                if result is not None:
                    chunk_results.append(result)
            
            results.extend(chunk_results)
            print(f"Chunk {chunk_num} completed: {len(chunk_results)} valid results")
            
            if chunk_results:
                _save_intermediate_results(chunk_results, chunk_num, total_chunks, intermediate_results_dir)
                
                if output_mesh_zarr:
                    pairs_written = _write_chunk_to_mesh_zarr(
                        chunk_results, mask_3d, output_mesh_zarr, voxel_size_um, initialize=False
                    )
                    print(f"Wrote {pairs_written} pairs to mesh zarr")
                
                import gc
                del chunk_results
                gc.collect()
                gc.collect()
    else:
        # Parallel processing
        from joblib import Parallel, delayed
        
        results = []
        for i in range(0, len(pair_data), chunk_size):
            chunk = pair_data[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} pairs)...")
            
            chunk_results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(process_pair)(pair_info)
                for pair_info in tqdm(chunk, desc=f"Chunk {chunk_num}/{total_chunks}")
            )
            
            chunk_results = [r for r in chunk_results if r is not None]
            
            results.extend(chunk_results)
            print(f"Chunk {chunk_num} completed: {len(chunk_results)} valid results")
            
            if chunk_results:
                _save_intermediate_results(chunk_results, chunk_num, total_chunks, intermediate_results_dir)
                
                if output_mesh_zarr:
                    pairs_written = _write_chunk_to_mesh_zarr(
                        chunk_results, mask_3d, output_mesh_zarr, voxel_size_um, initialize=False
                    )
                    print(f"Wrote {pairs_written} pairs to mesh zarr")
                
                import gc
                del chunk_results
                gc.collect()
                gc.collect()
    
    return results

def _save_intermediate_results(chunk_results: List[Dict[str, Any]], chunk_num: int, total_chunks: int, intermediate_dir: str = "intermediate_interscellar_results") -> None:
    import pickle
    import os
    from datetime import datetime
    
    os.makedirs(intermediate_dir, exist_ok=True)
    
    chunk_file = os.path.join(intermediate_dir, f"chunk_{chunk_num:03d}_of_{total_chunks:03d}.pkl")

    mask_keys_to_strip = [
        'interscellar_mask', 'intercellular_mask', 'intracellular_mask',
        'touching_surface_mask', 'interscellar_mesh_mask',
        'cell_a_mask', 'cell_b_mask', 'mask_a_crop', 'mask_b_crop'
    ]
    
    def slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
        if result is None:
            return None
        slimmed = result.copy()
        for key in mask_keys_to_strip:
            slimmed.pop(key, None) 
        return slimmed
    
    slimmed_results = [slim_result(r) for r in chunk_results]
    
    chunk_data = {
        'chunk_num': chunk_num,
        'total_chunks': total_chunks,
        'results': slimmed_results,
        'timestamp': datetime.now().isoformat(),
        'num_results': len(slimmed_results)
    }
    
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunk_data, f)
    
    print(f"Saved intermediate results: {chunk_file} ({len(slimmed_results)} results)")
    
    summary_file = os.path.join(intermediate_dir, "progress_summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"Chunk {chunk_num}/{total_chunks}: {len(chunk_results)} results at {datetime.now().isoformat()}\n")

def _load_intermediate_results(intermediate_dir: str = "intermediate_interscellar_results") -> List[Dict[str, Any]]:
    import pickle
    import os
    import glob
    
    if not os.path.exists(intermediate_dir):
        return []
    
    chunk_files = sorted(glob.glob(os.path.join(intermediate_dir, "chunk_*.pkl")))
    
    if not chunk_files:
        return []
    
    print(f"Found {len(chunk_files)} intermediate result files")
    
    all_results = []
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            all_results.extend(chunk_data['results'])
            print(f"Loaded {chunk_data['num_results']} results from {os.path.basename(chunk_file)}")
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
    
    print(f"Total intermediate results loaded: {len(all_results)}")
    return all_results

def _cleanup_intermediate_results(intermediate_dir: str = "intermediate_interscellar_results") -> None:
    import shutil
    import os
    
    if os.path.exists(intermediate_dir):
        shutil.rmtree(intermediate_dir)
        print("Cleaned up intermediate results")

def _slice_array_to_union_bbox(
    array: np.ndarray,
    cell_bbox: Tuple[slice, slice, slice],
    union_bbox: Tuple[slice, slice, slice]
) -> np.ndarray:
    edt = array
    
    z_overlap_start = max(cell_bbox[0].start, union_bbox[0].start)
    z_overlap_stop = min(cell_bbox[0].stop, union_bbox[0].stop)
    y_overlap_start = max(cell_bbox[1].start, union_bbox[1].start)
    y_overlap_stop = min(cell_bbox[1].stop, union_bbox[1].stop)
    x_overlap_start = max(cell_bbox[2].start, union_bbox[2].start)
    x_overlap_stop = min(cell_bbox[2].stop, union_bbox[2].stop)
    
    if (z_overlap_stop <= z_overlap_start or 
        y_overlap_stop <= y_overlap_start or 
        x_overlap_stop <= x_overlap_start):
        return None
    
    z_cell_start = z_overlap_start - cell_bbox[0].start
    z_cell_stop = z_overlap_stop - cell_bbox[0].start
    y_cell_start = y_overlap_start - cell_bbox[1].start
    y_cell_stop = y_overlap_stop - cell_bbox[1].start
    x_cell_start = x_overlap_start - cell_bbox[2].start
    x_cell_stop = x_overlap_stop - cell_bbox[2].start
    
    edt_slice = edt[z_cell_start:z_cell_stop, y_cell_start:y_cell_stop, x_cell_start:x_cell_stop]
    
    union_shape = (union_bbox[0].stop - union_bbox[0].start,
                   union_bbox[1].stop - union_bbox[1].start,
                   union_bbox[2].stop - union_bbox[2].start)
    
    edt_in_union = np.full(union_shape, np.inf, dtype=edt.dtype)
    
    z_union_start = z_overlap_start - union_bbox[0].start
    z_union_stop = z_overlap_stop - union_bbox[0].start
    y_union_start = y_overlap_start - union_bbox[1].start
    y_union_stop = y_overlap_stop - union_bbox[1].start
    x_union_start = x_overlap_start - union_bbox[2].start
    x_union_stop = x_overlap_stop - union_bbox[2].start
    
    edt_in_union[z_union_start:z_union_stop, 
                 y_union_start:y_union_stop, 
                 x_union_start:x_union_stop] = edt_slice
    
    return edt_in_union

def _process_single_pair(
    pair_info: tuple,
    mask_3d: np.ndarray,
    voxel_size_um: tuple,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    max_distance_um: float,
    intracellular_threshold_um: float,
    edt_cache: Dict[int, Dict[str, Any]] = None
) -> Dict[str, Any]:
    cell_a_id, cell_b_id, cell_type_a, cell_type_b, pair_id = pair_info
    
    try:
        if cell_a_id not in halo_bboxes or cell_b_id not in halo_bboxes:
            return None
        
        bbox_a = halo_bboxes[cell_a_id]
        bbox_b = halo_bboxes[cell_b_id]
        
        z_start = min(bbox_a[0].start, bbox_b[0].start)
        z_stop = max(bbox_a[0].stop, bbox_b[0].stop)
        y_start = min(bbox_a[1].start, bbox_b[1].start)
        y_stop = max(bbox_a[1].stop, bbox_b[1].stop)
        x_start = min(bbox_a[2].start, bbox_b[2].start)
        x_stop = max(bbox_a[2].stop, bbox_b[2].stop)
        
        union_bbox = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
        
        mask_crop = mask_3d[union_bbox]
        
        mask_a_crop = (mask_crop == cell_a_id)
        mask_b_crop = (mask_crop == cell_b_id)
        
        if not mask_a_crop.any() or not mask_b_crop.any():
            return None
        
        global_surface_crop = global_surface[union_bbox] if global_surface is not None else None
        
        edt_a_crop = None
        edt_b_crop = None
        dilated_a_crop = None
        dilated_b_crop = None
        
        if edt_cache is not None:
            if cell_a_id in edt_cache:
                edt_a_crop = _slice_array_to_union_bbox(
                    edt_cache[cell_a_id]['edt'],
                    edt_cache[cell_a_id]['halo_bbox'],
                    union_bbox
                )
                if 'dilated_region' in edt_cache[cell_a_id]:
                    dilated_a_crop = _slice_array_to_union_bbox(
                        edt_cache[cell_a_id]['dilated_region'],
                        edt_cache[cell_a_id]['halo_bbox'],
                        union_bbox
                    )
            
            if cell_b_id in edt_cache:
                edt_b_crop = _slice_array_to_union_bbox(
                    edt_cache[cell_b_id]['edt'],
                    edt_cache[cell_b_id]['halo_bbox'],
                    union_bbox
                )
                if 'dilated_region' in edt_cache[cell_b_id]:
                    dilated_b_crop = _slice_array_to_union_bbox(
                        edt_cache[cell_b_id]['dilated_region'],
                        edt_cache[cell_b_id]['halo_bbox'],
                        union_bbox
                    )
        
        interscellar_results = compute_interscellar_volume(
            mask_a_crop, mask_b_crop, voxel_size_um, max_distance_um, 
            intracellular_threshold_um, mask_crop, global_surface_crop,
            edt_a_crop, edt_b_crop, dilated_a_crop, dilated_b_crop
        )
        
        result = {
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'cell_a_type': cell_type_a,
            'cell_b_type': cell_type_b,
            'pair_id': pair_id,
            'union_bbox': union_bbox,
            **interscellar_results
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing pair ({cell_a_id}, {cell_b_id}): {e}")
        return None

def compute_interscellar_volume_for_pair(
    mask_3d: np.ndarray, 
    cell_a_id: int, 
    cell_b_id: int, 
    voxel_size_um: tuple,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    max_distance_um: float = 3.0,
    intracellular_threshold_um: float = 1.0,
    pair_id: int = None
) -> Dict[str, Any]:
    if cell_a_id not in halo_bboxes or cell_b_id not in halo_bboxes:
        return None
    
    bbox_a = halo_bboxes[cell_a_id]
    bbox_b = halo_bboxes[cell_b_id]
    
    z_start = min(bbox_a[0].start, bbox_b[0].start)
    z_stop = max(bbox_a[0].stop, bbox_b[0].stop)
    y_start = min(bbox_a[1].start, bbox_b[1].start)
    y_stop = max(bbox_a[1].stop, bbox_b[1].stop)
    x_start = min(bbox_a[2].start, bbox_b[2].start)
    x_stop = max(bbox_a[2].stop, bbox_b[2].stop)
    
    union_bbox = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    
    mask_crop = mask_3d[union_bbox]
    surface_crop = global_surface[union_bbox]
    
    mask_a_crop = (mask_crop == cell_a_id)
    mask_b_crop = (mask_crop == cell_b_id)
    
    if not mask_a_crop.any() or not mask_b_crop.any():
        return None
    
    voxel_volume_um3 = np.prod(voxel_size_um)
    
    touching_surface, touching_voxels, touching_um3 = compute_touching_surface_volume(
        mask_a_crop, mask_b_crop, voxel_size_um
    )
    surface_area_um2 = touching_um3  # Surface area in um^2 is same as volume in um^3 for voxel-based calculation
    
    combined_cells = mask_a_crop | mask_b_crop
    
    surface_a_crop = (mask_crop == cell_a_id) & surface_crop
    surface_b_crop = (mask_crop == cell_b_id) & surface_crop
    
    if not surface_a_crop.any() or not surface_b_crop.any():
        return None
    
    from scipy.ndimage import binary_dilation, generate_binary_structure
    
    max_distance_voxels = int(np.ceil(max_distance_um / min(voxel_size_um)))
    struct_elem = generate_binary_structure(3, 1)  # 3D connectivity
    
    candidate_region_a = binary_dilation(mask_a_crop, structure=struct_elem, iterations=max_distance_voxels)
    candidate_region_b = binary_dilation(mask_b_crop, structure=struct_elem, iterations=max_distance_voxels)
    
    candidate_region = candidate_region_a & candidate_region_b
    candidate_background = candidate_region & ~combined_cells
    
    dist_to_surface_a = distance_transform_edt(~surface_a_crop, sampling=voxel_size_um)
    dist_to_surface_b = distance_transform_edt(~surface_b_crop, sampling=voxel_size_um)
    
    dist_to_surface_a = np.where(candidate_region, dist_to_surface_a, np.inf)
    dist_to_surface_b = np.where(candidate_region, dist_to_surface_b, np.inf)
    
    total_communication_distance = dist_to_surface_a + dist_to_surface_b
    
    intercellular_mask = (
        candidate_background &  # Only candidate background voxels
        (total_communication_distance <= max_distance_um)  # Within total communication distance (in μm)
    )
    
    intracellular_combined, intracellular_voxels, intracellular_volume_um3 = compute_intracellular_volume(
        mask_a_crop, mask_b_crop, voxel_size_um, intracellular_threshold_um, intercellular_mask
    )

    other_cells_mask = (mask_crop != 0) & ~mask_a_crop & ~mask_b_crop
    
    intercellular_mask = intercellular_mask & ~other_cells_mask
    intracellular_combined = intracellular_combined & ~other_cells_mask
    touching_surface = touching_surface & ~other_cells_mask
    
    edt_volume_voxels = intercellular_mask.sum()
    edt_volume_um3 = edt_volume_voxels * voxel_volume_um3
    
    intracellular_volume_voxels = intracellular_combined.sum()
    intracellular_volume_um3 = intracellular_volume_voxels * voxel_volume_um3
    
    total_interscellar_voxels = edt_volume_voxels + intracellular_volume_voxels
    total_interscellar_volume_um3 = total_interscellar_voxels * voxel_volume_um3
    

    interscellar_combined = intercellular_mask.copy()
    
    if intracellular_combined.any():
        interscellar_combined = interscellar_combined | intracellular_combined
    
    if pair_id is not None and interscellar_combined.any():
        interscellar_mesh_mask = interscellar_combined.astype(np.uint16) * pair_id
    else:
        interscellar_mesh_mask = interscellar_combined.astype(np.uint16)
    
    if total_interscellar_voxels > 0:
        if intercellular_mask.any():
            distances = total_communication_distance[intercellular_mask]
            mean_distance_um = distances.mean() * min(voxel_size_um)
            max_distance_um = distances.max() * min(voxel_size_um)
        else:
            mean_distance_um = 0.0
            max_distance_um = 0.0
        
        labeled_intercellular, num_components = label(intercellular_mask)
        largest_component_size = 0
        if num_components > 0:
            component_sizes = [np.sum(labeled_intercellular == i) for i in range(1, num_components + 1)]
            largest_component_size = max(component_sizes) * voxel_volume_um3
        
        return {
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'pair_id': pair_id,
            # Interscellar volume
            'total_interscellar_volume_um3': total_interscellar_volume_um3,
            'total_interscellar_volume_voxels': total_interscellar_voxels,
            # Volume components
            'edt_volume_um3': edt_volume_um3,
            'edt_volume_voxels': edt_volume_voxels,
            'intracellular_volume_um3': intracellular_volume_um3,
            'intracellular_volume_voxels': intracellular_volume_voxels,
            'touching_surface_area_um2': surface_area_um2,
            'touching_surface_area_voxels': touching_voxels,
            # Distance statistics
            'mean_distance_um': mean_distance_um,
            'max_distance_um': max_distance_um,
            'num_components': num_components,
            'largest_component_volume_um3': largest_component_size,
            'voxel_volume_um3': voxel_volume_um3,
            # Parameters used
            'max_distance_threshold_um': max_distance_um,
            'intracellular_threshold_um': intracellular_threshold_um,
            # 3D mesh data
            'interscellar_mesh_mask': interscellar_mesh_mask,
            'union_bbox': union_bbox
        }
    else:
        return {
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'pair_id': pair_id,
            'total_interscellar_volume_um3': 0.0,
            'total_interscellar_volume_voxels': 0,
            'edt_volume_um3': 0.0,
            'edt_volume_voxels': 0,
            'intracellular_volume_um3': 0.0,
            'intracellular_volume_voxels': 0,
            'touching_surface_area_um2': surface_area_um2,
            'touching_surface_area_voxels': touching_voxels,
            'mean_distance_um': 0.0,
            'max_distance_um': 0.0,
            'num_components': 0,
            'largest_component_volume_um3': 0.0,
            'voxel_volume_um3': voxel_volume_um3,
            'max_distance_threshold_um': max_distance_um,
            'intracellular_threshold_um': intracellular_threshold_um,
            'interscellar_mesh_mask': np.zeros_like(interscellar_combined, dtype=np.uint16),
            'union_bbox': union_bbox
        }

def compute_interscellar_volumes_for_neighbor_pairs(
    mask_3d: np.ndarray,
    neighbor_pairs_df: pd.DataFrame,
    voxel_size_um: tuple,
    global_surface: np.ndarray,
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    max_distance_um: float = 3.0,
    intracellular_threshold_um: float = 1.0,
    n_jobs: int = 4,
    intermediate_results_dir: str = "intermediate_interscellar_results",
    output_mesh_zarr: str = None 
) -> List[Dict[str, Any]]:
    print(f"Computing interscellar volumes for {len(neighbor_pairs_df)} neighbor pairs...")
    
    results = compute_interscellar_volumes_for_all_pairs(
        mask_3d=mask_3d,
        neighbor_pairs_df=neighbor_pairs_df,
        voxel_size_um=voxel_size_um,
        global_surface=global_surface,
        halo_bboxes=halo_bboxes,
        max_distance_um=max_distance_um,
        intracellular_threshold_um=intracellular_threshold_um,
        n_jobs=n_jobs,
        intermediate_results_dir=intermediate_results_dir,
        output_mesh_zarr=output_mesh_zarr  # Pass for incremental writing
    )
    
    return results

def _write_chunk_to_mesh_zarr(
    chunk_results: List[Dict[str, Any]],
    mask_3d: np.ndarray,
    zarr_path: str,
    voxel_size_um: tuple,
    initialize: bool = False
) -> int:
    import zarr
    
    pairs_written = 0
    
    if initialize:
        zarr_group = zarr.open(zarr_path, mode='w')

        zarr_dataset = zarr_group.create_dataset(
            'interscellar_meshes',
            shape=mask_3d.shape,
            dtype=np.uint16,
            chunks=(64, 64, 64),
            compression='gzip',
            compression_opts=6,
            fill_value=0
        )

        zarr_group.attrs['description'] = 'Global interscellar volume meshes with unique pair IDs'
        zarr_group.attrs['voxel_size_um'] = voxel_size_um
        zarr_group.attrs['shape'] = mask_3d.shape
        zarr_group.attrs['dtype'] = str(np.uint16)
        zarr_group.attrs['coordinate_system'] = 'same_as_input_segmentation'
        zarr_group.attrs['alignment_reference'] = 'input_segmentation_mask'
    else:
        zarr_group = zarr.open(zarr_path, mode='r+')
        zarr_dataset = zarr_group['interscellar_meshes']
    
    for result in chunk_results:
        if 'interscellar_mask' in result and 'union_bbox' in result:
            interscellar_mask = result['interscellar_mask']
            union_bbox = result['union_bbox']
            pair_id = result.get('pair_id', 0)
            
            if interscellar_mask.any() and pair_id > 0:
                z_start, z_stop = union_bbox[0].start, union_bbox[0].stop
                y_start, y_stop = union_bbox[1].start, union_bbox[1].stop
                x_start, x_stop = union_bbox[2].start, union_bbox[2].stop
                
                z_start = max(0, z_start)
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                z_stop = min(mask_3d.shape[0], z_stop)
                y_stop = min(mask_3d.shape[1], y_stop)
                x_stop = min(mask_3d.shape[2], x_stop)
                
                mesh_z_size = z_stop - z_start
                mesh_y_size = y_stop - y_start
                mesh_x_size = x_stop - x_start
                
                if (mesh_z_size > 0 and mesh_y_size > 0 and mesh_x_size > 0 and
                    mesh_z_size <= interscellar_mask.shape[0] and 
                    mesh_y_size <= interscellar_mask.shape[1] and 
                    mesh_x_size <= interscellar_mask.shape[2]):
                    
                    mesh_region_bool = interscellar_mask[:mesh_z_size, :mesh_y_size, :mesh_x_size]
                    mesh_region_labeled = (mesh_region_bool.astype(np.uint16) * pair_id)
                    
                    if np.any(mesh_region_labeled > 0):
                        existing_region = np.asarray(zarr_dataset[z_start:z_stop, y_start:y_stop, x_start:x_stop])
                        merged_region = np.maximum(existing_region, mesh_region_labeled)
                        zarr_dataset[z_start:z_stop, y_start:y_stop, x_start:x_stop] = merged_region
                        
                        del existing_region, merged_region
                        del mesh_region_labeled, mesh_region_bool  # Clean up source arrays too
                    
                    pairs_written += 1
                    
                    if pairs_written % 5 == 0:
                        import gc
                        gc.collect()
    
    if 'num_pairs' in zarr_group.attrs:
        zarr_group.attrs['num_pairs'] = zarr_group.attrs['num_pairs'] + pairs_written
    else:
        zarr_group.attrs['num_pairs'] = pairs_written
    
    del zarr_dataset, zarr_group
    import gc
    gc.collect()
    gc.collect()
    
    return pairs_written

def create_global_interscellar_mesh_zarr(
    volume_results: List[Dict[str, Any]], 
    mask_3d: np.ndarray,
    output_zarr_path: str,
    voxel_size_um: tuple
) -> None:
    print(f"Creating global interscellar mesh zarr file: {output_zarr_path}")
    
    global_mesh = np.zeros_like(mask_3d, dtype=np.uint16)
    
    pairs_written = 0
    for result in volume_results:
        if 'interscellar_mask' in result and 'union_bbox' in result:
            interscellar_mask = result['interscellar_mask']
            union_bbox = result['union_bbox']
            pair_id = result.get('pair_id', 0)
            
            if interscellar_mask.any() and pair_id > 0:
                z_start, z_stop = union_bbox[0].start, union_bbox[0].stop
                y_start, y_stop = union_bbox[1].start, union_bbox[1].stop
                x_start, x_stop = union_bbox[2].start, union_bbox[2].stop
                
                z_start = max(0, z_start)
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                z_stop = min(global_mesh.shape[0], z_stop)
                y_stop = min(global_mesh.shape[1], y_stop)
                x_stop = min(global_mesh.shape[2], x_stop)
                
                mesh_z_size = z_stop - z_start
                mesh_y_size = y_stop - y_start
                mesh_x_size = x_stop - x_start
                
                if (mesh_z_size > 0 and mesh_y_size > 0 and mesh_x_size > 0 and
                    mesh_z_size <= interscellar_mask.shape[0] and 
                    mesh_y_size <= interscellar_mask.shape[1] and 
                    mesh_x_size <= interscellar_mask.shape[2]):
                    
                    mesh_region_bool = interscellar_mask[:mesh_z_size, :mesh_y_size, :mesh_x_size]
                    mesh_region_labeled = (mesh_region_bool.astype(np.uint16) * pair_id)
                    
                    global_mesh[z_start:z_stop, y_start:y_stop, x_start:x_stop] = np.maximum(
                        global_mesh[z_start:z_stop, y_start:y_stop, x_start:x_stop],
                        mesh_region_labeled
                    )
                    pairs_written += 1
    
    print(f"Processed {pairs_written} pairs for mesh zarr")
    
    import zarr
    
    zarr_group = zarr.open(output_zarr_path, mode='w')
    
    zarr_group.create_dataset(
        'interscellar_meshes',
        data=global_mesh,
        chunks=(64, 64, 64),
        compression='gzip',
        compression_opts=6
    )
    
    zarr_group.attrs['description'] = 'Global interscellar volume meshes with unique pair IDs'
    zarr_group.attrs['voxel_size_um'] = voxel_size_um
    zarr_group.attrs['shape'] = global_mesh.shape
    zarr_group.attrs['dtype'] = str(global_mesh.dtype)
    zarr_group.attrs['num_pairs'] = pairs_written
    
    zarr_group.attrs['coordinate_system'] = 'same_as_input_segmentation'
    zarr_group.attrs['alignment_reference'] = 'input_segmentation_mask'
    
    print(f"Created global interscellar mesh zarr with {global_mesh.shape} shape")
    print(f"Contains {zarr_group.attrs['num_pairs']} interscellar volume pairs")
    
    return global_mesh

def create_global_cell_only_volumes_zarr(
    original_segmentation_zarr: str,
    interscellar_volumes_zarr: str,
    output_zarr_path: str
) -> None:
    import zarr
    import numpy as np
    import os
    
    print(f"Creating global cell-only volumes zarr using cookie cutter approach...")
    print(f"Original segmentation: {original_segmentation_zarr}")
    print(f"Interscellar volumes: {interscellar_volumes_zarr}")
    print(f"Output: {output_zarr_path}")
    
    if not os.path.exists(original_segmentation_zarr):
        raise FileNotFoundError(f"Original segmentation zarr not found: {original_segmentation_zarr}")
    if not os.path.exists(interscellar_volumes_zarr):
        raise FileNotFoundError(f"Interscellar volumes zarr not found: {interscellar_volumes_zarr}")
    
    print("Loading original segmentation...")
    original_zarr = zarr.open(original_segmentation_zarr, mode='r')
    
    original_segmentation = None
    if 'labels' in original_zarr:
        arr = original_zarr['labels']
        # Dimensions: 5D (t, c, z, y, x) or 3D (z, y, x)
        if arr.ndim == 5:
            original_segmentation = arr[0, 0]
        elif arr.ndim == 3:
            original_segmentation = arr
    elif '0' in original_zarr:
        if isinstance(original_zarr['0'], zarr.hierarchy.Group):
            if '0' in original_zarr['0']:
                arr = original_zarr['0']['0']
                if arr.ndim == 5:
                    original_segmentation = arr[0, 0]
                elif arr.ndim == 3:
                    original_segmentation = arr
        else:
            arr = original_zarr['0']
            if arr.ndim == 5:
                original_segmentation = arr[0, 0]
            elif arr.ndim == 3:
                original_segmentation = arr
    
    if original_segmentation is None:
        for key in original_zarr.keys():
            node = original_zarr[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                if node.ndim == 5:
                    original_segmentation = node[0, 0]
                elif node.ndim == 3:
                    original_segmentation = node
                break
    
    if original_segmentation is None:
        raise ValueError("Could not find 3D segmentation mask in original zarr file")
    
    print(f" Original segmentation shape: {original_segmentation.shape}, dtype: {original_segmentation.dtype}")
    original_segmentation = np.asarray(original_segmentation)
    
    if original_segmentation.size == 0:
        raise ValueError("Original segmentation array is empty")
    if not np.issubdtype(original_segmentation.dtype, np.integer):
        print(f"Warning: Original segmentation dtype is {original_segmentation.dtype}, converting to uint16")
        original_segmentation = original_segmentation.astype(np.uint16)
    
    print("Loading interscellar volumes...")
    if not os.path.exists(interscellar_volumes_zarr):
        raise FileNotFoundError(f"Interscellar volumes zarr not found: {interscellar_volumes_zarr}")
    
    interscellar_zarr = zarr.open(interscellar_volumes_zarr, mode='r')
    if 'interscellar_meshes' not in interscellar_zarr:
        available_keys = list(interscellar_zarr.keys())
        raise ValueError(
            f"Could not find 'interscellar_meshes' key in {interscellar_volumes_zarr}. "
            f"Available keys: {available_keys}"
        )
    
    interscellar_volumes_z = interscellar_zarr['interscellar_meshes']
    
    print(f"  Interscellar volumes shape: {interscellar_volumes_z.shape}, dtype: {interscellar_volumes_z.dtype}")
    interscellar_volumes = np.asarray(interscellar_volumes_z)
    
    if interscellar_volumes.size == 0:
        raise ValueError("Interscellar volumes array is empty")
    
    if original_segmentation.shape != interscellar_volumes.shape:
        raise ValueError(f"Shape mismatch: original {original_segmentation.shape} vs interscellar {interscellar_volumes.shape}")
    
    print(f"Processing arrays with shape: {original_segmentation.shape}")
    
    print("Applying cookie cutter subtraction...")
    cell_only_volumes = np.where(
        interscellar_volumes > 0, 
        0,
        original_segmentation 
    ).astype(np.uint16)
    
    print("Creating output zarr file...")
    
    output_key = 'labels' if 'labels' in original_zarr else '0'
    
    chunks = None
    compression = 'gzip'
    compression_opts = 6
    
    if output_key in original_zarr:
        original_dataset = original_zarr[output_key]
        if hasattr(original_dataset, 'chunks') and original_dataset.chunks:
            if original_dataset.ndim == 5:
                chunks = (1, 1) + original_dataset.chunks[-3:]
            elif original_dataset.ndim == 3:
                chunks = (1, 1, 64, 64, 64)
        if hasattr(original_dataset, 'compression') and original_dataset.compression:
            compression = original_dataset.compression
        if hasattr(original_dataset, 'compressor') and original_dataset.compressor:
            compression = original_dataset.compressor.codec_id
            compression_opts = getattr(original_dataset.compressor, 'level', 6)
    else:
        chunks = (1, 1, 64, 64, 64)
    
    output_zarr = zarr.open(output_zarr_path, mode='w')
    
    cell_only_volumes_5d = cell_only_volumes[None, None, :, :, :]
    print(f"  Creating dataset '{output_key}' with shape {cell_only_volumes_5d.shape}")
    
    try:
        output_zarr.create_dataset(
            output_key,
            data=cell_only_volumes_5d,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts if isinstance(compression_opts, int) else None
        )
    except Exception as e:
        output_zarr = None
        raise RuntimeError(f"Failed to create zarr dataset '{output_key}': {e}") from e
    
    print("Loading metadata from original segmentation...")
    try:
        for key, value in original_zarr.attrs.items():
            try:
                output_zarr.attrs[key] = value
            except Exception as e:
                print(f"Warning: Could not load metadata key '{key}': {e}")
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    
    output_zarr.attrs['description'] = 'Cell-only volumes (original cell segmentation minus interscellar volumes)'
    output_zarr.attrs['subtracted_volumes'] = 'interscellar_volumes'
    output_zarr.attrs['original_cells'] = len(np.unique(original_segmentation)) - 1
    output_zarr.attrs['remaining_cells'] = len(np.unique(cell_only_volumes)) - 1
    
    original_cell_voxels = (original_segmentation > 0).sum()
    remaining_cell_voxels = (cell_only_volumes > 0).sum()
    interscellar_voxels = (interscellar_volumes > 0).sum()
    
    if not os.path.exists(output_zarr_path):
        raise RuntimeError(f"Output zarr file was not created: {output_zarr_path}")
    
    try:
        verify_zarr = zarr.open(output_zarr_path, mode='r')
        if output_key not in verify_zarr:
            raise RuntimeError(f"Output zarr created but missing expected key '{output_key}'")
        verify_data = verify_zarr[output_key]
        if verify_data.size == 0:
            raise RuntimeError(f"Output zarr dataset '{output_key}' is empty")
        print(f"Verified output zarr file created successfully")
        del verify_zarr 
    except Exception as e:
        print(f"Warning: Could not verify output zarr: {e}")
    
    del output_zarr, original_zarr, interscellar_zarr
    
    print(f"Created global cell-only volumes zarr with {cell_only_volumes.shape} shape")
    print(f"Original cell voxels: {original_cell_voxels:,}")
    print(f"Remaining cell voxels: {remaining_cell_voxels:,}")
    print(f"Interscellar voxels: {interscellar_voxels:,}")
    if original_cell_voxels > 0:
        print(f"Subtraction efficiency: {interscellar_voxels/original_cell_voxels*100:.1f}% of original volume")
    else:
        print(f"Warning: Original cell voxels is 0")
    
    return cell_only_volumes

## Graph database

def create_interscellar_volume_database(db_path: str = 'interscellar_volumes.db') -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS interscellar_volumes")
    cursor.execute("DROP TABLE IF EXISTS cells")
    
    # Cell table
    cursor.execute("""
    CREATE TABLE cells (
        cell_id INTEGER PRIMARY KEY,
        cell_type TEXT,
        centroid_x REAL,
        centroid_y REAL,
        centroid_z REAL
    )
    """)
    
    # Interscellar Volume table
    cursor.execute("""
    CREATE TABLE interscellar_volumes (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cell_a_id INTEGER,
        cell_b_id INTEGER,
        cell_a_type TEXT,
        cell_b_type TEXT,
        -- Comprehensive interscellar volume
        total_interscellar_volume_um3 REAL,
        total_interscellar_volume_voxels INTEGER,
        -- Component breakdown
        edt_volume_um3 REAL,
        edt_volume_voxels INTEGER,
        intracellular_volume_um3 REAL,
        intracellular_volume_voxels INTEGER,
        touching_surface_area_um2 REAL,
        touching_surface_area_voxels INTEGER,
        -- Distance statistics
        mean_distance_um REAL,
        max_distance_um REAL,
        num_components INTEGER,
        largest_component_volume_um3 REAL,
        voxel_volume_um3 REAL,
        -- Parameters used
        max_distance_threshold_um REAL,
        intracellular_threshold_um REAL,
        FOREIGN KEY(cell_a_id) REFERENCES cells(cell_id),
        FOREIGN KEY(cell_b_id) REFERENCES cells(cell_id),
        UNIQUE(cell_a_id, cell_b_id)
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_cell_type ON cells(cell_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cells_centroid ON cells(centroid_x, centroid_y, centroid_z)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interscellar_cell_a ON interscellar_volumes(cell_a_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interscellar_cell_b ON interscellar_volumes(cell_b_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interscellar_cell_types ON interscellar_volumes(cell_a_type, cell_b_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interscellar_volume ON interscellar_volumes(total_interscellar_volume_um3)")
    
    conn.commit()
    return conn

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
    print(f"Populated cells table with {len(cells_data)} cells")

def get_cells_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM cells", conn)

def export_interscellar_volumes_to_csv(conn: sqlite3.Connection, output_file: str = 'interscellar_volumes.csv') -> None:
    df_volumes = pd.read_sql_query("SELECT * FROM interscellar_volumes", conn)
    df_volumes.to_csv(output_file, index=False)
    print(f"Interscellar volumes table saved as '{output_file}'")

def export_interscellar_volumes_to_duckdb(conn: sqlite3.Connection, output_file: str = 'interscellar_volumes.duckdb') -> None:
    try:
        import duckdb
    except ImportError:
        print("Error: DuckDB not available. Install with: pip install duckdb")
        return
    
    print(f"Exporting interscellar volumes to DuckDB: {output_file}")
    
    try:
        df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
        print(f"  - {len(df_cells)} cells (nodes)")
    except Exception as e:
        print(f"Warning: Could not load cells table: {e}")
        df_cells = pd.DataFrame()
    
    df_volumes = pd.read_sql_query("SELECT * FROM interscellar_volumes", conn)
    print(f"  - {len(df_volumes)} volume pairs")
    
    if df_volumes.empty:
        print("Warning: No volume data to export")
        return
    
    duckdb_conn = duckdb.connect(output_file)
    
    if not df_cells.empty:
        duckdb_conn.execute("""
            CREATE TABLE cells (
                cell_id INTEGER PRIMARY KEY,
                cell_type VARCHAR,
                centroid_x DOUBLE,
                centroid_y DOUBLE,
                centroid_z DOUBLE
            )
        """)
        duckdb_conn.register('df_cells', df_cells)
        duckdb_conn.execute("INSERT INTO cells SELECT * FROM df_cells")
    
    duckdb_conn.execute("""
        CREATE TABLE interscellar_volumes (
            pair_id INTEGER PRIMARY KEY,
            cell_a_id INTEGER,
            cell_b_id INTEGER,
            cell_a_type VARCHAR,
            cell_b_type VARCHAR,
            total_interscellar_volume_um3 DOUBLE,
            total_interscellar_volume_voxels INTEGER,
            edt_volume_um3 DOUBLE,
            edt_volume_voxels INTEGER,
            intracellular_volume_um3 DOUBLE,
            intracellular_volume_voxels INTEGER,
            touching_surface_area_um2 DOUBLE,
            touching_surface_area_voxels INTEGER,
            mean_distance_um DOUBLE,
            max_distance_um DOUBLE,
            num_components INTEGER,
            largest_component_volume_um3 DOUBLE,
            voxel_volume_um3 DOUBLE,
            max_distance_threshold_um DOUBLE,
            intracellular_threshold_um DOUBLE
        )
    """)
    
    duckdb_conn.register('df_volumes', df_volumes)
    duckdb_conn.execute("INSERT INTO interscellar_volumes SELECT * FROM df_volumes")
    
    print("Creating analytical views for volume analysis...")
    
    duckdb_conn.execute("""
        CREATE VIEW cell_type_volume_statistics AS
        SELECT 
            cell_a_type,
            cell_b_type,
            COUNT(*) as pair_count,
            AVG(total_interscellar_volume_um3) as mean_volume,
            MIN(total_interscellar_volume_um3) as min_volume,
            MAX(total_interscellar_volume_um3) as max_volume,
            SUM(total_interscellar_volume_um3) as total_volume,
            AVG(mean_distance_um) as mean_distance,
            AVG(num_components) as mean_components
        FROM interscellar_volumes
        GROUP BY cell_a_type, cell_b_type
        ORDER BY pair_count DESC
    """)
    
    duckdb_conn.execute("""
        CREATE VIEW volume_distribution AS
        SELECT 
            pair_id,
            cell_a_id,
            cell_b_id,
            cell_a_type,
            cell_b_type,
            total_interscellar_volume_um3,
            edt_volume_um3,
            intracellular_volume_um3,
            touching_surface_area_um2,
            mean_distance_um,
            max_distance_um,
            num_components,
            (edt_volume_um3 / NULLIF(total_interscellar_volume_um3, 0)) * 100 as edt_percentage,
            (intracellular_volume_um3 / NULLIF(total_interscellar_volume_um3, 0)) * 100 as intracellular_percentage,
            (touching_surface_area_um2 / NULLIF(total_interscellar_volume_um3, 0)) * 100 as surface_percentage
        FROM interscellar_volumes
    """)
    
    duckdb_conn.execute("""
        CREATE VIEW high_volume_interactions AS
        SELECT 
            pair_id,
            cell_a_id,
            cell_b_id,
            cell_a_type,
            cell_b_type,
            total_interscellar_volume_um3,
            mean_distance_um,
            num_components
        FROM interscellar_volumes
        WHERE total_interscellar_volume_um3 > (
            SELECT AVG(total_interscellar_volume_um3) + 2 * STDDEV(total_interscellar_volume_um3)
            FROM interscellar_volumes
        )
        ORDER BY total_interscellar_volume_um3 DESC
    """)
    
    duckdb_conn.execute("""
        CREATE TABLE metadata (
            key VARCHAR,
            value VARCHAR
        )
    """)
    
    metadata = [
        ('total_volume_pairs', str(len(df_volumes))),
        ('total_cells', str(len(df_cells)) if not df_cells.empty else '0'),
        ('unique_cell_types', str(df_volumes['cell_a_type'].nunique() + df_volumes['cell_b_type'].nunique()) if not df_volumes.empty else '0'),
        ('export_timestamp', pd.Timestamp.now().isoformat()),
        ('database_type', 'interscellar_volumes'),
        ('format', 'duckdb')
    ]
    
    for key, value in metadata:
        duckdb_conn.execute("INSERT INTO metadata VALUES (?, ?)", [key, value])
    
    duckdb_conn.close()
    
    print(f"DuckDB export completed: {output_file}")
    print(f"  - {len(df_volumes)} volume pairs")
    if not df_cells.empty:
        print(f"  - {len(df_cells)} cells (nodes)")
    print(f"  - Analytical views created for volume analysis")
    print(f"  - Metadata table populated")
    
    # Print example queries
    print("\nExample DuckDB queries:")
    print("1. Cell type volume stats: SELECT * FROM cell_type_volume_statistics")
    print("2. Volume distribution: SELECT * FROM volume_distribution LIMIT 10")
    print("3. High volume interactions: SELECT * FROM high_volume_interactions LIMIT 10")
    print("4. Metadata: SELECT * FROM metadata")

def get_anndata_from_interscellar_database(conn: sqlite3.Connection):
    if not ANNDATA_AVAILABLE:
        print("Warning: AnnData not available. Install with: pip install anndata")
        return None
    
    try:
        import anndata as ad
        from scipy.sparse import csr_matrix
    except ImportError:
        print("Warning: AnnData or scipy.sparse not available. Install with: pip install anndata scipy")
        return None
    
    try:
        df_cells = pd.read_sql_query("SELECT * FROM cells", conn)
        if df_cells.empty:
            raise ValueError("Cells table is empty")
        df_cells.set_index('cell_id', inplace=True)
    except Exception as e:
        print(f" No cells table found or empty ({e}), creating from interscellar volumes...")
        df_volumes = pd.read_sql_query("SELECT * FROM interscellar_volumes", conn)
        
        if df_volumes.empty:
            print("Warning: No interscellar volumes found in database")
            return None
        
        all_cells = set(df_volumes['cell_a_id'].unique()) | set(df_volumes['cell_b_id'].unique())
        
        cell_types = {}
        for _, row in df_volumes.iterrows():
            if pd.notna(row.get('cell_a_type')):
                cell_types[row['cell_a_id']] = row['cell_a_type']
            if pd.notna(row.get('cell_b_type')):
                cell_types[row['cell_b_id']] = row['cell_b_type']
        
        df_cells = pd.DataFrame({
            'cell_id': list(all_cells),
            'cell_type': [cell_types.get(cid, 'unknown') for cid in all_cells],
            'centroid_x': [0.0] * len(all_cells),
            'centroid_y': [0.0] * len(all_cells),
            'centroid_z': [0.0] * len(all_cells)
        })
        df_cells.set_index('cell_id', inplace=True)
        print(f"  Created cells dataframe with {len(df_cells)} cells from volume data")
    
    df_volumes = pd.read_sql_query("SELECT * FROM interscellar_volumes", conn)
    
    if df_volumes.empty:
        print("Warning: No interscellar volumes found in database")
        return None
    
    n_cells = len(df_cells)
    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
    
    cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(df_cells.index)}
    
    for _, row in df_volumes.iterrows():
        idx_a = cell_id_to_idx.get(row['cell_a_id'])
        idx_b = cell_id_to_idx.get(row['cell_b_id'])
        if idx_a is not None and idx_b is not None:
            volume = float(row['total_interscellar_volume_um3'])
            adjacency_matrix[idx_a, idx_b] = volume
            adjacency_matrix[idx_b, idx_a] = volume
    
    sparse_adjacency = csr_matrix(adjacency_matrix)
    
    edt_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
    intracellular_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
    touching_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
    
    for _, row in df_volumes.iterrows():
        idx_a = cell_id_to_idx.get(row['cell_a_id'])
        idx_b = cell_id_to_idx.get(row['cell_b_id'])
        if idx_a is not None and idx_b is not None:
            edt_matrix[idx_a, idx_b] = float(row.get('edt_volume_um3', 0.0))
            edt_matrix[idx_b, idx_a] = edt_matrix[idx_a, idx_b]
            
            intracellular_matrix[idx_a, idx_b] = float(row.get('intracellular_volume_um3', 0.0))
            intracellular_matrix[idx_b, idx_a] = intracellular_matrix[idx_a, idx_b]
            
            touching_matrix[idx_a, idx_b] = float(row.get('touching_surface_area_um2', 0.0))
            touching_matrix[idx_b, idx_a] = touching_matrix[idx_a, idx_b]
    
    adata = ad.AnnData(
        X=sparse_adjacency,  # Weighted adjacency matrix (total interscellar volumes)
        obs=df_cells,  # Cell metadata
        var=df_cells.copy(),  # Same metadata for variables
        obsp={'spatial_connectivities': sparse_adjacency}  # Store in obsp
    )
    
    adata.layers['edt_volume'] = csr_matrix(edt_matrix)
    adata.layers['intracellular_volume'] = csr_matrix(intracellular_matrix)
    adata.layers['touching_surface_area'] = csr_matrix(touching_matrix)
    
    max_dist_thresh = df_volumes['max_distance_threshold_um'].iloc[0] if len(df_volumes) > 0 else None
    intra_thresh = df_volumes['intracellular_threshold_um'].iloc[0] if len(df_volumes) > 0 else None
    
    volume_info = {
        'total_pairs': int(len(df_volumes)),
        'total_cells': int(n_cells),
        'graph_type': 'undirected_weighted',
        'edge_weights': 'total_interscellar_volume_um3',
    }
    
    if max_dist_thresh is not None:
        volume_info['max_distance_threshold_um'] = float(max_dist_thresh)
    if intra_thresh is not None:
        volume_info['intracellular_threshold_um'] = float(intra_thresh)
    
    if len(df_volumes) > 0:
        volume_info['mean_volume_um3'] = float(df_volumes['total_interscellar_volume_um3'].mean())
        volume_info['total_volume_um3'] = float(df_volumes['total_interscellar_volume_um3'].sum())
    else:
        volume_info['mean_volume_um3'] = 0.0
        volume_info['total_volume_um3'] = 0.0
    
    adata.uns['interscellar_volume_info'] = volume_info
    
    volume_records = df_volumes.to_dict('records')
    serializable_records = []
    for record in volume_records:
        serializable_record = {}
        for key, value in record.items():
            if hasattr(value, 'item'):  # numpy scalar
                serializable_record[key] = value.item()
            elif pd.isna(value): 
                serializable_record[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                serializable_record[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, (list, dict, tuple)):
                serializable_record[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                serializable_record[key] = str(value)
            else:
                serializable_record[key] = value
        serializable_records.append(serializable_record)
    
    if len(serializable_records) <= 1000:
        adata.uns['interscellar_volumes'] = serializable_records
    else:
        adata.uns['interscellar_volumes_count'] = len(serializable_records)
        adata.uns['interscellar_volumes_note'] = 'Full volume records available in database and CSV files'
    
    return adata

def export_interscellar_volumes_to_anndata(
    conn: sqlite3.Connection, 
    output_file: str = 'interscellar_volumes.h5ad'
):
    if not ANNDATA_AVAILABLE:
        print("Warning: AnnData not available. Install with: pip install anndata")
        return None
    
    adata = get_anndata_from_interscellar_database(conn)
    
    if adata is None:
        return None
    
    try:
        print(f"Writing AnnData to file: {output_file}")
        adata.write(output_file)
        
        import os
        if not os.path.exists(output_file):
            raise RuntimeError(f"AnnData file was not created: {output_file}")
        
        file_size = os.path.getsize(output_file)
        if file_size == 0:
            raise RuntimeError(f"AnnData file was created but is empty: {output_file}")
        
        print(f"AnnData object saved to '{output_file}' ({file_size:,} bytes)")
        print(f"  - {adata.n_obs} cells")
        print(f"  - {len(adata.uns.get('interscellar_volumes', []))} volume pairs")
        print(f"  - Weighted adjacency matrix shape: {adata.X.shape}")
        print(f"  - Component volumes stored in layers: edt_volume, intracellular_volume, touching_surface_area")
        return adata
    except Exception as e:
        import traceback
        print(f"Error saving AnnData file: {e}")
        print(f"Other outputs (CSV, DB, Zarr) still available")
        traceback.print_exc()
        return None

def build_interscellar_volume_database_from_neighbors(
    mask_3d: np.ndarray,
    neighbor_pairs_csv: str = None,
    neighbor_db_path: str = None,
    global_surface_pickle: str = None,
    halo_bboxes_pickle: str = None,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    db_path: str = 'interscellar_volumes.db',
    output_csv: str = None,
    output_anndata: str = None,
    output_mesh_zarr: str = None,
    max_distance_um: float = 3.0,
    intracellular_threshold_um: float = 1.0,
    n_jobs: int = 4,
    intermediate_results_dir: str = "intermediate_interscellar_results"
) -> sqlite3.Connection:
    print(f"Building interscellar volume database from pre-computed neighbor pairs")
    print(f"Voxel size: {voxel_size_um} μm")
    
    if neighbor_db_path and os.path.exists(neighbor_db_path):
        try:
            neighbor_pairs_df = load_neighbor_pairs_from_db(neighbor_db_path)
        except Exception as e:
            print(f"Warning: Could not load from database {neighbor_db_path}: {e}")
            print(f"Falling back to CSV: {neighbor_pairs_csv}")
            if neighbor_pairs_csv and os.path.exists(neighbor_pairs_csv):
                neighbor_pairs_df = load_neighbor_pairs_from_csv(neighbor_pairs_csv)
            else:
                raise ValueError(f"Neither database nor CSV file is available. DB: {neighbor_db_path}, CSV: {neighbor_pairs_csv}")
    elif neighbor_pairs_csv and os.path.exists(neighbor_pairs_csv):
        neighbor_pairs_df = load_neighbor_pairs_from_csv(neighbor_pairs_csv)
    else:
        raise ValueError(f"Must provide either neighbor_db_path or neighbor_pairs_csv. DB: {neighbor_db_path}, CSV: {neighbor_pairs_csv}")
    
    global_surface = load_global_surface_from_pickle(global_surface_pickle)
    halo_bboxes = load_halo_bboxes_from_pickle(halo_bboxes_pickle)
    
    conn = create_interscellar_volume_database(db_path)
    
    # Populate cells table from neighbor database if available
    if neighbor_db_path and os.path.exists(neighbor_db_path):
        try:
            import sqlite3 as sqlite3_neighbor
            neighbor_conn = sqlite3_neighbor.connect(neighbor_db_path)
            try:
                neighbor_cells = pd.read_sql_query("SELECT * FROM cells", neighbor_conn)
                if not neighbor_cells.empty:
                    # Ensure column names match
                    if 'cell_id' not in neighbor_cells.columns and 'CellID' in neighbor_cells.columns:
                        neighbor_cells = neighbor_cells.rename(columns={'CellID': 'cell_id'})
                    if 'cell_id' in neighbor_cells.columns:
                        neighbor_cells[['cell_id', 'cell_type', 'centroid_x', 'centroid_y', 'centroid_z']].to_sql(
                            'cells', conn, if_exists='replace', index=False
                        )
                        print(f"Populated cells table with {len(neighbor_cells)} cells from neighbor database")
            except Exception as e:
                print(f"Warning: Could not load cells from neighbor database: {e}")
            finally:
                neighbor_conn.close()
        except Exception as e:
            print(f"Warning: Could not access neighbor database for cells: {e}")
    
    # If cells table is still empty, create from volume pairs
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cells")
    if cursor.fetchone()[0] == 0:
        print("  Creating cells table from volume pairs...")
        all_cells = set(neighbor_pairs_df['cell_a_id'].unique()) | set(neighbor_pairs_df['cell_b_id'].unique())
        cell_types = {}
        for _, row in neighbor_pairs_df.iterrows():
            if pd.notna(row.get('cell_a_type')):
                cell_types[row['cell_a_id']] = row['cell_a_type']
            if pd.notna(row.get('cell_b_type')):
                cell_types[row['cell_b_id']] = row['cell_b_type']
        
        cells_data = []
        for cell_id in all_cells:
            cells_data.append((
                int(cell_id),
                cell_types.get(cell_id, 'unknown'),
                0.0, 0.0, 0.0  # Placeholder centroids
            ))
        cursor.executemany(
            "INSERT INTO cells (cell_id, cell_type, centroid_x, centroid_y, centroid_z) VALUES (?, ?, ?, ?, ?)",
            cells_data
        )
        conn.commit()
        print(f"Created cells table with {len(cells_data)} cells from neighbor pairs")
    
    print("Computing interscellar volumes for all neighbor pairs...")
    volume_results = compute_interscellar_volumes_for_neighbor_pairs(
        mask_3d, neighbor_pairs_df, voxel_size_um, global_surface, halo_bboxes,
        max_distance_um, intracellular_threshold_um, n_jobs, intermediate_results_dir,
        output_mesh_zarr=output_mesh_zarr
    )
    
    if volume_results:
        cursor = conn.cursor()
        volume_data = []
        for result in volume_results:
            volume_data.append((
                result['cell_a_id'], 
                result['cell_b_id'], 
                result['cell_a_type'], 
                result['cell_b_type'],
                # Interscellar volume
                result['total_interscellar_volume_um3'],
                result['total_interscellar_volume_voxels'],
                # Volume components
                result['edt_volume_um3'],
                result['edt_volume_voxels'],
                result['intracellular_volume_um3'],
                result['intracellular_volume_voxels'],
                result['touching_surface_area_um2'],
                result['touching_surface_area_voxels'],
                # Distance statistics
                result['mean_distance_um'],
                result['max_distance_um'],
                result['num_components'],
                result['largest_component_volume_um3'],
                result['voxel_volume_um3'],
                # Parameters used
                result['max_distance_threshold_um'],
                result['intracellular_threshold_um']
            ))
        
        try:
            cursor.executemany(
                """INSERT INTO interscellar_volumes 
                   (cell_a_id, cell_b_id, cell_a_type, cell_b_type, 
                    total_interscellar_volume_um3, total_interscellar_volume_voxels,
                    edt_volume_um3, edt_volume_voxels, intracellular_volume_um3, intracellular_volume_voxels,
                    touching_surface_area_um2, touching_surface_area_voxels,
                    mean_distance_um, max_distance_um, num_components,
                    largest_component_volume_um3, voxel_volume_um3,
                    max_distance_threshold_um, intracellular_threshold_um) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                volume_data
            )
            conn.commit()
            print(f"Successfully inserted {len(volume_results)} interscellar volume records into database")
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                print(f"Warning: Some duplicate pairs were found and skipped due to UNIQUE constraint")
                inserted_count = 0
                for data in volume_data:
                    try:
                        cursor.execute(
                            """INSERT INTO interscellar_volumes 
                               (cell_a_id, cell_b_id, cell_a_type, cell_b_type, 
                                total_interscellar_volume_um3, total_interscellar_volume_voxels,
                                edt_volume_um3, edt_volume_voxels, intracellular_volume_um3, intracellular_volume_voxels,
                                touching_surface_area_um2, touching_surface_area_voxels,
                                mean_distance_um, max_distance_um, num_components,
                                largest_component_volume_um3, voxel_volume_um3,
                                max_distance_threshold_um, intracellular_threshold_um) 
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                            data
                        )
                        inserted_count += 1
                    except sqlite3.IntegrityError:
                        continue
                conn.commit()
                print(f"Successfully inserted {inserted_count} unique interscellar volume records into database")
            else:
                raise e
    else:
        print("No interscellar volumes found")
    
    # Export to CSV
    if output_csv:
        export_interscellar_volumes_to_csv(conn, output_csv)
    
    # Export to AnnData
    if output_anndata:
        print(f"\nExporting to AnnData format: {output_anndata}")
        try:
            result = export_interscellar_volumes_to_anndata(conn, output_anndata)
            if result is not None:
                print(f"AnnData export completed successfully")
            else:
                print(f"Warning: AnnData export returned None (may have failed silently)")
                print(f"Check error messages above for details.")
                print(f"Other outputs (CSV, DB, Zarr) are still available.")
        except Exception as e:
            print(f"Warning: AnnData export failed (non-fatal): {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Other outputs (CSV, DB, Zarr) are still available.")
            import traceback
            traceback.print_exc()
    
    if output_mesh_zarr:
        import zarr
        if os.path.exists(output_mesh_zarr) and os.path.isdir(output_mesh_zarr):
            zarr_group = zarr.open(output_mesh_zarr, mode='r')
            final_pairs = zarr_group.attrs.get('num_pairs', 0)
            zarr_shape = zarr_group['interscellar_meshes'].shape
            print(f"Final mesh zarr complete:")
            print(f"Shape: {zarr_shape}")
            print(f"Total pairs: {final_pairs}")
            print(f"Unique pair IDs range from 1 to {np.asarray(zarr_group['interscellar_meshes']).max()}")
        else:
            has_masks = any('interscellar_mask' in r for r in volume_results)
            if has_masks:
                create_global_interscellar_mesh_zarr(
                    volume_results, mask_3d, output_mesh_zarr, voxel_size_um
                )
            else:
                print(f"Warning: Mesh zarr not created. Results don't contain masks.")
    
    return conn, volume_results

def save_surfaces_to_pickle(surfaces: Dict[int, np.ndarray], filepath: str = "cell_surfaces.pkl") -> None:
    with open(filepath, "wb") as f:
        pickle.dump(surfaces, f)
    print(f"Cell surfaces saved to pickle file: {filepath}")

def load_surfaces_from_pickle(filepath: str = "cell_surfaces.pkl") -> Dict[int, np.ndarray]:
    with open(filepath, "rb") as f:
        surfaces = pickle.load(f)
    print(f"Loaded surfaces for {len(surfaces)} cells from pickle file: {filepath}")
    return surfaces

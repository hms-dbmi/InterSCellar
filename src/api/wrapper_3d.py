# Import

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import time
import sqlite3

from find_cell_neighbors_3d import(
    create_neighbor_edge_table_database_optimized,
    find_all_neighbors_by_surface_distance_optimized,
    create_neighbor_edge_table_optimized,
    get_anndata_from_database,
    query_cell_type_pairs,
    get_graph_statistics,
    save_surfaces_to_pickle,
    load_surfaces_from_pickle,
    save_graph_state_to_pickle,
    load_graph_state_from_pickle
)

from compute_interscellar_volumes_3d import(
    build_interscellar_volume_database_from_neighbors,
    create_global_interscellar_mesh_zarr,
    create_global_cell_only_volumes_zarr,
    export_interscellar_volumes_to_duckdb,
    get_anndata_from_interscellar_database,
    export_interscellar_volumes_to_anndata,
    ANNDATA_AVAILABLE
)

# API: Wrapper functions

def find_cell_neighbors_3d(
    ome_zarr_path: str,
    metadata_csv_path: str,
    max_distance_um: float = 0.5,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    centroid_prefilter_radius_um: float = 75.0,
    cell_id: str = 'CellID',
    cell_type: str = 'phenotype',
    centroid_x: str = 'X_centroid',
    centroid_y: str = 'Y_centroid',
    centroid_z: str = 'Z_centroid',
    db_path: str = 'cell_neighbor_graph.db',
    output_csv: Optional[str] = None,
    output_anndata: Optional[str] = None,
    n_jobs: int = 1,
    return_connection: bool = False,
    save_surfaces_pickle: Optional[str] = None,
    load_surfaces_pickle: Optional[str] = None,
    save_graph_state_pickle: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[object], Optional[object]]:
    
    print("=" * 60)
    print("InterSCellar: Surface-based Cell Neighbor Detection - 3D")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    print(f"\n1. Loading metadata from: {metadata_csv_path}...")
    step1_start = time.time()
    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"Loaded {len(metadata_df)} cells")
    except Exception as e:
        raise ValueError(f"Error loading metadata CSV: {e}")
    
    required_cols = [cell_id, cell_type, centroid_x, centroid_y, centroid_z]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    step1_time = time.time() - step1_start
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    print(f"\n2. Building neighbor graph...")
    print(f"Parameters: max_distance={max_distance_um}μm, n_jobs={n_jobs}")
    if max_distance_um == 0.0:
        print(f"Mode: Touching cells only")
    else:
        print(f"Mode: Touching cells + near-neighbors (on-demand surface extraction)")
    
    step2_start = time.time()
    
    try:
        conn = create_neighbor_edge_table_database_optimized(
            ome_zarr_path=ome_zarr_path,
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
            output_csv=output_csv,
            output_anndata=output_anndata,
            n_jobs=n_jobs,
            save_surfaces_pickle=save_surfaces_pickle,
            load_surfaces_pickle=load_surfaces_pickle,
            save_graph_state_pickle=save_graph_state_pickle
        )
        
        if save_surfaces_pickle:
            print(f"   Saving global surface to: {save_surfaces_pickle}")
        
        if save_graph_state_pickle:
            print(f"   Saving graph state to: {save_graph_state_pickle}")
        
        step2_time = time.time() - step2_start
        print(f"Neighbor graph created successfully")
        print(f"Step 2 completed in {step2_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error in neighbor detection pipeline: {e}")
    
    print(f"\n3. Retrieving results...")
    step3_start = time.time()
    
    neighbor_table_df = None
    if output_csv:
        try:
            neighbor_table_df = pd.read_sql_query("SELECT * FROM neighbors", conn)
            print(f"Neighbor table: {len(neighbor_table_df)} pairs")
        except Exception as e:
            print(f"Warning: Could not retrieve neighbor table: {e}")
    
    adata = None
    try:
        adata = get_anndata_from_database(conn)
        if adata is not None:
            print(f"AnnData object created: {adata.shape}")
        else:
            print(f"Warning: AnnData not available (install with: pip install anndata)")
    except Exception as e:
        print(f"Warning: Could not create AnnData object: {e}")
    
    try:
        stats = get_graph_statistics(conn)
        print(f"Graph statistics: {stats['total_cells']} cells, {stats['total_edges']} pairs")
    except Exception as e:
        print(f"Warning: Could not retrieve statistics: {e}")
    
    step3_time = time.time() - step3_start
    print(f"Step 3 completed in {step3_time:.2f} seconds")
    
    overall_time = time.time() - overall_start_time
    print(f"\n4. Pipeline completed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print(f"Database: {db_path}")
    if output_csv:
        print(f"CSV output: {output_csv}")
    if output_anndata:
        print(f"AnnData output: {output_anndata}")
    
    print("=" * 60)
    
    if return_connection:
        return neighbor_table_df, adata, conn
    else:
        conn.close()
        return neighbor_table_df, adata, None

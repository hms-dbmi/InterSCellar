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

def compute_interscellar_volumes(
    ome_zarr_path: str,
    neighbor_pairs_csv: str,
    global_surface_pickle: str,
    halo_bboxes_pickle: str,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    db_path: str = 'interscellar_volumes.db',
    output_csv: Optional[str] = None,
    output_anndata: Optional[str] = None,
    output_mesh_zarr: Optional[str] = None,
    output_cell_only_zarr: Optional[str] = None,
    max_distance_um: float = 3.0,
    intracellular_threshold_um: float = 1.0,
    n_jobs: int = 4,
    return_connection: bool = False,
    intermediate_results_dir: str = "intermediate_interscellar_results"
) -> Tuple[Optional[pd.DataFrame], Optional[object], Optional[object]]:
    
    print("=" * 60)
    print("InterSCellar: Volume Computation - 3D")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    import os
    if output_mesh_zarr is None:
        base_name = os.path.splitext(db_path)[0]
        output_mesh_zarr = f"{base_name}_interscellar_volumes.zarr"
        print(f"Auto-setting output_mesh_zarr: {output_mesh_zarr}")
    
    if output_cell_only_zarr is None:
        base_name = os.path.splitext(db_path)[0]
        output_cell_only_zarr = f"{base_name}_cell_only_volumes.zarr"
        print(f"Auto-setting output_cell_only_zarr: {output_cell_only_zarr}")
    
    print(f"\n1. Validating input files...")
    step1_start = time.time()
    
    required_files = [ome_zarr_path, neighbor_pairs_csv, global_surface_pickle, halo_bboxes_pickle]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print(f"All input files found")
    step1_time = time.time() - step1_start
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    print(f"\n2. Computing interscellar volumes...")
    print(f"Parameters: max_distance={max_distance_um}μm, intracellular_threshold={intracellular_threshold_um}μm")
    print(f"Voxel size: {voxel_size_um} μm")
    
    step2_start = time.time()
    
    try:
        import zarr
        print(f"Loading segmentation mask from: {ome_zarr_path}")
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
        
        print(f"Mask shape: {mask_3d.shape}, dtype: {mask_3d.dtype}")
        
        if mask_3d.dtype.byteorder == '>':
            mask_3d = mask_3d.astype(mask_3d.dtype.newbyteorder('='))
        
        if output_anndata is None:
            base_name = os.path.splitext(db_path)[0]
            output_anndata = f"{base_name}.h5ad"
            print(f"Auto-setting output_anndata: {output_anndata}")
        
        conn, volume_results = build_interscellar_volume_database_from_neighbors(
            mask_3d=mask_3d,
            neighbor_pairs_csv=neighbor_pairs_csv,
            global_surface_pickle=global_surface_pickle,
            halo_bboxes_pickle=halo_bboxes_pickle,
            voxel_size_um=voxel_size_um,
            db_path=db_path,
            output_csv=output_csv,
            output_anndata=output_anndata,
            output_mesh_zarr=output_mesh_zarr,
            max_distance_um=max_distance_um,
            intracellular_threshold_um=intracellular_threshold_um,
            n_jobs=n_jobs,
            intermediate_results_dir=intermediate_results_dir
        )
        
        print(f"\n3. Verifying mesh zarr completion...")
        import os
        import zarr
        mesh_zarr_exists = False
        if output_mesh_zarr and os.path.exists(output_mesh_zarr) and os.path.isdir(output_mesh_zarr):
            try:
                zarr_group = zarr.open(output_mesh_zarr, mode='r')
                if 'interscellar_meshes' in zarr_group:
                    final_pairs = zarr_group.attrs.get('num_pairs', 0)
                    zarr_shape = zarr_group['interscellar_meshes'].shape
                    max_pair_id = np.asarray(zarr_group['interscellar_meshes']).max()
                    print(f"Mesh zarr verified: {output_mesh_zarr}")
                    print(f"Shape: {zarr_shape}")
                    print(f"Total pairs written: {final_pairs}")
                    print(f"Max pair ID: {max_pair_id}")
                    mesh_zarr_exists = True
                else:
                    print(f"Mesh zarr exists but missing 'interscellar_meshes' key")
            except Exception as e:
                print(f"Error verifying mesh zarr: {e}")
        else:
            print(f"Mesh zarr not found at: {output_mesh_zarr}")
        
        if output_cell_only_zarr:
            print(f"\n4. Creating cell-only volumes zarr...")
            if not mesh_zarr_exists:
                print(f"Warning: Interscellar mesh zarr not available. Skipping cell-only zarr creation.")
                print(f"Run the cell-only zarr creation separately after the interscellar zarr is ready.")
            else:
                try:
                    create_global_cell_only_volumes_zarr(
                        original_segmentation_zarr=ome_zarr_path,
                        interscellar_volumes_zarr=output_mesh_zarr,
                        output_zarr_path=output_cell_only_zarr
                    )
                    if os.path.exists(output_cell_only_zarr):
                        print(f"Cell-only volumes zarr created and verified: {output_cell_only_zarr}")
                    else:
                        print(f"Warning: Cell-only zarr creation reported success but file not found")
                except Exception as e:
                    print(f"Error: Failed to create cell-only volumes zarr: {e}")
                    print(f"Other outputs (CSV, DB, Zarr) still available")
                    import traceback
                    traceback.print_exc()
        
        step2_time = time.time() - step2_start
        print(f"Interscellar volumes computed successfully")
        print(f"Step 2 completed in {step2_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error in interscellar volume computation pipeline: {e}")
    
    print(f"\n5. Retrieving results")
    step3_start = time.time()
    
    volume_results_df = None
    if output_csv:
        try:
            volume_results_df = pd.read_sql_query("SELECT * FROM interscellar_volumes", conn)
            print(f"Volume results table: {len(volume_results_df)} pairs")
        except Exception as e:
            print(f"Warning: Could not retrieve volume results table: {e}")
    
    adata = None
    if output_anndata:
        if os.path.exists(output_anndata):
            try:
                if ANNDATA_AVAILABLE:
                    try:
                        import anndata as ad
                        adata = ad.read_h5ad(output_anndata)
                        print(f"AnnData file verified: {output_anndata}")
                        print(f"Shape: {adata.shape}")
                        print(f"Weighted adjacency matrix with interscellar volumes")
                        print(f"Component volumes in layers: edt_volume, intracellular_volume, touching_surface_area")
                    except Exception as e:
                        print(f"Warning: AnnData file exists but could not be loaded: {e}")
                else:
                    print(f"AnnData file created: {output_anndata}")
                    print(f"Warning: AnnData package not available for verification (install with: pip install anndata)")
            except Exception as e:
                print(f"Warning: Could not verify AnnData file: {e}")
        else:
            print(f"Warning: AnnData file not found at: {output_anndata}")
    
    step3_time = time.time() - step3_start
    print(f"Step 3 completed in {step3_time:.2f} seconds")
    
    print(f"\n6. Exporting to DuckDB format...")
    step4_start = time.time()
    try:
        duckdb_output = db_path.replace('.db', '.duckdb')
        export_interscellar_volumes_to_duckdb(conn, duckdb_output)
        step4_time = time.time() - step4_start
        print(f"DuckDB export completed in {step4_time:.2f} seconds")
    except Exception as e:
        print(f"Warning: DuckDB export failed: {e}")
    
    overall_time = time.time() - overall_start_time
    print(f"\n7. Pipeline completed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print(f"Database: {db_path}")
    print(f"DuckDB: {db_path.replace('.db', '.duckdb')}")
    if output_csv:
        print(f"CSV output: {output_csv}")
    if output_anndata:
        print(f"AnnData output: {output_anndata}")
    print(f"Interscellar volumes zarr: {output_mesh_zarr}")
    print(f"Cell-only volumes zarr: {output_cell_only_zarr}")
    
    print("=" * 60)
    
    if return_connection:
        return volume_results_df, adata, conn
    else:
        conn.close()
        return volume_results_df, adata, None

# Import

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import time
import sqlite3
import os

from ..core.find_cell_neighbors_3d import(
    create_neighbor_edge_table_database_3d,
    get_anndata_from_database,
    query_cell_type_pairs,
    get_graph_statistics,
    save_surfaces_to_pickle,
    load_surfaces_from_pickle,
    save_graph_state_to_pickle,
    load_graph_state_from_pickle
)

from ..core.compute_interscellar_volumes_3d import(
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
    db_path: Optional[str] = None,
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
    
    metadata_dir = os.path.dirname(metadata_csv_path) if os.path.dirname(metadata_csv_path) else "."
    base_name = os.path.splitext(os.path.basename(metadata_csv_path))[0]
    
    if db_path is None:
        db_path = os.path.join(metadata_dir, f"{base_name}_neighbor_graph.db")
        print(f"db_path: {db_path}")
    
    if output_csv is None:
        output_csv = os.path.join(metadata_dir, f"{base_name}_neighbors_3d.csv")
        print(f"output_csv: {output_csv}")
    
    if output_anndata is None:
        output_anndata = os.path.join(metadata_dir, f"{base_name}_neighbors_3d.h5ad")
        print(f"output_anndata: {output_anndata}")
    
    if save_surfaces_pickle is None:
        base_name = os.path.splitext(db_path)[0]
        save_surfaces_pickle = f"{base_name}_surfaces.pkl"
        print(f"Surfaces pickle path: {save_surfaces_pickle}")
    
    if save_graph_state_pickle is None:
        base_name = os.path.splitext(db_path)[0]
        save_graph_state_pickle = f"{base_name}_graph_state.pkl"
        print(f"Graph state pickle path: {save_graph_state_pickle}")
    
    print(f"\n2. Building neighbor graph...")
    print(f"Parameters: max_distance={max_distance_um}μm, n_jobs={n_jobs}")
    if max_distance_um == 0.0:
        print(f"Mode: Touching cells only")
    else:
        print(f"Mode: Touching cells + near-neighbors")
    
    step2_start = time.time()
    
    try:
        conn = create_neighbor_edge_table_database_3d(
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
        
        print(f"Saving global surface to: {save_surfaces_pickle}")
        print(f"Saving graph state to: {save_graph_state_pickle}")
        
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

def compute_interscellar_volumes_3d(
    ome_zarr_path: str,
    neighbor_pairs_csv: str,
    global_surface_pickle: Optional[str] = None,
    halo_bboxes_pickle: Optional[str] = None,
    neighbor_db_path: Optional[str] = None,
    voxel_size_um: tuple = (0.56, 0.28, 0.28),
    db_path: Optional[str] = None,
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
    
    csv_dir = os.path.dirname(neighbor_pairs_csv) if os.path.dirname(neighbor_pairs_csv) else "."
    csv_base_name = os.path.splitext(os.path.basename(neighbor_pairs_csv))[0]
    csv_base_name = csv_base_name.replace("_neighbors_3d", "").replace("_neighbors", "").replace("neighbors", "")
    
    if neighbor_db_path is None:
        possible_db = os.path.join(csv_dir, f"{csv_base_name}_neighbor_graph.db")
        if os.path.exists(possible_db):
            neighbor_db_path = possible_db
            print(f"Detected neighbor_db_path: {neighbor_db_path}")
        else:
            alt_db = os.path.join(csv_dir, os.path.basename(neighbor_pairs_csv).replace("_neighbors_3d.csv", "_neighbor_graph.db").replace("_neighbors.csv", "_neighbor_graph.db").replace(".csv", "_neighbor_graph.db"))
            if os.path.exists(alt_db):
                neighbor_db_path = alt_db
                print(f"Detected neighbor_db_path: {neighbor_db_path}")
    
    if db_path is None:
        db_path = os.path.join(csv_dir, f"{csv_base_name}_interscellar_volumes.db")
        print(f"db_path: {db_path}")
    
    if output_csv is None:
        output_csv = os.path.join(csv_dir, f"{csv_base_name}_volumes.csv")
        print(f"output_csv: {output_csv}")
    
    if output_anndata is None:
        output_anndata = os.path.join(csv_dir, f"{csv_base_name}_volumes.h5ad")
        print(f"output_anndata: {output_anndata}")
    
    if global_surface_pickle is None or halo_bboxes_pickle is None:
        if neighbor_db_path:
            base_name = os.path.splitext(neighbor_db_path)[0]
            search_dir = os.path.dirname(neighbor_db_path) if os.path.dirname(neighbor_db_path) else "."
        else:
            csv_dir = os.path.dirname(neighbor_pairs_csv) if os.path.dirname(neighbor_pairs_csv) else "."
            csv_basename = os.path.basename(neighbor_pairs_csv)
            base_name_stem = csv_basename.replace("_neighbors_3d.csv", "").replace("_neighbors.csv", "").replace("neighbors.csv", "").replace(".csv", "")
            possible_db = os.path.join(csv_dir, f"{base_name_stem}_neighbor_graph.db")
            if os.path.exists(possible_db):
                base_name = os.path.splitext(possible_db)[0]
                search_dir = csv_dir
            else:
                base_name = os.path.join(csv_dir, base_name_stem)
                search_dir = csv_dir
        
        if global_surface_pickle is None:
            possible_surfaces = [
                f"{base_name}_surfaces.pkl",
                f"{base_name}_graph_surfaces.pkl",
            ]
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'surface' in f.lower() and os.path.basename(base_name) in f:
                        possible_surfaces.append(os.path.join(search_dir, f))
            
            for path in possible_surfaces:
                if os.path.exists(path):
                    global_surface_pickle = path
                    print(f"Detected global_surface_pickle: {global_surface_pickle}")
                    break
            else:
                global_surface_pickle = f"{base_name}_surfaces.pkl"
        
        if halo_bboxes_pickle is None:
            possible_halo = [
                f"{base_name}_halo_bboxes.pkl",
                f"{base_name.replace('_surfaces', '_halo_bboxes')}.pkl",
            ]
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'halo' in f.lower() and os.path.basename(base_name) in f:
                        possible_halo.append(os.path.join(search_dir, f))
            
            for path in possible_halo:
                if os.path.exists(path):
                    halo_bboxes_pickle = path
                    print(f"Detected halo_bboxes_pickle: {halo_bboxes_pickle}")
                    break
            else:
                halo_bboxes_pickle = f"{base_name}_halo_bboxes.pkl"
    
    if output_mesh_zarr is None:
        base_name = os.path.splitext(db_path)[0]
        if base_name.endswith('_interscellar_volumes'):
            base_name = base_name[:-len('_interscellar_volumes')]
        output_mesh_zarr = f"{base_name}_interscellar_volumes.zarr"
        print(f"output_mesh_zarr: {output_mesh_zarr}")
    
    if output_cell_only_zarr is None:
        base_name = os.path.splitext(db_path)[0]
        if base_name.endswith('_interscellar_volumes'):
            base_name = base_name[:-len('_interscellar_volumes')]
        output_cell_only_zarr = f"{base_name}_cell_only_volumes.zarr"
        print(f"output_cell_only_zarr: {output_cell_only_zarr}")
    
    print(f"\n1. Validating input files...")
    step1_start = time.time()
    
    required_files = [ome_zarr_path]
    
    if not neighbor_pairs_csv and not neighbor_db_path:
        raise ValueError("Must provide either neighbor_pairs_csv or neighbor_db_path")
    
    if neighbor_pairs_csv and not os.path.exists(neighbor_pairs_csv):
        if neighbor_db_path and os.path.exists(neighbor_db_path):
            print(f"Warning: neighbor_pairs_csv not found: {neighbor_pairs_csv}")
            print(f"Use db instead: {neighbor_db_path}")
        else:
            raise FileNotFoundError(f"neighbor_pairs_csv not found: {neighbor_pairs_csv}")
    elif neighbor_pairs_csv:
        required_files.append(neighbor_pairs_csv)
    
    if neighbor_db_path and not os.path.exists(neighbor_db_path):
        if neighbor_pairs_csv and os.path.exists(neighbor_pairs_csv):
            print(f"Warning: neighbor_db_path not found: {neighbor_db_path}")
            print(f"Use CSV instead: {neighbor_pairs_csv}")
        else:
            raise FileNotFoundError(f"neighbor_db_path not found: {neighbor_db_path}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    if global_surface_pickle and not os.path.exists(global_surface_pickle):
        print(f"Warning: global_surface_pickle not found: {global_surface_pickle}")
        print(f"Loading from graph_state.pkl as fallback")
    elif global_surface_pickle:
        print(f"Using global_surface_pickle: {global_surface_pickle}")
    
    if halo_bboxes_pickle and not os.path.exists(halo_bboxes_pickle):
        print(f"Warning: halo_bboxes_pickle not found: {halo_bboxes_pickle}")
    elif halo_bboxes_pickle:
        print(f"Using halo_bboxes_pickle: {halo_bboxes_pickle}")
    
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
            neighbor_db_path=neighbor_db_path,  # Use database if available (more efficient)
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
    
    try:
        from ..core.compute_interscellar_volumes_3d import _cleanup_intermediate_results
        _cleanup_intermediate_results(intermediate_results_dir)
    except Exception as e:
        print(f"Warning: Could not clean up intermediate results: {e}")
        print(f"Intermediate results directory: {intermediate_results_dir}")
    
    print("=" * 60)
    
    if return_connection:
        return volume_results_df, adata, conn
    else:
        conn.close()
        return volume_results_df, adata, None

def compute_cell_only_volumes_3d(
    ome_zarr_path: str,
    interscellar_volumes_zarr: str,
    output_zarr_path: Optional[str] = None,
    neighbor_db_path: Optional[str] = None
) -> pd.DataFrame:
    print("=" * 60)
    print("InterSCellar: Cell-Only Volumes Computation - 3D")
    print("=" * 60)
    
    import os
    import zarr
    import numpy as np
    from ..core.compute_interscellar_volumes_3d import create_global_cell_only_volumes_zarr
    
    if output_zarr_path is None:
        interscellar_dir = os.path.dirname(interscellar_volumes_zarr) if os.path.dirname(interscellar_volumes_zarr) else "."
        interscellar_basename = os.path.basename(interscellar_volumes_zarr)
        base_name = os.path.splitext(interscellar_basename)[0]
        
        while base_name.endswith('_interscellar_volumes'):
            base_name = base_name[:-len('_interscellar_volumes')]
        
        if not base_name.endswith('_cell_only_volumes'):
            output_basename = base_name + '_cell_only_volumes.zarr'
        else:
            output_basename = base_name + '.zarr'
        
        output_zarr_path = os.path.join(interscellar_dir, output_basename)
    
    if not os.path.exists(ome_zarr_path):
        raise FileNotFoundError(f"Cell segmentation zarr not found: {ome_zarr_path}")
    if not os.path.exists(interscellar_volumes_zarr):
        raise FileNotFoundError(f"Interscellar volumes zarr not found: {interscellar_volumes_zarr}")
    
    try:
        interscellar_zarr = zarr.open(interscellar_volumes_zarr, mode='r')
        if 'voxel_size_um' in interscellar_zarr.attrs:
            voxel_size_um = tuple(interscellar_zarr.attrs['voxel_size_um'])
            print(f"Detected voxel_size_um from interscellar zarr: {voxel_size_um}")
        else:
            voxel_size_um = (0.56, 0.28, 0.28)
            print(f"Warning: voxel_size_um not found in zarr attributes, using default: {voxel_size_um}")
        del interscellar_zarr
    except Exception as e:
        voxel_size_um = (0.56, 0.28, 0.28)
        print(f"Warning: Could not read voxel_size_um from zarr, using default: {voxel_size_um} ({e})")
    
    print(f"\nInput files:")
    print(f"Cell segmentation: {ome_zarr_path}")
    print(f"Interscellar volumes: {interscellar_volumes_zarr}")
    print(f"Output zarr: {output_zarr_path}")
    
    try:
        cell_only_mask = create_global_cell_only_volumes_zarr(
            original_segmentation_zarr=ome_zarr_path,
            interscellar_volumes_zarr=interscellar_volumes_zarr,
            output_zarr_path=output_zarr_path
        )
        print(f"Cell-only volumes zarr created: {output_zarr_path}")
    except Exception as e:
        print(f"\nError creating cell-only volumes zarr: {e}")
        print("=" * 60)
        raise
    
    print(f"\nComputing cell-only volume measurements...")
    voxel_volume_um3 = np.prod(voxel_size_um)
    
    print(f"Computing volumes for all cells...")
    unique_cells, counts = np.unique(cell_only_mask, return_counts=True)

    mask = unique_cells > 0
    unique_cells = unique_cells[mask]
    counts = counts[mask]
    
    print(f"Found {len(unique_cells)} cells in cell-only volumes")
    
    cell_types = {}
    if neighbor_db_path and os.path.exists(neighbor_db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(neighbor_db_path)
            try:
                cells_df = pd.read_sql_query("SELECT cell_id, cell_type FROM cells", conn)
                cell_types = dict(zip(cells_df['cell_id'], cells_df['cell_type']))
                print(f"Loaded cell types for {len(cell_types)} cells from neighbor database")
            except Exception as e:
                print(f"Warning: Could not load cell types from database: {e}")
            finally:
                conn.close()
        except Exception as e:
            print(f"Warning: Could not access neighbor database: {e}")
    
    volume_voxels_array = counts
    volume_um3_array = volume_voxels_array * voxel_volume_um3
    
    cell_only_df = pd.DataFrame({
        'cell_id': unique_cells.astype(int),
        'cell_only_volume_voxels': volume_voxels_array.astype(int),
        'cell_only_volume_um3': volume_um3_array.astype(float),
        'cell_type': [cell_types.get(int(cid), 'unknown') for cid in unique_cells]
    })
    cell_only_df = cell_only_df.sort_values('cell_id').reset_index(drop=True)
    
    print(f"Computed cell-only volumes for {len(cell_only_df)} cells")
    print(f"Total cell-only volume: {cell_only_df['cell_only_volume_um3'].sum():.2f} μm³")
    print(f"Mean cell-only volume: {cell_only_df['cell_only_volume_um3'].mean():.2f} μm³")
    print("=" * 60)
    
    return cell_only_df

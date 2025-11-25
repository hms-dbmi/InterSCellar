# Import

import pandas as pd
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING, Any
from tqdm import tqdm
import time
import os

from ..core.find_cell_neighbors_2d import (
    create_neighbor_edge_table_database_2d,
    find_all_neighbors_by_surface_distance_2d,
    get_cells_dataframe,
    query_cell_type_pairs,
    get_graph_statistics,
    export_to_anndata,
    export_graph_tables,
    build_global_mask_2d,
    build_global_mask_2d_with_mapping
)

from ..core.spatialdata_utils import (
    is_spatialdata,
    extract_polygons_from_spatialdata,
    extract_table_from_spatialdata,
    convert_polygons_to_temp_json,
    add_neighbors_to_spatialdata,
    create_spatialdata_with_table,
    get_pixel_size_from_spatialdata,
    SPATIALDATA_AVAILABLE,
)

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

if TYPE_CHECKING:
    try:
        from spatialdata import SpatialData
    except ImportError:
        SpatialData = Any

# API: Wrapper functions

def find_cell_neighbors_2d(
    polygon_json_path: str,
    metadata_csv_path: str,
    max_distance_um: float = 1.0,
    pixel_size_um: float = 0.1085,
    centroid_prefilter_radius_um: float = 75.0,
    cell_id: str = 'cell_id',
    cell_type: str = 'subclass',
    centroid_x: str = 'X',
    centroid_y: str = 'Y',
    db_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_anndata: Optional[str] = None,
    n_jobs: int = 1,
    return_connection: bool = False,
    save_surfaces_pickle: Optional[str] = None,
    load_surfaces_pickle: Optional[str] = None,
    save_graph_state_pickle: Optional[str] = None,
    return_spatialdata: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[object], Optional[object], Optional['SpatialData']]:
    
    print("=" * 60)
    print("InterSCellar: Surface-based Cell Neighbor Detection - 2D")
    print("=" * 60)
    
    overall_start_time = time.time()
    
    polygon_input_is_spatialdata = is_spatialdata(polygon_json_path)
    metadata_input_is_spatialdata = is_spatialdata(metadata_csv_path)
    input_sdata = polygon_json_path if polygon_input_is_spatialdata else None
    temp_polygon_file = None
    output_sdata = None
    
    if metadata_input_is_spatialdata or (metadata_csv_path is None and polygon_input_is_spatialdata):
        print("\n1. Loading metadata from SpatialData object...")
    else:
        print(f"\n1. Loading metadata from: {metadata_csv_path}...")
    
    step1_start = time.time()
    
    if metadata_input_is_spatialdata:
        try:
            metadata_df = extract_table_from_spatialdata(metadata_csv_path)
            print(f"Loaded {len(metadata_df)} cells from SpatialData table")
        except Exception as e:
            raise ValueError(f"Error extracting metadata from SpatialData: {e}")
    elif metadata_csv_path is None and polygon_input_is_spatialdata:
        try:
            metadata_df = extract_table_from_spatialdata(polygon_json_path)
            print(f"Loaded {len(metadata_df)} cells from SpatialData object")
        except Exception as e:
            raise ValueError(f"Error extracting metadata from SpatialData: {e}")
    else:
        try:
            metadata_df = pd.read_csv(metadata_csv_path)
            print(f"Loaded {len(metadata_df)} cells")
        except Exception as e:
            raise ValueError(f"Error loading metadata CSV: {e}")
    
    required_cols = [cell_id, cell_type, centroid_x, centroid_y]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    step1_time = time.time() - step1_start
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    metadata_dir = "."
    base_name = "cell_metadata"
    if isinstance(metadata_csv_path, str) and metadata_csv_path:
        metadata_dir = os.path.dirname(metadata_csv_path) if os.path.dirname(metadata_csv_path) else "."
        base_name = os.path.splitext(os.path.basename(metadata_csv_path))[0]
    elif polygon_input_is_spatialdata:
        base_name = "spatialdata"
    
    if db_path is None:
        db_path = os.path.join(metadata_dir, f"{base_name}_neighbor_graph_2d.db")
        print(f"db_path: {db_path}")
    
    if output_csv is None:
        output_csv = os.path.join(metadata_dir, f"{base_name}_neighbors_2d.csv")
        print(f"output_csv: {output_csv}")
    
    if output_anndata is None:
        output_anndata = os.path.join(metadata_dir, f"{base_name}_neighbors_2d.h5ad")
        print(f"output_anndata: {output_anndata}")
    
    polygon_file_for_processing = polygon_json_path
    if polygon_input_is_spatialdata:
        if not SPATIALDATA_AVAILABLE:
            raise ImportError(
                "SpatialData support requires the 'spatialdata' package. "
                "Install it with: pip install spatialdata"
            )
        
        print("\n2. Detected SpatialData input. Converting shapes to temporary JSON...")
        step2_start = time.time()
        try:
            polygon_mask = extract_polygons_from_spatialdata(polygon_json_path)
        except Exception as e:
            raise ValueError(f"Error extracting polygons from SpatialData: {e}")
        
        temp_polygon_file = convert_polygons_to_temp_json(polygon_mask)
        polygon_file_for_processing = temp_polygon_file
        step2_time = time.time() - step2_start
        print(f"Temporary polygon JSON created at: {polygon_file_for_processing}")
        print(f"Step 2 completed in {step2_time:.2f} seconds")
        
        if pixel_size_um is None or pixel_size_um == 0.0:
            extracted_pixel = get_pixel_size_from_spatialdata(polygon_json_path)
            if extracted_pixel is None:
                raise ValueError(
                    "pixel_size_um must be provided when SpatialData transformations "
                )
            pixel_size_um = extracted_pixel
    else:
        print(f"\n2. Validating input file: {polygon_json_path}...")
        step2_start = time.time()
        if not os.path.exists(polygon_json_path):
            raise FileNotFoundError(f"JSON file not found: {polygon_json_path}")
        print(f"JSON file found")
        step2_time = time.time() - step2_start
        print(f"Step 2 completed in {step2_time:.2f} seconds")
    
    if pixel_size_um is None or pixel_size_um == 0.0:
        raise ValueError(
            "pixel_size_um must be provided when using polygon JSON input."
        )
    
    print(f"\n3. Building neighbor graph...")
    print(f"Parameters: max_distance={max_distance_um}μm, n_jobs={n_jobs}")
    if max_distance_um == 0.0:
        print(f"Mode: Touching cells only")
    else:
        print(f"Mode: Touching cells + near-neighbors")
    
    step3_start = time.time()
    
    try:
        conn = create_neighbor_edge_table_database_2d(
            polygon_json_path=polygon_file_for_processing,
            metadata_df=metadata_df,
            max_distance_um=max_distance_um,
            pixel_size_um=pixel_size_um,
            centroid_prefilter_radius_um=centroid_prefilter_radius_um,
            db_path=db_path,
            cell_id=cell_id,
            cell_type=cell_type,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            output_csv=output_csv,
            output_anndata=output_anndata,
            n_jobs=n_jobs,
            save_surfaces_pickle=save_surfaces_pickle,
            load_surfaces_pickle=load_surfaces_pickle,
            save_graph_state_pickle=save_graph_state_pickle
        )
        
        step3_time = time.time() - step3_start
        print(f"Neighbor graph created successfully")
        print(f"Step 3 completed in {step3_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error in neighbor detection pipeline: {e}")
    
    print(f"\n4. Retrieving results...")
    step4_start = time.time()
    
    neighbor_table_df = None
    if output_csv:
        try:
            neighbor_table_df = pd.read_sql_query("SELECT * FROM neighbors", conn)
            print(f"Neighbor table: {len(neighbor_table_df)} pairs")
        except Exception as e:
            print(f"Warning: Could not retrieve neighbor table: {e}")
    
    adata = None
    if output_anndata:
        try:
            if ANNDATA_AVAILABLE:
                adata = ad.read_h5ad(output_anndata)
                print(f"AnnData object loaded: {adata.shape}")
            else:
                print(f"Warning: AnnData not available (install with: pip install anndata)")
        except Exception as e:
            print(f"Warning: Could not load AnnData object: {e}")
    
    try:
        stats = get_graph_statistics(conn)
        print(f"Graph statistics: {stats['total_cells']} cells, {stats['total_edges']} pairs")
    except Exception as e:
        print(f"Warning: Could not retrieve statistics: {e}")
    
    step4_time = time.time() - step4_start
    print(f"Step 4 completed in {step4_time:.2f} seconds")
    
    if input_sdata is not None and neighbor_table_df is not None:
        try:
            add_neighbors_to_spatialdata(input_sdata, neighbor_table_df)
            output_sdata = input_sdata
            print("Neighbor results added to SpatialData object")
        except Exception as e:
            print(f"Warning: Could not add results to SpatialData: {e}")
    
    if output_sdata is None and return_spatialdata and neighbor_table_df is not None:
        output_sdata = create_spatialdata_with_table(neighbor_table_df, "neighbors")
        if output_sdata is None:
            print("Warning: SpatialData dependencies missing; cannot create SpatialData output.")
    
    overall_time = time.time() - overall_start_time
    print(f"\n5. Pipeline completed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print(f"Database: {db_path}")
    if output_csv:
        print(f"CSV output: {output_csv}")
    if output_anndata:
        print(f"AnnData output: {output_anndata}")
    
    if temp_polygon_file and os.path.exists(temp_polygon_file):
        try:
            os.unlink(temp_polygon_file)
        except OSError:
            pass
    
    print("=" * 60)
    
    if return_connection:
        base_result = (neighbor_table_df, adata, conn)
    else:
        conn.close()
        base_result = (neighbor_table_df, adata, None)
    
    if return_spatialdata:
        return base_result + (output_sdata,)
    return base_result

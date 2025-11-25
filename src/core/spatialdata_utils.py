# Import

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Any
import json
import tempfile
import os

try:
    from spatialdata import SpatialData
    from spatialdata.models import ShapesModel, LabelsModel, TableModel
    from spatialdata._core.spatialdata import SpatialElement
    SPATIALDATA_AVAILABLE = True
except ImportError:
    SPATIALDATA_AVAILABLE = False
    SpatialData = None
    print("Warning: SpatialData not available. Install with: pip install spatialdata")

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


# Metadata Utilities

def is_spatialdata(obj: Any) -> bool:
    if not SPATIALDATA_AVAILABLE:
        return False
    return isinstance(obj, SpatialData)

def extract_table_from_spatialdata(
    sdata: 'SpatialData',
    table_key: Optional[str] = None
) -> pd.DataFrame:
    if not is_spatialdata(sdata):
        raise TypeError("Input must be a SpatialData object")
    if not sdata.tables:
        raise ValueError("SpatialData object has no tables elements")
    
    if table_key is None:
        table_key = list(sdata.tables.keys())[0]
        print(f"Using table element: '{table_key}'")
    if table_key not in sdata.tables:
        available = list(sdata.tables.keys())
        raise ValueError(f"Table key '{table_key}' not found. Available: {available}")
    table = sdata.tables[table_key]
    
    if ANNDATA_AVAILABLE and isinstance(table, ad.AnnData):
        metadata_df = table.obs.copy()
        if table.obs_names is not None and len(table.obs_names) > 0:
            metadata_df.index = table.obs_names
            if 'cell_id' not in metadata_df.columns:
                metadata_df = metadata_df.reset_index()
                if metadata_df.columns[0] == 'index':
                    metadata_df = metadata_df.rename(columns={'index': 'cell_id'})
    else:
        try:
            metadata_df = pd.DataFrame(table)
        except Exception as e:
            raise ValueError(f"Could not convert table to DataFrame: {e}")
    
    print(f"Extracted metadata from SpatialData table element '{table_key}': {len(metadata_df)} cells")
    return metadata_df

def add_neighbors_to_spatialdata(
    sdata: 'SpatialData',
    neighbor_df: pd.DataFrame,
    table_key: Optional[str] = None,
    neighbor_table_name: str = 'neighbors'
) -> 'SpatialData':
    if not is_spatialdata(sdata):
        raise TypeError("Input must be a SpatialData object")
    
    if ANNDATA_AVAILABLE:
        if 'pair_id' in neighbor_df.columns:
            neighbor_adata = ad.AnnData(obs=neighbor_df.set_index('pair_id'))
        else:
            neighbor_adata = ad.AnnData(obs=neighbor_df)
            neighbor_adata.obs_names = [f"pair_{i}" for i in range(len(neighbor_df))]
        
        sdata.tables[neighbor_table_name] = neighbor_adata
        print(f"Added neighbor pairs to SpatialData as table '{neighbor_table_name}': {len(neighbor_df)} pairs")
    else:
        print(f"Warning: AnnData not available. Neighbor data cannot be added as a proper SpatialData table.")
        print(f"Consider installing anndata: pip install anndata")
    
    return sdata


def add_volumes_to_spatialdata(
    sdata: 'SpatialData',
    volume_df: pd.DataFrame,
    volume_table_name: str = 'interscellar_volumes'
) -> 'SpatialData':
    if not is_spatialdata(sdata):
        raise TypeError("Input must be a SpatialData object")
    
    if ANNDATA_AVAILABLE:
        if 'pair_id' in volume_df.columns:
            volume_adata = ad.AnnData(obs=volume_df.set_index('pair_id'))
        else:
            volume_adata = ad.AnnData(obs=volume_df)
            volume_adata.obs_names = [f"pair_{i}" for i in range(len(volume_df))]
        
        sdata.tables[volume_table_name] = volume_adata
        print(f"Added volume results to SpatialData as table '{volume_table_name}': {len(volume_df)} pairs")
    else:
        print(f"Warning: AnnData not available. Volume data cannot be added as a proper SpatialData table.")
        print(f"Consider installing anndata: pip install anndata")
    
    return sdata


def create_spatialdata_with_table(
    table_df: pd.DataFrame,
    table_name: str
) -> Optional['SpatialData']:
    """
    Build a new SpatialData object that contains a single table.

    Parameters
    ----------
    table_df : pd.DataFrame
        Table data to embed (neighbors, volumes, etc.)
    table_name : str
        Name of the table entry inside SpatialData

    Returns
    -------
    SpatialData or None
        SpatialData instance if dependencies are available, otherwise None.
    """
    if not SPATIALDATA_AVAILABLE or not ANNDATA_AVAILABLE:
        return None
    
    if 'pair_id' in table_df.columns:
        adata = ad.AnnData(obs=table_df.set_index('pair_id'))
    else:
        adata = ad.AnnData(obs=table_df.copy())
        adata.obs_names = [f"pair_{i}" for i in range(len(table_df))]
    
    return SpatialData(tables={table_name: adata})


# 2D Utilities

def extract_polygons_from_spatialdata(
    sdata: 'SpatialData',
    shapes_key: Optional[str] = None
) -> Dict[str, list]:
    if not is_spatialdata(sdata):
        raise TypeError("Input must be a SpatialData object")
    if not sdata.shapes:
        raise ValueError("SpatialData object has no shapes elements")
    
    if shapes_key is None:
        shapes_key = list(sdata.shapes.keys())[0]
        print(f"Using shapes element: '{shapes_key}'")
    if shapes_key not in sdata.shapes:
        available = list(sdata.shapes.keys())
        raise ValueError(f"Shapes key '{shapes_key}' not found. Available: {available}")
    
    shapes = sdata.shapes[shapes_key]
    
    polygon_mask = {}
    
    if hasattr(shapes, 'geometry'):
        try:
            import geopandas as gpd
            is_geodataframe = isinstance(shapes, gpd.GeoDataFrame)
        except ImportError:
            is_geodataframe = False
        
        if is_geodataframe:
            for idx, row in shapes.iterrows():
                geom = row.geometry
                if 'cell_id' in row:
                    cell_id = str(row['cell_id'])
                else:
                    cell_id = str(idx)
                
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    polygon_mask[cell_id] = [[float(x), float(y)] for x, y in coords]
                elif geom.geom_type == 'MultiPolygon':
                    largest = max(geom.geoms, key=lambda p: p.area)
                    coords = list(largest.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    polygon_mask[cell_id] = [[float(x), float(y)] for x, y in coords]
    elif hasattr(shapes, 'values'):
        raise NotImplementedError("Array-based shapes extraction not yet implemented. Please use GeoDataFrame format.")
    else:
        try:
            if hasattr(shapes, 'iterrows'):
                for idx, row in shapes.iterrows():
                    cell_id = str(idx)
                    if 'geometry' in row:
                        geom = row['geometry']
                        if hasattr(geom, 'exterior'):
                            coords = list(geom.exterior.coords)
                            if len(coords) > 1 and coords[0] == coords[-1]:
                                coords = coords[:-1]
                            polygon_mask[cell_id] = [[float(x), float(y)] for x, y in coords]
        except Exception as e:
            raise ValueError(f"Could not extract polygons from shapes element: {e}")
    
    print(f"Extracted {len(polygon_mask)} polygons from SpatialData shapes element '{shapes_key}'")
    return polygon_mask


def convert_polygons_to_temp_json(
    polygon_mask: Dict[str, list],
    temp_dir: Optional[str] = None
) -> str:
    if temp_dir is None:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    else:
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            dir=temp_dir,
            delete=False
        )
    
    json.dump(polygon_mask, temp_file)
    temp_file.close()
    
    return temp_file.name

def get_pixel_size_from_spatialdata(
    sdata: 'SpatialData',
    shapes_key: Optional[str] = None
) -> Optional[float]:
    if not is_spatialdata(sdata):
        return None
    
    if not sdata.shapes:
        return None
    
    if shapes_key is None:
        shapes_key = list(sdata.shapes.keys())[0]
    
    if shapes_key not in sdata.shapes:
        return None
    
    if hasattr(sdata, 'transformations') and shapes_key in sdata.transformations:
        print("Warning: Pixel size extraction from transformations not fully implemented.")
        print("Please provide pixel_size_um explicitly.")
        return None
    
    return None


# 3D Utilities

def extract_labels_from_spatialdata(
    sdata: 'SpatialData',
    labels_key: Optional[str] = None
) -> np.ndarray:
    if not is_spatialdata(sdata):
        raise TypeError("Input must be a SpatialData object")
    if not sdata.labels:
        raise ValueError("SpatialData object has no labels elements")
    
    if labels_key is None:
        labels_key = list(sdata.labels.keys())[0]
        print(f"Using labels element: '{labels_key}'")
    if labels_key not in sdata.labels:
        available = list(sdata.labels.keys())
        raise ValueError(f"Labels key '{labels_key}' not found. Available: {available}")
    
    labels = sdata.labels[labels_key]
    
    if hasattr(labels, 'values'):
        if hasattr(labels.values, 'compute'):
            mask_3d = np.asarray(labels.values.compute())
        else:
            mask_3d = np.asarray(labels.values)
    elif hasattr(labels, 'to_numpy'):
        mask_3d = labels.to_numpy()
    else:
        mask_3d = np.asarray(labels)
    
    print(f"Extracted 3D mask from SpatialData labels element '{labels_key}': shape {mask_3d.shape}, dtype {mask_3d.dtype}")
    return mask_3d


def convert_mask_to_temp_zarr(
    mask_3d: np.ndarray,
    temp_dir: Optional[str] = None
) -> str:
    import zarr
    
    if temp_dir is None:
        temp_zarr_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
        temp_zarr_dir = tempfile.mkdtemp(dir=temp_dir)
    
    temp_zarr_path = os.path.join(temp_zarr_dir, "temp_segmentation.zarr")
    zarr_group = zarr.open(temp_zarr_path, mode='w')
    zarr_group.create_dataset('labels/0/0', data=mask_3d, chunks=(64, 64, 64))
    
    return temp_zarr_path


def get_voxel_size_from_spatialdata(
    sdata: 'SpatialData',
    labels_key: Optional[str] = None
) -> Optional[Tuple[float, float, float]]:
    if not is_spatialdata(sdata):
        return None
    if not sdata.labels:
        return None
    
    if labels_key is None:
        labels_key = list(sdata.labels.keys())[0]
    if labels_key not in sdata.labels:
        return None
    
    labels = sdata.labels[labels_key]
    
    if hasattr(sdata, 'transformations') and labels_key in sdata.transformations:
        print("Warning: Voxel size extraction from transformations not fully implemented.")
        print("Please provide voxel_size_um explicitly.")
        return None
    
    return None

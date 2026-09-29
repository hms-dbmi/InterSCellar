# InterSCellar
[![PyPI](https://img.shields.io/pypi/v/interscellar?logo=pypi&logoColor=blue)](https://pypi.org/project/interscellar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**InterSCellar** is a Python package for surface-Based cell neighborhood and interaction volume analysis in 3D spatial omics.

![Package overview](https://raw.githubusercontent.com/hms-dbmi/InterSCellar/main/docs/images/interscellar-package-overview.png)

## Installation

**Install package:**
```sh
pip install interscellar
```

## Usage

**Import:**
```sh
import interscellar
```

### 3D Pipeline:

**(1) Neighbor Graph: Cell Neighbor Detection & Graph Construction**

```sh
neighbors_3d, adata, conn = interscellar.find_cell_neighbors_3d(
    ome_zarr_path="data/segmentation.zarr",
    metadata_csv_path="data/cell_metadata.csv",
    max_distance_um=0.5,
    voxel_size_um=(0.56, 0.28, 0.28),
    db_path="results/sample_neighbor_graph.db",
    output_csv="results/sample_neighbors_3d.csv",
    n_jobs=8
)
```

**(2) Volume: Interscellar Volume Computation**

**(a) Absolute**: dilates inwards from the cell surface by a fixed, user-defined distance.

```sh
# Interscellar volumes
volumes_3d, adata, conn = interscellar.compute_interscellar_volumes_3d(
    ome_zarr_path="data/segmentation.zarr",
    neighbor_pairs_csv="results/sample_neighbors_3d.csv",
    neighbor_db_path="results/sample_neighbor_graph.db",
    voxel_size_um=(0.56, 0.28, 0.28),
    max_distance_um=3.0,
    intracellular_threshold_um=1.0,
    n_jobs=8
)
```

**(b) Adaptive**: dilates inwards from the cell surface by a depth ratio relative to the cell's maximum reach.

```sh
# Interscellar volumes
volumes_3d = interscellar.compute_interscellar_volumes_3d_adaptive(
    ome_zarr_path="data/segmentation.zarr",
    neighbor_pairs_csv="results/sample_neighbors_3d.csv",
    voxel_size_um=(0.56, 0.28, 0.28),
    max_distance_um=3.0,
    rho_threshold=0.5,
    n_jobs=8
)
```

*Cell-only volumes*
```sh
# Cell segmentation with the interscellar volumes removed
cellonly_3d = interscellar.compute_cell_only_volumes_3d(
    ome_zarr_path="data/segmentation.zarr",
    interscellar_volumes_zarr="results/sample_adaptive_interscellar_volumes.zarr"
)
```

**(3) Score: Biomarker Quantification in Interscellar Volumes**

```sh
# Points (transcriptomics, punctate proteomics): centrality score
scores_3d = interscellar.calculate_interscellar_scores_3d(
    interscellar_volumes_zarr="results/sample_adaptive_interscellar_volumes.zarr",
    spot_zarrs={"GZMB": "data/GZMB_spots.zarr"},
    n_jobs=8
)
```

```sh
# Immunofluorescence (metabolomics, intensity proteomics): intensity statistics
python -m interscellar.core.calculate_interscellar_scores_3d_intensity \
  --segmentation-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --raw-expression-zarr "data/raw_expression.zarr" \
  --n-jobs 8
```

### 2D Pipeline:

**(1) Neighbor Graph: Cell Neighbor Detection & Graph Construction**

```sh
neighbors_2d, adata, conn = interscellar.find_cell_neighbors_2d(
    polygon_json_path="data/cell_polygons.json",
    metadata_csv_path="data/cell_metadata.csv",
    max_distance_um=1.0,
    pixel_size_um=0.1085,
    n_jobs=8
)
```

### Utilities:

**Preprocessing**
```sh
# Find cells truncated at the top/bottom Z-edge of the volume
exclude-truncated \
  --segmentation-zarr "data/segmentation.zarr" \
  --buffer-voxel-min 5 \
  --buffer-voxel-max 5
```

```sh
# Split the segmentation into included vs. excluded cells and view them (Napari)
visualize-excluded-3d \
  --segmentation-zarr "data/segmentation.zarr" \
  --excluded-ids "data/segmentation_z_edge_excluded_cells.txt"
```

```sh
# Remove nuclei from any label volume (interscellar, cell-only, or cell segmentation)
exclude-nuclei \
  --input-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --nuclei-segmentation-zarr "data/nuclei_segmentation.zarr"
```

**Volume Processing**
```sh
# Merge cell-only and interscellar volumes into one label zarr
combine-volumes-3d \
  --cell-only-zarr "results/sample_adaptive_cell_only_volumes.zarr" \
  --interscellar-zarr "results/sample_adaptive_interscellar_volumes.zarr"
```

```sh
# XYZ centroids (um) of each interscellar volume
interscellar-centroids-3d \
  --combined-zarr "results/sample_adaptive_combined_volumes.zarr"
```

```sh
# Cell-only volumes for a single pair (removes only that pair's interscellar volume)
cell-only-pair-3d \
  --cell-segmentation-zarr "data/segmentation.zarr" \
  --interscellar-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --pair-id 123
```

**Feature Extraction**
```sh
# Per-volume expression features
feature-extract-3d \
  --segmentation-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --raw-expression-zarr "data/raw_expression.zarr" \
  --output-csv "results/features_3d.csv"
```

```sh
# Spot counts per volume from a spot coordinate CSV
spot-count-3d \
  --combined-zarr "results/sample_adaptive_combined_volumes.zarr" \
  --metadata-csv "results/volume_metadata.csv" \
  --spots-csv "data/GZMB_spots.csv" \
  --biomarker GZMB
```

**Volume Visualization**
```sh
# Full dataset (Napari)
visualize-all-3d \
  --cell-only-zarr "results/sample_adaptive_cell_only_volumes.zarr" \
  --interscellar-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --cell-only-opacity 0.7 \
  --interscellar-opacity 0.9
```

```sh
# Single pair (Napari)
visualize-pair-3d \
  --pair-id 123 \
  --cell-only-zarr "results/sample_absolute_cell_only_volumes.zarr" \
  --interscellar-zarr "results/sample_absolute_interscellar_volumes.zarr" \
  --pair-opacity 0.6 \
  --cells-opacity 0.7
```

```sh
# Single pair from the adaptive pathway, including voxels shared with other pairs (Napari)
visualize-pair-3d-adaptive \
  --pair-id 123 \
  --interscellar-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --mask "data/segmentation.zarr" \
  --show-overlapping
```

```sh
# Multiple pairs (Napari)
visualize-multi-3d \
  --pair-ids 12,48,103 \
  --cell-only-zarr "results/sample_adaptive_cell_only_volumes.zarr" \
  --interscellar-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
  --cell-only-opacity 0.7 \
  --interscellar-opacity 0.9
```

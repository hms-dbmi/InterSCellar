3D Pipeline Tutorial
====================

This tutorial demonstrates how to use InterSCellar for 3D spatial omics analysis. The 3D pipeline has three steps:

1. **Neighbor Graph**: detect neighboring cells and build a cell neighbor graph.
2. **Volume**: compute the interscellar volume between each pair of neighboring cells, using either the absolute or the adaptive pathway.
3. **Score**: quantify biomarker expression inside each interscellar volume.

Step 1: Neighbor Graph
----------------------

First, detect cell neighbors in 3D space:

.. code-block:: python

   import interscellar

   neighbors_3d, adata, conn = interscellar.find_cell_neighbors_3d(
       ome_zarr_path="data/segmentation.zarr",
       metadata_csv_path="data/cell_metadata.csv",
       max_distance_um=0.5,
       voxel_size_um=(0.56, 0.28, 0.28),
       db_path="results/sample_neighbor_graph.db",
       output_csv="results/sample_neighbors_3d.csv",
       n_jobs=8
   )

Parameters
^^^^^^^^^^

* ``ome_zarr_path``: Path to OME-Zarr file containing 3D segmentation
* ``metadata_csv_path``: Path to CSV file with cell metadata
* ``max_distance_um``: Maximum surface-to-surface distance in micrometers for neighbor detection
* ``voxel_size_um``: Tuple of (z, y, x) voxel sizes in micrometers
* ``db_path``: Output neighbor graph database
* ``output_csv``: Output CSV of neighbor pairs
* ``n_jobs``: Number of parallel jobs

Output
^^^^^^

The function returns:

* ``neighbors_3d``: DataFrame with neighbor pairs
* ``adata``: AnnData object (if available) with graph information
* ``conn``: Database connection object

Step 2: Volume
--------------

After detecting neighbors, compute the interscellar volumes. There are two pathways, which differ in how far each volume reaches into its two cells:

* **(a) Absolute**: dilates inwards from the cell surface by a fixed, user-defined distance.
* **(b) Adaptive**: dilates inwards from the cell surface by a depth ratio relative to the cell's maximum reach.

(a) Absolute
^^^^^^^^^^^^

.. code-block:: python

   volumes_3d, adata, conn = interscellar.compute_interscellar_volumes_3d(
       ome_zarr_path="data/segmentation.zarr",
       neighbor_pairs_csv="results/sample_neighbors_3d.csv",
       neighbor_db_path="results/sample_neighbor_graph.db",
       voxel_size_um=(0.56, 0.28, 0.28),
       max_distance_um=3.0,
       intracellular_threshold_um=1.0,
       n_jobs=8
   )

**Parameters**

* ``ome_zarr_path``: Path to OME-Zarr file containing 3D segmentation
* ``neighbor_pairs_csv``: Path to CSV file with neighbor pairs from step 1
* ``neighbor_db_path``: Path to database file with neighbor pairs from step 1
* ``voxel_size_um``: Tuple of (z, y, x) voxel sizes in micrometers
* ``max_distance_um``: Maximum distance for volume computation
* ``intracellular_threshold_um``: Fixed depth the volume reaches into each cell
* ``n_jobs``: Number of parallel jobs

**Output**

The function returns:

* ``volumes_3d``: DataFrame with interscellar volume measurements
* ``adata``: AnnData object with volume information
* ``conn``: Database connection object

Outputs are written next to the neighbor pairs CSV as ``sample_absolute_volumes.csv``, ``sample_absolute_interscellar_volumes.zarr`` and ``sample_absolute_cell_only_volumes.zarr``.

(b) Adaptive
^^^^^^^^^^^^

.. code-block:: python

   volumes_3d = interscellar.compute_interscellar_volumes_3d_adaptive(
       ome_zarr_path="data/segmentation.zarr",
       neighbor_pairs_csv="results/sample_neighbors_3d.csv",
       voxel_size_um=(0.56, 0.28, 0.28),
       max_distance_um=3.0,
       rho_threshold=0.5,
       n_jobs=8
   )

**Parameters**

* ``ome_zarr_path``: Path to OME-Zarr file containing 3D segmentation
* ``neighbor_pairs_csv``: Path to CSV file with neighbor pairs from step 1
* ``voxel_size_um``: Tuple of (z, y, x) voxel sizes in micrometers
* ``max_distance_um``: Maximum distance for volume computation
* ``rho_threshold``: Depth ratio, relative to each cell's maximum reach, that the volume extends into the cell
* ``n_jobs``: Number of parallel jobs

**Output**

The function returns:

* ``volumes_3d``: DataFrame with interscellar volume measurements

Outputs are written next to the neighbor pairs CSV as ``sample_adaptive_volumes.csv`` and ``sample_adaptive_interscellar_volumes.zarr``. Pairs with no extracellular corridor connecting the two cells are rejected and listed in ``sample_adaptive_rejected_pairs.csv``.

Cell-only volumes
^^^^^^^^^^^^^^^^^

After computing interscellar volumes with either pathway, you can compute cell-only volumes by subtracting the interscellar volumes from the original segmentation:

.. code-block:: python

   cellonly_3d = interscellar.compute_cell_only_volumes_3d(
       ome_zarr_path="data/segmentation.zarr",
       interscellar_volumes_zarr="results/sample_adaptive_interscellar_volumes.zarr"
   )

**Parameters**

* ``ome_zarr_path``: Path to original OME-Zarr file containing 3D segmentation
* ``interscellar_volumes_zarr``: Path to interscellar volumes zarr file from step 2

The output zarr file is automatically saved in the same directory as the interscellar_volumes_zarr.

**Output**

The function returns:

* ``cellonly_3d``: DataFrame with cell-only volume measurements

Step 3: Score
-------------

Finally, quantify biomarker expression inside each interscellar volume.

Points (transcriptomics, punctate proteomics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The centrality score counts each spot inside a volume, weighted by its distance from the pair's contact interface:

.. code-block:: python

   scores_3d = interscellar.calculate_interscellar_scores_3d(
       interscellar_volumes_zarr="results/sample_adaptive_interscellar_volumes.zarr",
       spot_zarrs={"GZMB": "data/GZMB_spots.zarr"},
       n_jobs=8
   )

**Parameters**

* ``interscellar_volumes_zarr``: Interscellar volumes zarr from the adaptive pathway of step 2
* ``spot_zarrs``: Mapping of biomarker name to spot mask zarr on the same voxel grid
* ``n_jobs``: Number of parallel jobs

**Output**

The function returns:

* ``scores_3d``: DataFrame with one row per pair and biomarker, also written as ``sample_adaptive_scores.csv``

Immunofluorescence (metabolomics, intensity proteomics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intensity statistics per volume are computed from a raw expression image:

.. code-block:: bash

   python -m interscellar.core.calculate_interscellar_scores_3d_intensity \
     --segmentation-zarr "results/sample_adaptive_interscellar_volumes.zarr" \
     --raw-expression-zarr "data/raw_expression.zarr" \
     --n-jobs 8

3D Pipeline Tutorial
====================

This tutorial demonstrates how to use InterSCellar for 3D spatial omics analysis.

Step 1: Cell Neighbor Detection & Graph Construction
-----------------------------------------------------

First, detect cell neighbors in 3D space:

.. code-block:: python

   import interscellar

   neighbors_3d, adata, conn = interscellar.find_cell_neighbors_3d(
       ome_zarr_path="data/segmentation.zarr",
       metadata_csv_path="data/cell_metadata.csv",
       max_distance_um=0.5,
       voxel_size_um=(0.56, 0.28, 0.28),
       n_jobs=4
   )

Parameters
----------

* ``ome_zarr_path``: Path to OME-Zarr file containing 3D segmentation
* ``metadata_csv_path``: Path to CSV file with cell metadata
* ``max_distance_um``: Maximum distance in micrometers for neighbor detection
* ``voxel_size_um``: Tuple of (z, y, x) voxel sizes in micrometers
* ``n_jobs``: Number of parallel jobs

Step 2: Interscellar Volume Computation
-----------------------------------------

After detecting neighbors, compute the interscellar volumes:

.. code-block:: python

   volumes_3d, adata, conn = interscellar.compute_interscellar_volumes_3d(
       ome_zarr_path="data/segmentation.zarr",
       neighbor_pairs_csv="results/neighbors_3d.csv",
       voxel_size_um=(0.56, 0.28, 0.28),
       max_distance_um=3.0,
       intracellular_threshold_um=1.0,
       n_jobs=4
   )

Parameters
----------

* ``ome_zarr_path``: Path to OME-Zarr file containing 3D segmentation
* ``neighbor_pairs_csv``: Path to CSV file with neighbor pairs from step 1
* ``voxel_size_um``: Tuple of (z, y, x) voxel sizes in micrometers
* ``max_distance_um``: Maximum distance for volume computation
* ``intracellular_threshold_um``: Threshold for intracellular volume classification
* ``n_jobs``: Number of parallel jobs

Output
------

The function returns:
* ``volumes_3d``: DataFrame with interscellar volume measurements
* ``adata``: AnnData object with volume information
* ``conn``: Database connection object


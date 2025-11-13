2D Pipeline Tutorial
====================

This tutorial demonstrates how to use InterSCellar for 2D spatial omics analysis.

Cell Neighbor Detection & Graph Construction
--------------------------------------------

The 2D pipeline starts with detecting cell neighbors based on surface distance:

.. code-block:: python

   import interscellar

   neighbors_2d, adata, conn = interscellar.find_all_neighbors_2d(
       polygon_json_path="data/cell_polygons.json",
       metadata_csv_path="data/cell_metadata.csv",
       max_distance_um=1.0,
       pixel_size_um=0.1085,
       n_jobs=4
   )

Parameters
----------

* ``polygon_json_path``: Path to JSON file containing cell polygon coordinates
* ``metadata_csv_path``: Path to CSV file with cell metadata
* ``max_distance_um``: Maximum distance in micrometers for neighbor detection
* ``pixel_size_um``: Pixel size in micrometers
* ``n_jobs``: Number of parallel jobs

Output
------

The function returns:
* ``neighbors_2d``: DataFrame with neighbor pairs
* ``adata``: AnnData object (if available) with graph information
* ``conn``: Database connection object


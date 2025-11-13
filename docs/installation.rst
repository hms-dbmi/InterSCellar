Installation
=============

Install package
---------------

InterSCellar can be installed via pip from PyPI:

.. code-block:: bash

   pip install interscellar

Or install from source:

.. code-block:: bash

   git clone https://github.com/euniceyl/InterSCellar.git
   cd InterSCellar
   pip install -e .

Requirements
------------

InterSCellar requires Python 3.8 or higher and the following dependencies:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* scikit-image >= 0.18.0
* opencv-python >= 4.5.0
* zarr >= 2.10.0
* dask[array] >= 2021.0.0
* matplotlib >= 3.4.0
* seaborn >= 0.11.0
* tqdm >= 4.62.0

Optional dependencies:

* anndata >= 0.8.0 (for AnnData integration)
* napari >= 0.4.0 (for visualization)
* duckdb >= 0.6.0 (for database operations)


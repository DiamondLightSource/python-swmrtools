Getting Started
===============

swmr_tools is a python package for making live data processing of hdf5_files
easy.

swmr_tools can be installed from conda-forge using::

    conda install -c conda-forge swmr-tools


It can also be installed from PyPi::

    pip install swmr_tools

Alternatively you can clone the git repository containing swmr_tools using::

    git clone https://github.com/DiamondLightSource/python-swmrtools.git

HDF5 File Requirements
======================

To live process HDF5 data using the swmr_tools package there are a few requirements on the file structure.

 - The file must be created in swmr mode (see https://docs.h5py.org/en/stable/swmr.html)
 - The file must have one (or more) *key* datasets (see below)
 - (Optional) The file can have a *finished* dataset (see below)

Key Datasets
------------
Although swmr allows HDF5 to be read while being written, it can be difficult to determine whether a slice of the data has been written to or is just the fill data HDF5 uses when a dataset is expanded. To determine whether real data is actually written, swmr_tools needs a *key* dataset. The *key* dataset is usually an integer dataset, with a fill value of zero, which is flushed with a non-zero integer value after the corresponding frame of the main dataset is flushed. By monitoring these *key* datasets, swmr_tools can determine when each data frame is readable.

Finished Dataset
----------------

Since HDF5 datasets can be expanded it can be difficult to tell whether a file is complete or whether more data is likely to be written. The swmr_tools library uses a time out to determine when to finish, but this can also be paired with a *finished* dataset. The *finished* dataset is a single integer dataset, with a value zero when the file is still being written to and non-zero when the file is complete. This allows a long time out to be used without wasting time waiting when the file is complete.



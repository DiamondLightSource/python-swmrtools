##########
Basic Use
##########

The DataSource class is the simplest way to interact with a live swmr file. The DataSource is an iterator that provides a map of data for each frame.


The DataSource class requires 3 arguments:

* An instance of an h5py.File object containing the datasets of interest.
* A list of paths to *key* datasets (or groups containing only *key* datasets).
* A list of paths to datasets containing the data you wish to process.

The DataSource also has an optional *timeout* argument, which defaults to 10
second unless otherwise specified, and *finished_dataset* argument, which is the path to the *finished* dataset.

The DataSource works out the dimensions of the frame (whether scalar, vector or image) by looking at the difference between the rank of the key and data datasets. It assumes that the data is written row-major and the data frames are in the fastest dimensions.

Reading Data
------------

As an example we will create two small datasets (of the same size but containing different values) and corresponding unique key dataset to use in our example. This example shows a 2 x 2 grid scan of a detector with shape [5,10]. The keys will all be non-zero so we should expect to receive every frame of the dataset ::

    from swmr_tools import DataSource
    import h5py
    import numpy as np
    
    #Create a small dataset to extract frames from
    data_1 = np.random.randint(low = -10000, high = 10000, size = (2,2,5,10))
    data_2 = np.random.randint(low = -10000, high = 10000, size = (2,2,5,10))
    keys_1 = np.arange(1,5).reshape(2,2,1,1)
    
    #Save data to an hdf5 File
    with h5py.File("example.h5", "w", libver = "latest") as f:
        f.create_group("keys")
        f.create_group("data")
        f["keys"].create_dataset("keys_1", data = keys_1)
        f["data"].create_dataset("data_1", data = data_1)
        f["data"].create_dataset("data_2", data = data_2)
        
Then we simply setup a DataSource pointing at the keys and datasets and let it run::

    with h5py.File("example.h5", "r") as f:
        ds = DataSource(f, ["/keys"],["/data/data_1","/data/data_2"])

        for data_map in ds:
            frame = data_map["/data/data_1"]
            print(data_map.slice_metadata)
            print(str(frame))
            
   (slice(0, 1, None), slice(0, 1, None))
   [[[[ 3980 -3645 -5966  8665   360  1863  7697  -769 -5559 -2142]
      [ 4588 -9254  8550 -1948  1172  -886  5600 -4307 -3488  2684]
      [ 6961 -6236 -4299 -7908  4577  4358 -6297 -8586 -4147 -3344]
      [ 7149 -2261  1190 -6692  -828  4310  5177 -1239  8868 -4319]
      [ 2442  5367 -1959  6815  5524 -2185 -2171 -8405 -2000 -6897]]]]
   (slice(0, 1, None), slice(1, 2, None))
   [[[[-4746  9432  4913 -7990 -7969   508 -4400 -4904   749 -1777]
      [-5639 -6433   214 -9282   951 -9444  3568   147 -3306  3393]
      [-9036 -9871 -9149  3938 -4487  9919  -170  5348  3916   289] 
      [-3024   237  6456  8663  3531  8984 -3129  9678  3566  1306]
      [ 1891 -6206  9541 -4270 -7572 -6388 -1389  7990 -9341  8785]]]]
   (slice(1, 2, None), slice(0, 1, None))
   [[[[ 5964  6778 -1285 -4820  1111  5613 -3506 -2496 -6278  2581]
      [ 5037 -1065 -5667  1903  -311 -3747  1912  8773  1429   459]
      [ 4058  6380 -8450 -6520  7715  2446  8190 -6177 -9543  5414]
      [-6701  -870 -7936 -1994  9943  7053  9467 -5751 -7643  1843]
      [ 5033  4083  4520 -3509  9507  1576  9728 -1245  3678 -9098]]]]
       ...
       
The data (as numpy arrays) can be accessed from the data_map for each point using the dataset path as a key in the map. The slice_metadata attribute on the data_map shows the slice the data was taken from.

The slice_metadata can be used to write processed data into a new hdf5 dataset, and the DataSource class has some convenience methods to help with this.

Writing Data
------------

The DataSource class has two methods to assist with writing processed data back into a hdf5 file::
    
    ds.create_dataset(result_data,file_handle,hdf5_path)

which creates a new hdf5 dataset, with the correct type and shape for the result_data numpy array, and::

    ds.append_data(result_data,slice_metadata,output_dataset)

which adds new result datasets into this hdf5 dataset.

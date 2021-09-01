###########
Advanced Use
###########

The DataSource class is designed to be simple but because of this may not work for every method of processing (for example if for performance reasons you dont want to read every frame, or only want to read a region of each frame).

For more complicated use cases the KeyFollower and FrameReader classes can be used.

KeyFollower
-----------

The KeyFollower is the most fundamental class in swmrtools; it follows the key datasets and reports the highest index for which all the key datasets are non-zero.

As an example we will create a dataset of non-zero integers, respresenting a complete set of
scans all flushed to disk ::
 
    import h5py
    from swmr_tools import KeyFollower
    import numpy as np
    
    #create a sequential array of the numbers 1-8 and reshape them into an array
    # of shape (2,4,1,1)
    complete_key_array = np.arange(8).reshape(2,4,1,1) + 1


We then create an empty hdf5 file, create a group called "keys" and create
a dataset in that group called "key_1" where we will add our array of non-zero
keys ::

    with h5py.File("test_file.h5", "w", libver = "latest") as f:
        f.create_group("keys")
        f["/keys"].create_dataset("key_1", data = complete_key_array)

Next, we shall create an instance of the KeyFollower class and demonstrate a
simple example of its use. At a minimum we must pass the h5py.File object 
we wish to read from and a list containing the paths to the hdf5 groups 
containing our keys.

Shown below is an example of using an instance of KeyFollower within a for loop, 
as you would with any standard iterable object. For this basic example of a 
dataset containing only non-zero values, the loop runs 8 times and stops as 
expected ::

    # using an instance of Follower in a for loop
    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = KeyFollower(f, ["/keys"])
        for key in kf:
            print(key)
    0
    1
    2
    3
    4
    5
    6
    7
            
As with the DataSource, the timeout and finished_dataset path can be set on contruction of the KeyFollower.

Running the KeyFollower should not be computationally expensive, because all of the *key* datasets should be relatively small, allowing the KeyFollower to follow a very rapid scan.

The DataSource class is just a KeyFollower that uses a FrameReader to read a frame from each requested dataset. The FrameReader class can also be used outside the DataSource.

FrameReader
-----------

The FrameReader class is constructed using the path to the dataset to read, the file handle to the hdf5 file open in swmr read mode, and the rank of the scan (1 for a stack of images, 2 for a grid scan etc).

the read_frame(index) method then reads the frame corresponding to the index *i* which can be provided by the KeyFollower.



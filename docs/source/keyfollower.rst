###########
KeyFollower
###########

A Jupyter Notebook version of this tutorial can be found at: 
https://github.com/rparke/Iterator/blob/master/tutorials/KeyFollower.ipynb

The KeyFollower class is designed to facilitate live data processing of
datasets contained within an hdf5 file. It achieves this through the use of
two main classes:

* Follower
* FrameGrabber

This tutorial assumes a basic level of skill using the h5py library.
Specifically, you should be comfortable with using h5py to:

* Open and create hdf5_files
* Navigate files using python dictionary methods: *e.g.* using the get() method
* Create groups and datasets

If you are unfamiliar with how to do any of this we recommend reading the
h5py quick start guide: https://docs.h5py.org/en/stable/quick.html


Follower
========

The Follower class can be used to create instances of a python iterator object.
The Follower is central to everything that swmr_tools does and most other
classes either directly use it or are dependent upon the keys it produces.

Example - Iteration through an all non-zero key dataset
-------------------------------------------------------

We will create a dataset of non-zero integers, respresenting a complete set of
scans all flushed to disk ::
 
    import h5py
    from swmr_tools.KeyFollower import Follower
    import numpy as np
    
    #create a sequential array of the numbers 1-8 and reshape them into an array
    # of shape (2,4,1,1)
    complete_key_array = np.arange(8).reshape(2,4,1,1) + 1


We will create an empty hdf5 file, create a group called "keys" and create
a dataset in that group called "key_1" where we will add our array of non-zero
keys ::

    with h5py.File("test_file.h5", "w", libver = "latest") as f:
        f.create_group("keys")
        f["keys"].create_dataset("key_1", data = complete_key_array)

Next, we shall create an instance of the Follower class and demonstrate a
simple example of its use. At a minimum we must pass the h5py.File object 
we wish to read from and a list containing the paths to the hdf5 groups 
containing our keys.

Shown below is an example of using an instance of Follower within a for loop, 
as you would with any standard iterable object. For this basic example of a 
dataset containing only non-zero values, the loop runs 8 times and stops as 
expected ::

    # using an instance of Follower in a for loop
    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = Follower(f, ["keys"])
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
            

Example - Iteration through a dataset containg zeros
----------------------------------------------------

The key dataset is a form of metadata which (as we will see in 
detail when looking at the FrameGrabber class) represents whether a frame of 
a given dataset is complete and has been flushed to disk.

Non-zero key values represent frames that have been completely written and 
flushed to disk, while values of zero represent a frame that has not. We 
therefore expect the iterator to halt when the next key is zero and either to
wait for it to update to a non-zero value and continue or to stop iteration 
entirely if a termination condition is met.

We will demonstrate a simple example of this below using a timeout method as 
a termination condition. Timeout is the default method used by Follower 
(although others can be set) ::


    with h5py.File("test_file.h5", "r+") as f:
        #set all values in the second row to zero
        f["keys/key_1"][1,:,:,:] = 0

    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = Follower(f, ["keys"], timeout = 1)
        for key in kf:
            print(key)
    0
    1
    2
    3
            
The example above clearly shows that the follower iterates through the first 
row waits for the timeout and then proceeds to halt iteration when the key at
index [1,0] does not change to a non-zero value within the 1 second timeout.

Example - Using other termination methods
-----------------------------------------

The timeout method is the default for halting iteration. Other methods can be
used by passing a list of method names (as strings) as an argument when 
instantiating the Follower ::

    with h5py.File("test_file.h5" "r", swmr_mode = True) as f:
        kf = Follower(f, ["keys"], termination_conditions = ["always_true"])
        for key in kf:
            print(key)
    0
    1
    2
    3
    
As expected, we see the same outcome above as when a timeout was used. What
has happened is that whilever there were non-zero keys the iterator behaved as 
normal. As soon as the next available key was zero the iterator stopped 
straight away (rather than waiting for a timeout).


FrameGrabber
============

Indices produced by instances of the KeyFollower class correspond to frames of
relavent datasets. To understand how the FrameGrabber class works it is important
to understand that instances of Follower do **not** return the value of a key,
they return the index of the key for a flattened version of the array. We will
demonstrate this with an example ::

    
    complete_key_array = np.random.randint(low = 10, high = 20000, size = (2,4))
    with h5py.File("test_file.h5", "w", libver = "latest") as f:
        f.create_group("keys")
        f["keys"].create_dataset("key_1", data = complete_key_array)
        
        #print dataset to demonstrate the non-sequential nature of the keys
        print(f["keys/key_1"][...])
    array([[15083, 15092, 15918, 11475], 
    [10070,  9500, 15115,  8331]])
       
As you can see above the key values are all non-zero, however they are not in
sequential order and many of the values are quite high. When using an instance 
of the KeyFollower to iterate through this we simply recieve an index ::

    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = Follower(f, ["keys"], timeout = 1)
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

If we just want to access the value corresponding to the index we can use
numpys unravel_index() method ::

    with h5py.File("test_file.h5", "r", swmr = True) as f:
        print(f["keys/key_1"][np.unravel_index(6, shape = (2,4))])
    15115

This is fine for extracting a scalar, but does not help when trying to extract
a vector valued frame from a dataset. For this purpose we have created the
FrameGrabber class


Using FrameGrabber to Extract Frames from a key index
-----------------------------------------------------


First, we will create a small dataset with a corresponding key dataset containing
with all values non-zero ::

    complete_key_dataset = np.arange(4).reshape(2,2,1,1) + 1
    complete_data_dataset = np.random.randint(low = 0, high = 1000, size = (2,2,5,10))
    with h5py.File("test_file.h5", "w", libver = "latest") as f:
        f.create_group("keys")
        f.create_group("data")
        f["keys"].create_dataset("key_1", data = complete_key_dataset)
        f["data"].create_dataset("data_1", data = complete_data_dataset)
        
        

FrameGrabber takes two arguments, the full path to the dataset you want to
extract frames from and an open h5py.File object containing the dataset. To 
extract a frame, call the method FrameGrabber.Grabber() with the key index ::

    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = Follower(f, ["keys"], timeout = 1)
        fg = FrameGrabber("data/data_1", f)
        for key in kf:
        
            frame = fg.Grabber(key)
            print(f"Printing frame {key}:")
            print(frame +"\n")
            print(f"Shape of frame: {frame}")
            
    Printing frame 0:
    [[[[913  25 989  89 425 221 634 947 510 616]
       [819  56 268 162 474 543 471 368 948 295]
       [723 453 937 548 473 463 542 230 759 567]
       [517 821 388 941 523 420 564 606 491 985]
       [427 967 845 115 526 812 742 419 411 531]]]]
    Shape: (1, 1, 5, 10)
       
    Printing frame 1: 
    [[[[533 411 801 739 470 908 493 634 137 678]
       [862 382 633 113 952 152 520 937 413 685]
       [414 985  69 161  69  53 453 978 846 953]
       [ 94 346 223 891 499 992 888 846 573 507]
       [139 345 834 396 445 789 361  73 504 500]]]]
    Shape: (1, 1, 5, 10)
       
    Printing frame 2: 
    [[[[492 428 465 627 165 583 558 868 133  64]
       [926 732 564 725 424 144 991 139 114 356]
       [941 653 303 665 768 384 894 239 720 510]
       [663 815 228 888 325 356 293 225 481 700]
       [155 506 906  29 307 589  16 264 616  88]]]]
    Shape: (1, 1, 5, 10)
       
    Printing frame 3:
    [[[[376  22 142 805 266 176 824  85 886 771]
       [403 795 603 528 349 117 384 176 186 324]
       [561 467 322 430 792 977 606 906 833 243]
       [954 466 125 597 959 245 699  36 254 410]
       [943 629 468 131 657 717 734 482 657 895]]]]
    Shape: (1, 1, 5, 10)
       
       
The above example demonstrates the ability of the FrameGrabber class to
return corresponding vector-valued dataset frames of the correct shape. This
lets us do operations frame by frame live as frames are being written. Below
is a simple data reduction example where we return the sum of each frame ::

    with h5py.File("test_file.h5", "r", swmr = True) as f:
        kf = Follower(f, ["keys"], timeout = 1)
        fg = FrameGrabber("data/data_1", f)
        for key in kf:
            current_frame = fg.Grabber(key)
            data_reduced_frame = current_frame.sum()
            data_reduced_frame = data_reduced_frame.reshape((1,1,1,1))
            print(f"Printing frame number {key}")
            print(f"Frame = {data_reduced_frame}\n Shape = {data_reduced_frame.shape}\n")
    
    Printing frame number 0
    Frame = [[[[25616]]]] 
    Shape = (1, 1, 1, 1)
    
    Printing frame number 1
    Frame = [[[[25727]]]]
    Shape = (1, 1, 1, 1)
    
    Printing frame number 2
    Frame = [[[[23705]]]]
    Shape = (1, 1, 1, 1)
    
    Printing frame number 3
    Frame = [[[[28003]]]] 
    Shape = (1, 1, 1, 1)
     

     

        

    







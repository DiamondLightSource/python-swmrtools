##########
DataSource
##########

The DataSource submodule provides an efficient wrapper for the 
KeyFollower.Follower and KeyFollower.FrameGrabber classes. Instances of the
DataSource.DataFollower class are iterators with similar functionality to
instances of the KeyFollower.Follower class, however rather than indices they
simply produce the frames themselves.

DataFollower
============

The DataFollower class requires 3 arguments:

* An instance of an h5py.File object containing the datasets of interest.
* A list of paths to **groups** containing datasets of keys.
* A list of paths to **datasets** containing the data you wish to process.

THe DataFollower also has an optional timeout argument, which defaults to 1
second unless otherwise specified. This works in exactly the same way as the
timeout for the KeyFollower.Follower class.

First we will create two small datasets (of the same size but containing different values)
 and corresponding unique key dataset to use in our example. 
 The keys will all be non-zero so we should expect to recieve
every frame of the dataset ::

    from swmr_tools.KeyFollower import Follower, FrameGrabber
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
        
Firstly we will iterate through the frames just using the classes found in the
KeyFollower submodule. Because we have two datasets, we will need to use two
instances of the FrameGrabber class (one for each dataset) ::

    with h5py.File("example.h5", "r") as f:
        kf = Follower(f, ["keys"], timeout = 1)
        fg = FrameGrabber("data/data_1", f)
        for key in kf:
            frame = fg.Grabber(key)
            print(f"Frame number: {key}")
            print(str(frame) + "\n")
            
    Frame number: 0
    [[[[-8413 -4962 -3426 -6842 -3934 -4392  9312  5963  8307 -9903]
       [-2198 -5086  5499  8634  8331 -6489 -5903 -3303  8240  4940]
       [ 2914 -3116 -9394 -2278  5610   149 -2528 -2241 -2079 -4000]
       [-4618  1229  7165 -9145 -9989  -251  8967  3337 -8502  3152]
       [-4523  4213  7319  2616  5154 -9504  3033 -7215 -5730  9359]]]]
    
    Frame number: 1
    [[[[  377  4001  -131 -9234 -1448 -3293 -3206  7901 -2459 -4136]
       [-8078 -4532  1453 -2026 -9359 -3752 -2411  8719  2899 -8269]
       [ 1743  2874 -9490  6787 -8046  5746  6194  3449 -4307 -3109]
       [ 8508  4287  6991  5490 -1735 -3935  2550 -9582  8134 -5800]
       [ 5859 -5751 -1139 -5811  9798  4419  5066 -7785  9789  3066]]]]
    
    Frame number: 2
    [[[[ 2558 -2139  6940  9709  6259  -927 -7148  4958 -7662 -3837]
       [ -960 -1594  8898  4041 -4059 -8173  3008  5634  5087   889]
       [-3183 -5754  3789 -8662 -8770 -7044  2911  6795 -5413  3436]
       [-2440 -9428 -1148  3154  3652  9267 -2069 -9301 -5041  1342]
       [-8556 -9352  4634  1776 -3079 -2928 -4657  9694 -4980  8641]]]]
    
    Frame number: 3
    [[[[-1441  2707   847 -3451  7348  2484 -6207  -963  8323  5124]
       [-2508  6665  4316  -228  7643  6145  9529  6488 -7155  5790]
       [ 6075  1122  8370  1685  7974  4867 -6738  3799  9999  6302]
       [-3221  9007  9592 -1987  2557  7606  2523 -6574 -5345  5295]
       [ 7097  4039  8969  2935  8069 -4251 -3341 -1560  5056 -1055]]]]
       
    with h5py.File("example.h5", "r") as f:
        kf = Follower(f, ["keys"], timeout = 1)
        fg = FrameGrabber("data/data_2", f)
        for key in kf:
            frame = fg.Grabber(key)
            print(f"Frame number: {key}")
            print(str(frame) + "\n")
            
    Frame number: 0
    [[[[-1832 -3594  -833  2126  3599  3192 -7188  -938  9832  8252]
       [-2638  7400 -6365 -9652 -1601 -6388  1537  7066 -1527  -383]
       [-4078 -7539 -6215 -7609  1492 -1057 -3855 -7770 -4820  8740]
       [ 6744  1452   805 -2627 -3993   166 -1486 -6720  1410  6767]
       [-2299 -9901  5531 -6645 -9352  1918 -8036  9882 -7806  7062]]]]
    
    Frame number: 1
    [[[[-4871 -4023  6640 -6253  2880   455  9829 -8619  3512  4547]
       [-6865 -8752  9776 -3163 -2049 -1606  9361  7776 -3332 -4160]
       [-5794 -1619 -4403  8297  6136 -2485  6868  8087 -8258  8918]
       [ 8565 -1141  3101 -6049  9251 -6366 -8178  7719  -893  7639]
       [ 3809 -8340 -4892 -1101  4215 -1570  2379  1591 -5118  6832]]]]
    
    Frame number: 2
    [[[[-5808  3226  7026  7591 -2887 -9362 -9426 -9935  1342  9211]
       [ 5501  5206  -287  2411 -9397 -2703  1303 -2805  1773  8464]
       [ 3243  4806  7835  2281 -5257  4634 -2574 -7787  1816 -6675]
       [-4581  4558 -4136  6348  9617  9979  -841  2962  4163  2452]
       [ 4331 -8888    36  9899  8622  6178  3079  4917  3395 -1572]]]]
    
    Frame number: 3
    [[[[-9686 -2745  -183 -1081  7607 -3595  7142 -4366  9838 -1823]
       [-4235 -8093 -2057  5847  2782 -2140  6692 -2325   193 -3354]
       [-4588 -3740  2184  8685  2328 -6366  9542 -6778 -8696 -3343]
       [ 4665  6484 -6864  9027  1866  7228 -6731 -6816 -5016 -8029]
       [-4631 -4936 -7307 -2692 -1528 -3214 -7812  2367 -3423 -9516]]]]
       
Use of the DataFollower class eliminates the need for creating multiple FrameGrabber
instances. Like the KeyFollower.Follower class, instances of the DataFollower
class are iterators. Like with the KeyFollower.Follower class, we instantiate 
it with the data containing h5py.File object, and a list of paths to key containing
groups. We also pass a list of paths to datasets we want frames from.

Once we have an instance of the class, we can use it in a for loop as with any
other iterator. At each step of the iteration a list containing the frame for
each dataset is returned. The ordering of the frames is the same as the ordering
of the list of datasets. ::

    with h5py.File("example.h5", "r") as f:
        df = DataFollower(f, ['keys'], ['data/data_1', 'data/data_2'])
        key = 0
        for frames in df:
            print(f"Frame: {key}")
            print(frames)
            print("")
            key += 1
    
    Frame: 0
    [array([[[[-8413, -4962, -3426, -6842, -3934, -4392,  9312,  5963,
               8307, -9903],
             [-2198, -5086,  5499,  8634,  8331, -6489, -5903, -3303,
               8240,  4940],
             [ 2914, -3116, -9394, -2278,  5610,   149, -2528, -2241,
              -2079, -4000],
             [-4618,  1229,  7165, -9145, -9989,  -251,  8967,  3337,
              -8502,  3152],
             [-4523,  4213,  7319,  2616,  5154, -9504,  3033, -7215,
              -5730,  9359]]]]), array([[[[-1832, -3594,  -833,  2126,  3599,  3192, -7188,  -938,
               9832,  8252],
             [-2638,  7400, -6365, -9652, -1601, -6388,  1537,  7066,
              -1527,  -383],
             [-4078, -7539, -6215, -7609,  1492, -1057, -3855, -7770,
              -4820,  8740],
             [ 6744,  1452,   805, -2627, -3993,   166, -1486, -6720,
               1410,  6767],
             [-2299, -9901,  5531, -6645, -9352,  1918, -8036,  9882,
              -7806,  7062]]]])]
    
    Frame: 1
    [array([[[[  377,  4001,  -131, -9234, -1448, -3293, -3206,  7901,
              -2459, -4136],
             [-8078, -4532,  1453, -2026, -9359, -3752, -2411,  8719,
               2899, -8269],
             [ 1743,  2874, -9490,  6787, -8046,  5746,  6194,  3449,
              -4307, -3109],
             [ 8508,  4287,  6991,  5490, -1735, -3935,  2550, -9582,
               8134, -5800],
             [ 5859, -5751, -1139, -5811,  9798,  4419,  5066, -7785,
               9789,  3066]]]]), array([[[[-4871, -4023,  6640, -6253,  2880,   455,  9829, -8619,
               3512,  4547],
             [-6865, -8752,  9776, -3163, -2049, -1606,  9361,  7776,
              -3332, -4160],
             [-5794, -1619, -4403,  8297,  6136, -2485,  6868,  8087,
              -8258,  8918],
             [ 8565, -1141,  3101, -6049,  9251, -6366, -8178,  7719,
               -893,  7639],
             [ 3809, -8340, -4892, -1101,  4215, -1570,  2379,  1591,
              -5118,  6832]]]])]
    
    Frame: 2
    [array([[[[ 2558, -2139,  6940,  9709,  6259,  -927, -7148,  4958,
              -7662, -3837],
             [ -960, -1594,  8898,  4041, -4059, -8173,  3008,  5634,
               5087,   889],
             [-3183, -5754,  3789, -8662, -8770, -7044,  2911,  6795,
              -5413,  3436],
             [-2440, -9428, -1148,  3154,  3652,  9267, -2069, -9301,
              -5041,  1342],
             [-8556, -9352,  4634,  1776, -3079, -2928, -4657,  9694,
              -4980,  8641]]]]), array([[[[-5808,  3226,  7026,  7591, -2887, -9362, -9426, -9935,
               1342,  9211],
             [ 5501,  5206,  -287,  2411, -9397, -2703,  1303, -2805,
               1773,  8464],
             [ 3243,  4806,  7835,  2281, -5257,  4634, -2574, -7787,
               1816, -6675],
             [-4581,  4558, -4136,  6348,  9617,  9979,  -841,  2962,
               4163,  2452],
             [ 4331, -8888,    36,  9899,  8622,  6178,  3079,  4917,
               3395, -1572]]]])]
    
    Frame: 3
    [array([[[[-1441,  2707,   847, -3451,  7348,  2484, -6207,  -963,
               8323,  5124],
             [-2508,  6665,  4316,  -228,  7643,  6145,  9529,  6488,
              -7155,  5790],
             [ 6075,  1122,  8370,  1685,  7974,  4867, -6738,  3799,
               9999,  6302],
             [-3221,  9007,  9592, -1987,  2557,  7606,  2523, -6574,
              -5345,  5295],
             [ 7097,  4039,  8969,  2935,  8069, -4251, -3341, -1560,
               5056, -1055]]]]), array([[[[-9686, -2745,  -183, -1081,  7607, -3595,  7142, -4366,
               9838, -1823],
             [-4235, -8093, -2057,  5847,  2782, -2140,  6692, -2325,
                193, -3354],
             [-4588, -3740,  2184,  8685,  2328, -6366,  9542, -6778,
              -8696, -3343],
             [ 4665,  6484, -6864,  9027,  1866,  7228, -6731, -6816,
              -5016, -8029],
             [-4631, -4936, -7307, -2692, -1528, -3214, -7812,  2367,
              -3423, -9516]]]])]
                
        
    
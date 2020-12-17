#####################
Interfacing With Dask
#####################


Dask is an open source library in python for parallel computing. It has a very
extensive feature set allowing projects built in standard scientific python
libraries to scale for very large datasets. In swmr_tools we currently only
use a small subset of dasks features to help parallelise operations on dataset
frames.

===============================================
Example - Using Dask to Speed up Data Reduction
===============================================

------------------
Sequential Example
------------------
For this example we will create a reasonably large dataset containing random numbers ::

    import h5py
    from swmr_tools.KeyFollower import Follower, FrameGrabber
    import numpy as np
    import time
    complete_keys = np.arange(25).reshape(5,5,1,1) + 1
    complete_dataset = np.random.randint(low = 1, high = 5000, size = (5,5,10,20))
    
    with h5py.File("test.h5", "w", libver = "latest") as f:
        f.create_group('keys')
        f.create_group('data')
        f['keys'].create_dataset("key_1", data = complete_keys)
        f['data'].create_dataset("data_1", data = complete_dataset)

We will next simulate the running of an artficially long calculation ::
    
    
    def long_function(key, filepath = "test.h5", dataset = "data/data_1"):
    time.sleep(1)
    with h5py.File(filepath, "r", swmr = True) as f:
        fg = FrameGrabber(dataset, f)
        frame = fg.Grabber(key)
        return frame.sum()

    def key_generator(queue, filepath = "test.h5"):
        with h5py.File(filepath, "r", swmr = True) as f:
            kf = Follower(f, ['keys'], timeout = 0.1)
            for key in kf:
                queue.put(key)
            queue.put("End")
    

                    
We will run this serial job and time how long it takes to complete ::

    from threading import Thread
    from queue import Queue
    
    def frame_consumer_serial(queue, filepath = "test.h5", dataset = "data/data_1"):
        return_list = []
        key = queue.get()
        while key != 'End':
            return_list.append(long_function(key))
            key = queue.get()
        return return_list
    
    def run_in_serial():
    
        #Create two threads that will read and write keys from a shared queue object
        queue = Queue()
        key_generator_thread = Thread(target = key_generator(queue))
        frame_consumer_serial_thread = Thread(target = frame_consumer_serial, args = (queue,))
    
        #Start timer and start threads running
        start_time = time.time()
        key_generator_thread.start()
        frame_consumer_serial_thread.start()
        
        #Wait for both threads to finish, stop timer and print time taken
        key_generator_thread.join()
        frame_consumer_serial_thread.join()
        finish_time = time.time()
        print(f"Serial time taken = {finish_time - start_time}") 
           
    run_in_serial()
    Serial time taken = 25.042722702026367
        
We will slightly augment the run_in_serial function to run on dask ::

    def frame_consumer_parallel(queue, filepath = "test.h5", dataset = "data/data_1"):
        return_list = []
        client = Client()
        key = queue.get()
        while key != 'End':
            return_list.append(client.submit(long_function, key))
            key = queue.get()
        return client.gather(return_list)
        
    def run_in_parallel_in_dask():
        queue = Queue()
        
        #Create two threads that will read and write keys from a shared queue object
        key_generator_thread = Thread(target = key_generator, args = (queue,))
        frame_consumer_serial_thread = Thread(target = frame_consumer_parallel, args = (queue,))
    
        #Start timer and start threads running
        start_time = time.time()
        key_generator_thread.start()
        frame_consumer_serial_thread.start()
        
        #Wait for both threads to finish, stop timer and print time taken
        key_generator_thread.join()
        frame_consumer_serial_thread.join()
        finish_time = time.time()
        print(f"Serial time taken = {finish_time - start_time}")
        
    run_in_parallel_in_dask()
    Parallel time taken = 5.716917276382446
    
    
    
                    

                
            

----------------------
Job Size and Overheads
----------------------

The action of calling :title: 'client.submit(*args)' carries with it an overhead of 
~1 ms per task. Consequently, for tasks that are already fast (like calling 
np.sum on a reasonably small frame) we either recommend submitting several
frames in a single job or running the job in a serial fashion depending upon
your needs.






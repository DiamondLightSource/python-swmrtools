import numpy as np
import os
import time
    




class Follower():
    '''Iterator for following keys datasets in nexus files
    
    Parameters
    ----------
    hdf5_file: h5py.File
        Instance of h5py.File object. Choose the file containing data you wish
        to follow.
        
    key_datasets: list
        A list of paths (as strings) to groups in hdf5_file containing unique
        key datasets.
        
    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is trigerred and iteration is halted. If a value
        is not set this will default to 10 seconds.
        
    termination_conditions: list (optional)
        A list of strings containing conditions for stopping iteration. Set as
        timeout by default.
        

        
    Examples
    --------
    
    
    >>> # open hdf5 file using context manager with swmr mode activated
    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>> # create an instance of the Follower object to iterate through
    >>>     kf = Follower(f, 
    >>>                   ['path/to/key/group/one', 'path/to/key/group/two'], 
    >>>                   timeout = 1, 
    >>>                   termination_conditions = ['timeout'])
    >>> # iterate through the iterator as with a standard iterator/generator object
    >>>     for key in kf:
    >>>         print(key)
    
    
    
    '''
    
    def __init__(self,
                 hdf5_file,
                 key_datasets,
                 timeout = 10,
                 termination_conditions = ["timeout"]):
                     self.hdf5_file = hdf5_file
                     self.current_key = -1
                     self.current_max = -1
                     self.timeout = timeout
                     self.key_datasets = key_datasets
                     self.termination_conditions = termination_conditions
                     self._finish_tag = False

                         
                         
    def __iter__(self):
        return(self)
    
    def __next__(self):

        self._timer_reset()
        if not self.is_finished():
            while not self._is_next():
                time.sleep(0.2)
                if self.is_finished():
                    self._finish_tag = True
                    raise StopIteration
            
            if self._is_next():
                self.current_key += 1
                x = self.current_key
                self._timer_reset()
                return x
        
        else:
            self._finish_tag = True
            raise StopIteration
    
        
    
    def reset(self):
        '''Reset the iterator to start again from index 0
        '''
        self.current_key = -1
        self.current_max = -1
        self._finish_tag = False
        
    def _timer_reset(self):
        # Hidden method, restarts timer for timeout method
        self.start_time = time.time()
        
    
      
    def _is_next(self):
        # returns true if all the keys for index current_key + 1 are nonzero
        is_next = True
        
        for key_path in self.key_datasets:
            for dataset in self.hdf5_file[key_path].values():
                dataset.refresh()
                #print(dataset.flatten()[self.current_key+1])
                
                try:
                    if dataset[...].flatten()[self.current_key+1] == 0:
                        is_next = False
                    else:
                        pass
                except:
                    is_next = False
        return is_next
                
        
    def _timeout(self):
        if time.time() > self.start_time + self.timeout:
            return True
        else:
            return False
        
        
    def _finish_condition(self):
        conditions = {"timeout": self._timeout(), "always_true" : True}
        
        #Set finish to true if any of the termination conditions are met
        finish = False
        for condition in self.termination_conditions:
            finish = finish or conditions[condition]
        
        
        #return conditions[self.termination_conditions[0]]
        return finish
        
        
    def is_finished(self):
        """ Returns True if the KeyFollower instance has completed its iteration
        
        """
        if (not self._is_next()) and (self._finish_condition()):
            return True
        
        else:
            return False
        
    def _any_next(self):
        any_next = True
        
        for key_path in self.key_datasets:
            for dataset in self.hdf5_file[key_path].values():
                dataset.refresh()
                #print(dataset.flatten()[self.current_key+1])
                
                try:
                    if (dataset[...].flatten()[self.current_key+1:] == 0).all():
                        any_next = False
                    else:
                        pass
                except:
                    any_next = False
        return any_next




    
class FrameGrabber():
    """Class for extracting frames from a dataset given an index generated by
    from an instance of the Follower class
    
    Parameters
    ----------
    hdf5_file : h5py.File
        Instance of h5py.File object. Choose the file containing dataset you 
        want to extract frames from.
        
    dataset : str
        The full path to the dataset to extract frames from in the hdf5_File
        
        
    Examples
    --------
    
    >>> #open and hdf5 file using context manager
    >>> with h5py.File("path/to/file") as f:
    >>>     fg = FrameGrabber(f, "path/to/dataset")
    >>>     #call methods on fg to get the data that you want
    
    """
    def __init__(self, dataset, hdf5_file):
        self.dataset = dataset
        self.hdf5_file = hdf5_file
        
    def Grabber(self, index):
        """ Method for using an index from KeyFollower to extract that frame
        from the chosen hdf5 dataset.
        
        Parameters
        ----------
        index : int
            Index for the correspondng non-zero unique key generated by an
            instance of KeyFollower.Follower
            
            
        Examples
        --------
        
        >>> #create a list of frames from all the keys returned by an instance
        >>> #of KeyFollower.Follower
        >>> frame_list = []
        >>> with h5py.File("path/to/file") as f:
        >>>     fg = KeyFollower.FrameGrabber(f, "path/to/dataset")
        >>>     kf = KeyFollower.Follower(f, ["path/to/keys_1", 
        >>>                          "path/to/keys_2"], 
        >>>                      timeout = 10)
        >>>     for key in kf:
        >>>         frame = fg.Grabber(key)
        >>>         frame_list.append(frame)
        """
        ds =  self.hdf5_file[self.dataset]
        assert len(ds.shape) > 2
        
        self.ds_shape = ds.shape
        self.ds_frame_shape = ds.shape[-2:]
        
        return_shape = [1]*(len(self.ds_shape)-2)
        return_shape.append(self.ds_frame_shape[-2])
        return_shape.append(self.ds_frame_shape[-1])
        frame =  ds[self._get_frame_index(index)]
        frame = frame.reshape(return_shape)
        return frame
    
    
    def _get_frame_index(self, index):
        ds = self.hdf5_file[self.dataset]
        self.ds_shape = ds.shape
        self.ds_frame_shape = ds.shape[-2:]
        ds_frame_index = np.unravel_index(index, self.ds_shape[:-2])
        return ds_frame_index
            
             
    

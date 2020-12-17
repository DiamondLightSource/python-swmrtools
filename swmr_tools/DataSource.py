import numpy as np
from swmr_tools.KeyFollower import Follower, FrameGrabber




class DataFollower():
    """Iterator for returning dataset frames for any number of datasets. This
    class acts as a wrapper for the KeyFollower.Follower and 
    KeyFollower.FrameGrabber classes.
    
    Parameters
    ----------
    
    hdf5_file: h5py.File
        Instance of h5py.File object. Choose the file containing data you wish
        to follow.
        
    keypaths: list
        A list of paths (as strings) to groups in hdf5_file containing unique
        key datasets. (Note: paths must be to the group containing the dataset
                       and not full paths to the dataset itself)
        
    dataset_paths: list
        A list of paths (as strings) to datasets in hdf5_file that you wish
        to return frames from (Note: paths must be to the dataset and not
                               to the group containing it)
        
    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is trigerred and iteration is halted. If a value
        is not set this will default to 10 seconds.
        
        
    Examples
    --------
    
    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>>     df = DataFollower(f, path_to_key_group, path_to_datasets, timeout = 1)
    >>>     for list_of_frames in df:
    >>>         print(list_of_frames)
    
    """
    def __init__(self, hdf5_file, keypaths, dataset_paths, timeout = 1, as_dict = False):
        self.hdf5_file = hdf5_file
        self.dataset_paths = dataset_paths
        self.kf = Follower(hdf5_file, keypaths, timeout)
        self.as_dict = as_dict
        
            
        
    def __iter__(self):
        return(self)
    
    
    def __next__(self):
        
            

        if self.kf.is_finished():
            raise StopIteration
        
        else:
            
            current_dataset_index = next(self.kf)
            
            if not self.as_dict:
                current_dataset_slice = []
                for path in self.dataset_paths:
                    fg = FrameGrabber(path, self.hdf5_file)
                    current_dataset_slice.append(fg.Grabber(current_dataset_index))
            
            else:
                pass
                
            
            #Old method, works but a lot of code
            #current_dataset_slice = self._get_dataset_flattened(current_dataset_index)
            
            #New method, not currently working, gets index using np.unravel_index method
            #current_dataset_slice = self._get_dataset_shaped(current_dataset_index)
            return current_dataset_slice
        
        
    def reset(self):
        '''Reset the iterator to start again from frame 0
        '''
        self.kf.reset()
        
        

                
                
        
            
            



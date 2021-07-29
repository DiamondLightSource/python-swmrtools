import numpy as np
import time


class Follower:
    """Iterator for following keys datasets in nexus files

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



    """

    def __init__(
        self, hdf5_file, key_datasets, timeout=10, finished_dataset=None
    ):
        self.hdf5_file = hdf5_file
        self.current_key = -1
        self.current_max = -1
        self.timeout = timeout
        self.start_time = None
        self.key_datasets = key_datasets
        self.finished_dataset = finished_dataset
        self._finish_tag = False
        self._check_successful = False
        self.scan_rank = -1

    def __iter__(self):
        return self

    def check_datasets(self):
        if self._check_successful:
            return

        rank = -1

        key_list = self._get_key_list()

        for k in key_list:
            #do some exception checking here
            tmp = self.hdf5_file[k]
            r = self._get_rank(tmp.maxshape)

            if rank == -1:
                rank = r

            if rank != -1 and rank != r:
                pass
                #throw exception
        
        self.scan_rank = rank

    def _get_key_list(self):
        key_list = self.key_datasets

        if len(key_list) == 1 and not hasattr(self.hdf5_file[key_list[0]], "shape"):
            k0 = key_list[0]
            ks = []
            for k in self.hdf5_file[k0]:
                ks.append(k0 + "/" + k)
            return ks
        else:
            return key_list

    def _get_rank(self, max_shape):
        rank = len(max_shape)
        rank_cor = 0
        for i in range(len(max_shape)-1,-1,-1):
            if max_shape[i] == 1:
                rank_cor = rank_cor+1
            else:
                break

        return rank - rank_cor

    def __next__(self):

        if self.current_key < self.current_max:
            self.current_key += 1
            return self.current_key

        self._timer_reset()
        while not self._is_next():
            time.sleep(0.2)
            if self.is_finished():
                self._finish_tag = True
                raise StopIteration

        self.current_key += 1
        return self.current_key

    def reset(self):
        """Reset the iterator to start again from index 0"""
        self.current_key = -1
        self._finish_tag = False

    def _timer_reset(self):
        # Hidden method, restarts timer for timeout method
        self.start_time = time.time()

    def _is_next(self):
        # returns true if all the keys for index current_key + 1 are nonzero

        karray = self._get_keys()
        if not karray:
            return False

        if len(karray) == 1:
            merged = karray[0]
        else:
            max_size = max([x.size for x in karray])

            merged = np.zeros((len(karray), max_size))
            first = karray[0]
            merged[0, : first.size] = merged[0, : first.size] + first
            for k in karray[1:]:
                merged[0, : k.size] = merged[0, : k.size] * k

        new_max = np.argmax(merged == 0) - 1
        if new_max < 0:
            new_max = merged.size - 1

        if self.current_max == new_max:
            return False

        self.current_max = new_max
        return True

    def _get_keys(self):
        kds = []
        for key_path in self._get_key_list():
            dataset = self.hdf5_file[key_path]
            dataset.refresh()

            try:
               test = dataset[...]
               kds.append(dataset[...].flatten())
            except Exception:
                return None
        return kds

    def _timeout(self):
        if not self.start_time:
            return False

        if time.time() > self.start_time + self.timeout:
            return True
        else:
            return False

    def _check_finished_dataset(self):
        if self.finished_dataset is None:
            return False

        f = self.hdf5_file[self.finished_dataset]
        f.refresh()
        return f[0] == 1


    def is_finished(self):
        """Returns True if the KeyFollower instance has completed its iteration"""
          
        self._is_next()

        if self.current_max == self.current_key or self._check_finished_dataset():
            return True

        else:
            return False


class FrameReader:
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
    >>>     fg = FrameReader(f, "path/to/dataset")
    >>>     #call methods on fg to get the data that you want

    """

    def __init__(self, dataset, hdf5_file, scan_rank):
        self.dataset = dataset
        self.hdf5_file = hdf5_file
        self.scan_rank = scan_rank

        ds = self.hdf5_file[self.dataset]
        assert len(ds.shape) >= self.scan_rank

    def read_frame(self, index):
        """Method for using an index from KeyFollower to extract that frame
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

        ds = self.hdf5_file[self.dataset]
        shape = ds.shape
        rank = len(shape)
        slices = [slice(0,None,1)] * rank

        if (self.scan_rank == rank):
            shape_slice = slice(0,None,1)
        else:
            shape_slice = slice(0,self.scan_rank,1)

        scan_shape = shape[shape_slice]
        pos = np.unravel_index(index,scan_shape)
        for i in range(len(pos)):
            slices[i] = slice(pos[i], pos[i] + 1)
        frame = ds[tuple(slices)]
        return frame

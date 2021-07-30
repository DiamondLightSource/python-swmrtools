import numpy as np
import time


class KeyFollower:
    """Iterator for following keys datasets in nexus files

    Parameters
    ----------
    h5file: h5py.File
        Instance of h5py.File object. Choose the file containing data you wish
        to follow.

    key_datasets: list
        A list of paths (as strings) to groups in h5file containing unique
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
        self, h5file, key_datasets, timeout=10, finished_dataset=None
    ):
        self.h5file = h5file
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
            tmp = self.h5file[k]
            r = self._get_rank(tmp.maxshape)

            if rank == -1:
                rank = r

            if rank != -1 and rank != r:
                pass
                #throw exception
        
        self.scan_rank = rank

    def _get_key_list(self):
        key_list = self.key_datasets

        if len(key_list) == 1 and not hasattr(self.h5file[key_list[0]], "shape"):
            k0 = key_list[0]
            ks = []
            for k in self.h5file[k0]:
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
            dataset = self.h5file[key_path]
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

        f = self.h5file[self.finished_dataset]
        f.refresh()
        return f[0] == 1


    def is_finished(self):
        """Returns True if the KeyFollower instance has completed its iteration"""
          
        self._is_next()

        if self.current_max == self.current_key or self._check_finished_dataset():
            return True

        else:
            return False


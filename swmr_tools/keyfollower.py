import numpy as np
import time
from .utils import refresh_dataset

import logging

logger = logging.getLogger(__name__)


class KeyFollower:
    """Iterator for following key datasets in hdf5 files

    Parameters
    ----------

    key_datasets: list
        A list of key datasets in the hdf5 file.

    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is trigerred and iteration is halted. If a value
        is not set this will default to 10 seconds.

    finished_dataset: dataset (optional)
        A scalar hdf5 dataset which is zero when the file is being
        written to and non-zero when the file is complete. Used to stop
        the iterator without waiting for the timeout



    Examples
    --------


    >>> # open hdf5 file using context manager with swmr mode activated
    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>>     # create an instance of the Follower object to iterate through
    >>>     keys = [f["key1"], f["key2]]
    >>>     finished = f["finished"]
    >>>     kf = KeyFollower(keys,
    >>>                   timeout = 10,
    >>>                   finished_dataset = finished)
    >>>     # iterate through the iterator as with a standard iterator/generator object
    >>>     for key in kf:
    >>>         print(key)



    """

    def __init__(self, key_datasets, timeout=10, finished_dataset=None):
        self.current_key = -1
        self.current_max = -1
        self.timeout = timeout
        self.timed_out = False
        self.end_time = None
        self.key_datasets = key_datasets
        self.finished_dataset = finished_dataset
        self._finish_tag = False
        self.finished_set = False
        self._check_successful = False
        self.scan_rank = -1
        self.maxshape = None

    def __iter__(self):
        return self

    def check_datasets(self):
        if self._check_successful:
            return

        rank = -1

        for k in self.key_datasets:
            # do some exception checking here
            r = self._get_rank(k.maxshape)

            if rank == -1:
                rank = r

            if rank != -1 and rank != r:
                raise RuntimeError("Key datasets must have the same rank!")

            if self.maxshape is None:
                self.maxshape = k.maxshape[:rank]
            else:
                if np.all(self.maxshape != k.maxshape[:rank]):
                    logger.warning("Max shape not consistent in keys")

        self.scan_rank = rank
        logger.debug("Dataset checks passed")

    def _get_rank(self, max_shape):
        rank = len(max_shape)
        rank_cor = 0
        for i in range(len(max_shape) - 1, -1, -1):
            if max_shape[i] == 1:
                rank_cor = rank_cor + 1
            else:
                break

        return rank - rank_cor

    def __next__(self):
        if self.current_key < self.current_max:
            self.current_key += 1
            return self.current_key

        self._timer_reset()
        while not self._is_next():
            time.sleep(self.timeout / 20.0)
            if self.is_finished():
                self._finish_tag = True
                raise StopIteration

        self.current_key += 1
        return self.current_key

    def reset(self):
        """Reset the iterator to start again from index 0"""
        self.current_key = -1
        self.current_max = -1
        self.timed_out = False
        self._finish_tag = False

    def _timer_reset(self):
        # Hidden method, restarts timer for timeout method
        self.end_time = time.time() + self.timeout

    def _is_next(self):
        karray = self._get_keys()
        if not karray:
            return False

        if len(karray) == 1:
            merged = karray[0]
        else:
            max_size = max([x.size for x in karray])

            merged = np.zeros(max_size)
            first = karray[0]
            merged[: first.size] = merged[: first.size] + first
            for k in karray[1:]:
                padded = np.zeros(max_size)
                padded[: k.size] = k
                merged = merged * padded

        new_max = np.argmax(merged == 0) - 1

        if new_max < 0 and merged[0] != 0:
            # all keys non zero
            new_max = merged.size - 1

        if self.current_max == new_max:
            return False

        self.current_max = new_max
        return True

    def _get_keys(self):
        kds = []
        for k in self.key_datasets:
            refresh_dataset(k)
            d = k[...].flatten()
            kds.append(d)

        return kds

    def _timeout(self):
        if not self.end_time:
            return False

        if time.time() > self.end_time:
            self.timed_out = True
            return True
        else:
            return False

    def _check_finished_dataset(self):
        if self.finished_dataset is None:
            return

        refresh_dataset(self.finished_dataset)

        if self.finished_dataset.size != 1:
            logger.warning(
                f"finished dataset ({self.finished_dataset}) is non-singular"
            )
            return

        # set on a attribute so the timeout loop runs once after
        # finished is set.
        # this is important due to race conditions between the final
        # keys being readable and the finished flag being set
        self.finished_set = not self.finished_dataset[0] == 0

        if self.finished_set:
            logger.debug("Finish flag set after finished dataset check")

    def is_finished(self):
        """Returns True if the KeyFollower instance has completed its iteration"""

        if self.current_key != self.current_max:
            return False

        if self._timeout():
            logger.debug("Finished on timeout")
            return True

        if not self.finished_set:
            self._check_finished_dataset()
            return False

        return self.finished_set

    def get_current_max(self):
        """Returns the current maximum key"""
        return self.current_max

    def refresh(self):
        """Force an update of the current maximum key"""
        return self._is_next()

    def are_keys_complete(self):
        """Check position of current maximum against the max shape"""
        return self.current_max == (np.prod(self.maxshape) - 1)


class RowKeyFollower:
    def __init__(self, key_datasets, timeout=10, finished_dataset=None, row_size=None):
        self.inner_key_follower = KeyFollower(
            key_datasets, timeout=timeout, finished_dataset=finished_dataset
        )
        self.row_size = row_size
        self.scan_rank = -1
        self.maxshape = None
        self._row_count = -1

    def __iter__(self):
        return self

    def check_datasets(self):
        self.inner_key_follower.check_datasets()
        self.scan_rank = self.inner_key_follower.scan_rank
        self.maxshape = self.inner_key_follower.maxshape

        if self.row_size is None:
            rsize = self.inner_key_follower.maxshape[-1]
            if rsize is None:
                raise RuntimeError(
                    "Row size must be defined if fastest max shape dimension is -1"
                )

            self.row_size = rsize

    def __next__(self):
        for i in range(self.row_size - 1):
            next(self.inner_key_follower)

        return next(self.inner_key_follower)

    def reset(self):
        """Reset the iterator to start again from index 0"""
        self._row_count = -1
        self.inner_key_follower.reset()

from .keyfollower import KeyFollower
import numpy as np
import h5py
import time
import logging

logger = logging.getLogger(__name__)

class DataSource:
    """Iterator for returning dataset frames for any number of datasets. This
    class is largely a wrapper around the KeyFollower and FrameReader classes.

    Parameters
    ----------

    h5file: h5py.File
        Instance of h5py.File object. Choose the file containing data you wish
        to follow.

    keypaths: list
        A list of paths (as strings) to key datasets in the hdf5 file. Can also be
        the path to groups, but the groups must contain only key datasets

    dataset_paths: list
        A list of paths (as strings) to datasets in h5file that you wish
        to return frames from (Note: paths must be to the dataset and not
                               to the group containing it)

    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is triggered and iteration is halted. If a value
        is not set this will default to 10 seconds.

    finished_dataset: string (optional)
        Path to a scalar hdf5 dataset which is zero when the file is being
        written to and non-zero when the file is complete. Used to stop
        the iterator without waiting for the timeout

    Examples
    --------

    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>>     df = DataSource(f, path_to_key_group, path_to_datasets)
    >>>     for frame_dict in df:
    >>>         print(frame_dict)

    """

    def __init__(
        self, h5file, keypaths, dataset_paths, timeout=10, finished_dataset=None
    ):
        self.h5file = h5file
        self.dataset_paths = dataset_paths
        self.kf = KeyFollower(h5file, keypaths, timeout, finished_dataset)
        self.kf.check_datasets()

    def __iter__(self):
        return self

    def __next__(self):

        if self.kf.is_finished():
            raise StopIteration

        else:

            current_dataset_index = next(self.kf)

            output = SliceDict()

            for path in self.dataset_paths:
                fg = FrameReader(path, self.h5file, self.kf.scan_rank)
                fd = fg.read_frame(current_dataset_index)
                output[path] = fd[0]
                if output.slice_metadata is None:
                    output.slice_metadata = fd[1]
                    output.maxshape = self.kf.maxshape
                    output.index = current_dataset_index

            return output

    def reset(self):
        """Reset the iterator to start again from frame 0"""
        self.kf.reset()

    def create_dataset(self, data, fh, path):

        scan_max = self.kf.maxshape
        maxshape = scan_max + data.shape
        shape = ([1] * len(scan_max) + list(data.shape))
        r = data.reshape(shape)
        return fh.create_dataset(path, data=r, maxshape = maxshape)

    def append_data(self, data, slice_metadata, dataset):
        ds = tuple(slice(0,s,1) for s in data.shape)
        fullslice = slice_metadata + ds
        current = dataset.shape
        new_shape = tuple(max(s.stop,c) for (s,c) in zip(fullslice,current))
        if (np.any(new_shape > current)):
            dataset.resize(new_shape)
        dataset[fullslice] = data

    @staticmethod
    def check_file_readable(path, datasets,timeout = 10, retrys = 5):
        start = time.time()
        dif = time.time() - start

        while dif < timeout:
            try:
                with h5py.File(path,'r',libver = "latest", swmr = True) as fh:
                    for d in datasets:
                        tmp = fh[d]
                    return True

            except Exception as e:
                logger.debug("Reading failed, retrying " + str(e))
                time.sleep(timeout/retrys)
                dif = time.time() - start

        logger.error("Could not read file " + path)
        return False




class SliceDict(dict):
    """Dictionary with attributes for the slice metadata and maxshape of the scan
    """

    def __init__(self, *args, **kw):
        super(SliceDict, self).__init__(*args, **kw)
        self.slice_metadata = None
        self.maxshape = None
        self.index = None


class FrameReader:
    """Class for extracting frames from a dataset given an index generated by
    from an instance of the KeyFollower class

    Parameters
    ----------
    h5file : h5py.File
        Instance of h5py.File object. Choose the file containing dataset you
        want to extract frames from.

    dataset : str
        The full path to the dataset to extract frames from in the hdf5_File

    scan_rank: int
        The rank of the "non-data-frame" part of the N-dimensional dataset
        for example 1 if the dataset is rank 3 and the frames are images (2D) or
        2 if the dataset is rank 3 and the frames are spectra/patterns/vectors
        (i.e. a grid scan). The KeyFollower class has the scan_rank as an attribute
        that can be used here.

    Examples
    --------

    >>> #open and hdf5 file using context manager
    >>> with h5py.File("/path/to/file") as f:
    >>>     fr = FrameReader(f, "/path/to/dataset", 2)
    >>>     #call methods on fg to get the data that you want

    """

    def __init__(self, dataset, h5file, scan_rank):
        self.dataset = dataset
        self.h5file = h5file
        self.scan_rank = scan_rank

        ds = self.h5file[self.dataset]
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

        ds = self.h5file[self.dataset]
        shape = ds.shape

        try:
            #might fail if dataset is cached
            pos, slices, shape_slice = self.get_pos(index, shape)
        except ValueError:
            #refresh dataset and try again
            if hasattr(ds, "refresh"):
                ds.refresh()

            shape = ds.shape
            pos, slices, shape_slice = self.get_pos(index, shape)

        for i in range(len(pos)):
            slices[i] = slice(pos[i], pos[i] + 1)
        frame = ds[tuple(slices)]
        return frame, tuple(slices[shape_slice])

    def get_pos(self, index, shape):

        rank = len(shape)
        slices = [slice(0, None, 1)] * rank

        if self.scan_rank == rank:
            shape_slice = slice(0, None, 1)
        else:
            shape_slice = slice(0, self.scan_rank, 1)

        scan_shape = shape[shape_slice]
        pos = np.unravel_index(index, scan_shape)

        return pos, slices, shape_slice


from .keyfollower import KeyFollower
import numpy as np

class DataSource:
    """Iterator for returning dataset frames for any number of datasets. This
    class acts as a wrapper for the KeyFollower.Follower and
    KeyFollower.FrameReader classes.

    Parameters
    ----------

    h5file: h5py.File
        Instance of h5py.File object. Choose the file containing data you wish
        to follow.

    keypaths: list
        A list of paths (as strings) to groups in h5file containing unique
        key datasets. (Note: paths must be to the group containing the dataset
                       and not full paths to the dataset itself)

    dataset_paths: list
        A list of paths (as strings) to datasets in h5file that you wish
        to return frames from (Note: paths must be to the dataset and not
                               to the group containing it)

    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is trigerred and iteration is halted. If a value
        is not set this will default to 10 seconds.


    Examples
    --------

    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>>     df = DataSource(f, path_to_key_group, path_to_datasets, timeout = 1)
    >>>     for frame_dict in df:
    >>>         print(frame_dict)

    """

    def __init__(self, h5file, keypaths, dataset_paths, timeout=1):
        self.h5file = h5file
        self.dataset_paths = dataset_paths
        self.kf = KeyFollower(h5file, keypaths, timeout)
        self.kf.check_datasets()

    def __iter__(self):
        return self

    def __next__(self):

        if self.kf.is_finished():
            raise StopIteration

        else:

            current_dataset_index = next(self.kf)

            output = {}

            for path in self.dataset_paths:
                fg = FrameReader(path, self.h5file, self.kf.scan_rank)
                output[path] = fg.read_frame(current_dataset_index)

            return output

    def reset(self):
        """Reset the iterator to start again from frame 0"""
        self.kf.reset()

class FrameReader:
    """Class for extracting frames from a dataset given an index generated by
    from an instance of the Follower class

    Parameters
    ----------
    h5file : h5py.File
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

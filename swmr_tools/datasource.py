from .keyfollower import KeyFollower
import logging
import numpy as np
from .utils import get_position, create_dataset, append_data, refresh_dataset
import sys
from time import sleep

try:
    import blosc
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DataSource:
    """Iterator for returning dataset frames for any number of datasets. This
    class is largely a wrapper around the KeyFollower and FrameReader classes.

    Parameters
    ----------

    key_datasets: list
        A list of key datasets from the hdf5 file.

    datasets: dict
        A dictionary of paths (as strings) to datasets that you wish
        to return frames from.

    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is triggered and iteration is halted. If a value
        is not set this will default to 10 seconds.

    finished_dataset: dataset (optional)
        Path to a scalar hdf5 dataset which is zero when the file is being
        written to and non-zero when the file is complete. Used to stop
        the iterator without waiting for the timeout.


    use_direct_chunk: bool (optional)
        If dataset chunking is aligned to a single frame, and data is blosc
        compressed, will use direct chunk read and compression outside of h5py
        for performance.

    interleaved_paths: dict (optional)
        A dictionary of string to lists of datasets. Where frames are written by multiple file
        writers in an interleaved fashion, this will take frames alternating from each
        dataset in the list. Key dataset should be a VDS of interleaved keys. For use where
        direct chunk read is required but not possible through a VDS stack of frames. Only
        for stacks of frames.


    Examples
    --------

    >>> with h5py.File("/home/documents/work/data/example.h5", "r", swmr = True) as f:
    >>>     keys = [f["key"]]
    >>>     data = {"data" : f["data"]}
    >>>     df = DataSource([keys], data, path_to_datasets)
    >>>     for frame_dict in df:
    >>>         print(frame_dict)

    """

    def __init__(
        self,
        key_datasets,
        datasets,
        timeout=10,
        finished_dataset=None,
        use_direct_chunk=False,
        interleaved_datasets=None,
    ):
        self._datasets = datasets
        self._interleaved_datasets = interleaved_datasets
        self.max_index = -1
        self.frame_readers = {}
        self.interleaved_frame_readers = {}
        self.kf = KeyFollower(key_datasets, timeout, finished_dataset)
        self.kf.check_datasets()

        if datasets is None and interleaved_datasets is None:
            raise RuntimeError("No data specified to follow!")

        self._add_datasets_to_cache(use_direct_chunk)
        self._add_interleaved_datasets_to_cache(use_direct_chunk)

    def _add_datasets_to_cache(self, use_direct_chunk):
        if self._datasets is not None:
            for path, data in self._datasets.items():
                self.frame_readers[path] = FrameReader(
                    data,
                    self.kf.scan_rank,
                    use_direct_chunk=use_direct_chunk,
                )

    def _add_interleaved_datasets_to_cache(self, use_direct_chunk):
        if self._interleaved_datasets is not None:
            for path, data_list in self._interleaved_datasets.items():
                self.interleaved_frame_readers[path] = []
                for data in data_list:
                    fr = FrameReader(
                        data,
                        self.kf.scan_rank,
                        use_direct_chunk=use_direct_chunk,
                    )

                    self.interleaved_frame_readers[path].append(fr)

    def __iter__(self):
        return self

    def __next__(self):
        current_dataset_index = next(self.kf)
        force_refresh = False
        if self.max_index < current_dataset_index:
            self.max_index = self.kf.current_max
            force_refresh = True

        output = SliceDict()

        self._add_datasets_to_output(current_dataset_index, output, force_refresh)
        self._add_interleaved_datasets_to_output(
            current_dataset_index, output, force_refresh
        )

        return output

    def _add_datasets_to_output(self, current_dataset_index, output, force_refresh):
        if self._datasets is None:
            return

        for path in self._datasets.keys():
            fg = self.frame_readers[path]
            frame, slice_metadata = fg.read_frame(
                current_dataset_index, force_refresh=force_refresh
            )
            output[path] = frame
            if output.slice_metadata is None:
                output.slice_metadata = slice_metadata
                output.maxshape = self.kf.maxshape
                output.index = current_dataset_index

    def _add_interleaved_datasets_to_output(
        self, current_dataset_index, output, force_refresh
    ):
        if self._interleaved_datasets is None:
            return

        for path, frs in self.interleaved_frame_readers.items():
            n_frs = len(frs)
            fr_index = current_dataset_index % (n_frs)

            frame, slice_metadata = frs[fr_index].read_frame(
                current_dataset_index // n_frs, force_refresh=force_refresh
            )
            output[path] = frame

            if output.slice_metadata is None:
                updated = (
                    slice(current_dataset_index, current_dataset_index + 1),
                    *slice_metadata[1:],
                )
                output.slice_metadata = updated
                output.maxshape = self.kf.maxshape
                output.index = current_dataset_index

    def reset(self):
        """Reset the iterator to start again from frame 0"""
        self.kf.reset()
        self.max_index = -1

    def create_dataset(self, data, fh, path):
        scan_max = self.kf.maxshape
        return create_dataset(data, scan_max, fh, path)

    def append_data(self, data, slice_metadata, dataset):
        return append_data(data, slice_metadata, dataset)

    def is_scan_finished(self):
        return self.kf.finished_set

    def has_timed_out(self):
        return self.kf.timed_out


class SliceDict(dict):
    """Dictionary with attributes for the slice metadata and maxshape of the scan"""

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
    dataset : dataset
        The dataset to extract frames from in the hdf5_File

    scan_rank: int
        The rank of the "non-data-frame" part of the N-dimensional dataset
        for example 1 if the dataset is rank 3 and the frames are images (2D) or
        2 if the dataset is rank 3 and the frames are spectra/patterns/vectors
        (i.e. a grid scan). The KeyFollower class has the scan_rank as an attribute
        that can be used here.

    use_direct_chunk: bool (optional)
        If dataset chunking is aligned to a single frame, and data is blosc
        compressed, will use direct chunk read and compression outside of h5py
        for performance.

    Examples
    --------

    >>> #open and hdf5 file using context manager
    >>> with h5py.File("/path/to/file") as f:
    >>>     data = f["data"]
    >>>     fr = FrameReader(data, 2)
    >>>     #call methods on fg to get the data that you want

    """

    def __init__(self, dataset, scan_rank, use_direct_chunk=False):
        self.dataset = dataset
        self.scan_rank = scan_rank
        self.use_direct_chunk = use_direct_chunk

        if use_direct_chunk:
            self.use_direct_chunk = False
            prop_dcid = self.dataset.id.get_create_plist()
            if prop_dcid.get_nfilters() == 1 and prop_dcid.get_filter(0)[0] == 32001:
                chunk = prop_dcid.get_chunk()
                shape = self.dataset.shape
                frame_rank = len(shape) - scan_rank
                if shape[-frame_rank:] == chunk[-frame_rank:] and all(
                    [i == 1 for i in chunk[:scan_rank]]
                ):
                    if "blosc" in sys.modules:
                        self.use_direct_chunk = True
                        self.chunk = chunk

    def read_frame(self, index, force_refresh=False):
        """Method for using an index from KeyFollower to extract that frame
        from the chosen hdf5 dataset.

        Parameters
        ----------
        index : int
            Index for the correspondng non-zero unique key generated by an
            instance of KeyFollower.Follower

        force_refresh: bool (optional)
        Forces refresh to be called on the dataset before the frame is read


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

        ds = self.dataset
        shape = ds.shape

        if force_refresh:
            refresh_dataset(ds)

        try:
            # might fail if dataset is cached
            pos = self.get_pos(index, shape)
        except ValueError:
            # refresh dataset and try again
            sleep(1)
            refresh_dataset(ds)

            shape = ds.shape
            pos = self.get_pos(index, shape)

        rank = len(shape)

        slices = [slice(0, None, 1)] * rank

        for i in range(len(pos)):
            slices[i] = slice(pos[i], pos[i] + 1)

        if self.use_direct_chunk:
            return self.get_frame_direct(ds, pos, rank, slices)
        else:
            return self.get_frame(ds, slices)

    def get_frame(self, ds, slices):
        frame = ds[tuple(slices)]
        return frame, tuple(slices[: self.scan_rank])

    def get_frame_direct(self, ds, pos, rank, slices):
        chunk_pos = [0] * rank
        for i in range(len(pos)):
            chunk_pos[i] = pos[i]

        try:
            out = ds.id.read_direct_chunk(chunk_pos)
        except Exception:
            # let the file system catch up
            sleep(1)
            refresh_dataset(ds)
            out = ds.id.read_direct_chunk(chunk_pos)

        decom = blosc.decompress(out[1])
        a = np.frombuffer(decom, dtype=ds.dtype, count=-1)
        return a.reshape(self.chunk), tuple(slices[: self.scan_rank])

    def get_pos(self, index, shape):
        return get_position(index, shape, self.scan_rank)

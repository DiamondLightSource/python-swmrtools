import numpy as np
import h5py
import logging
import time

logger = logging.getLogger(__name__)


def get_position(index, shape, scan_rank):
    """
    Returns the position in the scan associated with the given index

            Parameters:
                    index (int): Flattened index of scan point
                    shape (array): Shape of dataset of interest
                    scan_rank (int): Rank of scan (must be <= len(shape))

            Returns:
                    position (tuple): Ndimensional position in scan associated with the index
    Examples
    --------

    >>> utils.get_position(0, [3,4,5], 2)
    (0,0)

    """

    scan_shape = shape[:scan_rank]
    pos = np.unravel_index(index, scan_shape)

    return pos


def get_position_snake(index, shape, scan_rank):
    """
    Returns the position in a snake scan associated with the given index

            Parameters:
                    index (int): Flattened index of scan point
                    shape (array): Shape of dataset of interest
                    scan_rank (int): Rank of scan (must be <= len(shape))

            Returns:
                    position (tuple): Ndimensional position in scan associated with the index
    Examples
    --------

    >>> utils.get_position_snake(0, [3,4,5], 2)
    (0,0)

    """
    non_snake = get_position(index, shape, scan_rank)
    out = [*non_snake]
    sum = non_snake[0]
    for i in range(1, len(out)):
        if sum % 2 == 1:
            out[i] = shape[i] - non_snake[i] - 1

        sum += out[i]

    return tuple(out)


def get_row_slice(index, shape, scan_rank):
    """
    Returns a slice tuple corresponding to the row of the shape that contains the index

            Parameters:
                    index (int): Flattened index of scan point
                    shape (array): Shape of dataset of interest
                    scan_rank (int): Rank of scan (must be <= len(shape))

            Returns:
                    row_slice (tuple): tuple of slices (same length as scan_rank) corresponding to the row containing the index
    Examples
    --------


    >>> utils.get_row_slice(3, [3,4,5], 2)
    (slice(0,1,None),slice(None,None,None)

    """
    pos = get_position(index, shape, scan_rank)

    rank = len(shape)
    slices = [slice(0, None, 1)] * rank

    for i in range(len(pos)):
        slices[i] = slice(pos[i], pos[i] + 1)

    slices[scan_rank - 1] = slice(None)

    return tuple(slices[:scan_rank])


def create_dataset(data, scan_maxshape, fh, path, **kwargs):
    """
    Convenience method to create a hdf5 dataset corresponding to data being the first dataset in a scan with shape scan_maxshape

            Parameters:
                    data (numpy array): First dataset to save in hdf5, corresponding to the first point in the scan
                    scan_maxshape (array): Shape of the scan, for example [1000] for a stack or [100,100] for a grid
                    fh (h5py File or Group): File or Group to create dataset in
                    path (str): path to save the dataset under
                    kwargs: forwared to the h5py create dataset method allowing chunking and compression to be specified

            Returns:
                    dataset (h5py Dataset): the result dataset

    """

    maxshape = scan_maxshape + data.shape
    shape = [1] * len(scan_maxshape) + list(data.shape)
    r = data.reshape(shape)

    # if chunks not set use data shape
    if "chunks" not in kwargs:

        if data.size < 10:
            c = [1 if i is None else i for i in maxshape]
            kwargs["chunks"] = tuple(c)
        else:
            kwargs["chunks"] = tuple(shape)

    return fh.create_dataset(path, data=r, maxshape=maxshape, **kwargs)


def append_data(data, slice_metadata, dataset):
    """
    Convenience method to append data to a hdf5 dataset generated in create_dataset

            Parameters:
                    data (numpy array): dataset to save in hdf5, corresponding to the n-th point in the scan
                    slice_metadata (array): Slice describing current position in scan
                    dataset (h5py Dataset): Dataset to set slice in

    """
    ds = tuple(slice(0, s, 1) for s in data.shape)
    fullslice = slice_metadata + ds
    current = dataset.shape
    new_shape = tuple(max(s.stop, c) for (s, c) in zip(fullslice, current))
    if np.any(new_shape > current):
        dataset.resize(new_shape)
    dataset[fullslice] = data


def copy_nexus_axes(nxd_in, nxd_out, scan_rank, frame_axes=None):
    """
    Copy the axes and associated attributes from the input NXdata to the output NXdata

            Parameters:
                    nxd_in (h5py Group): input NXdata to copy from
                    nxd_out (h5py Group): output NXdata to copy to
                    scan_rank (int): number of scan dimensions
                    frame_axes(String array): additional axes for data frame

    """
    axout = ["."] * scan_rank
    if frame_axes is not None:
        axout = axout + frame_axes

    if "axes" in nxd_in.attrs:
        axin = nxd_in.attrs["axes"]
        for i in range(scan_rank):
            a = axin[i]
            ax = a.decode("utf-8") if hasattr(a, "decode") else a
            axout[i] = ax
            if ax in nxd_in:
                nxd_out.copy(nxd_in[ax], nxd_out, ax)
                at = ax + "_indices"
                if at in nxd_in.attrs:
                    nxd_out.attrs[at] = nxd_in.attrs[at]

    nxd_out.attrs["axes"] = axout


def create_nxdata(e, name, default=True):
    """
    Create NXdata group
            Parameters:
                    e (h5py Group): group to create NXdata in
                    name (str): Name of NXdata group
                    default (boolean): add default tag to e pointing at new NXdata

    """
    return _create_nexus_with_default(e, name, "NXdata", default=default)


def create_nxentry(fh, name, default=True):
    """
    Create NXentry group
            Parameters:
                    fh (h5py Group): group to create NXentry in
                    name (str): Name of NXentry group
                    default (boolean): add default tag to parent pointing at new NXentry

    """
    return _create_nexus_with_default(fh, name, "NXentry", default=default)


def _create_nexus_with_default(g, name, nxclass, default=True):
    d = g.create_group(name)
    d.attrs["NX_class"] = nxclass
    if default:
        g.attrs["default"] = name

    return d


def check_file_readable(path, datasets, timeout=10, retrys=5):
    """
    Check all datasets in a file can be read, used to determine if all files are readable if main file contains links

            Parameters:
                    path (str): path to hdf5 file being checked
                    datasets (array): List of paths to datasets in the file

    """
    start = time.time()
    dif = time.time() - start

    while dif < timeout:
        try:
            with h5py.File(path, "r", libver="latest", swmr=True) as fh:
                for d in datasets:
                    fh[d]
                return True

        except Exception as e:
            logger.debug("Reading failed, retrying " + str(e))
            time.sleep(timeout / retrys)
            dif = time.time() - start

    logger.error("Could not read file " + path)
    return False


def convert_stack_to_grid(slices, scan_shape, snake=False):

    index = slices[0].start

    if not snake:
        pos = get_position(index, scan_shape, len(scan_shape))
    else:
        pos = get_position_snake(index, scan_shape, len(scan_shape))

    return _get_slices_from_position(pos, slices)


def _get_slices_from_position(pos, slices):
    slices_out = [slice(p, p + 1, 1) for p in pos]
    slices_out += slices[1:]
    return slices_out

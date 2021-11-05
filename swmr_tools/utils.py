import numpy as np
import h5py
import logging
import time

logger = logging.getLogger(__name__)

def get_position(index, shape, scan_rank):

    rank = len(shape)
    slices = [slice(0, None, 1)] * rank

    if scan_rank == rank:
        shape_slice = slice(0, None, 1)
    else:
        shape_slice = slice(0, scan_rank, 1)

    scan_shape = shape[shape_slice]
    pos = np.unravel_index(index, scan_shape)

    return pos, slices, shape_slice

def get_row_slice(index, shape, scan_rank):

    pos, slices, shape_slice = get_position(index, shape, scan_rank)

    for i in range(len(pos)):
        slices[i] = slice(pos[i], pos[i] + 1)

    slices[scan_rank-1] = slice(None)

    return tuple(slices[shape_slice])

def create_dataset(data, scan_maxshape, fh, path, **kwargs):
    maxshape = scan_maxshape + data.shape
    shape = [1] * len(scan_maxshape) + list(data.shape)
    r = data.reshape(shape)

    #if chunks not set use data shape
    if "chunks" not in kwargs:

        if data.size < 10:
            c = [1 if i is None else i for i in maxshape]
            print(tuple(c))
            kwargs["chunks"] = tuple(c)
        else:
            print(tuple(shape))
            kwargs["chunks"] = tuple(shape)

    return fh.create_dataset(path, data=r, maxshape=maxshape, **kwargs)

def append_data(data, slice_metadata, dataset):
    ds = tuple(slice(0, s, 1) for s in data.shape)
    fullslice = slice_metadata + ds
    current = dataset.shape
    new_shape = tuple(max(s.stop, c) for (s, c) in zip(fullslice, current))
    if np.any(new_shape > current):
        dataset.resize(new_shape)
    dataset[fullslice] = data


def check_file_readable(path, datasets, timeout=10, retrys=5):
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

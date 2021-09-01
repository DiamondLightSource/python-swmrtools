import numpy as np
from swmr_tools import DataSource
import utils


def test_iterates_complete_dataset():

    mds = utils.make_mock([10])
    mdsc = utils.make_mock([10])
    mds.dataset[...] = 1
    mdsc.dataset[...] = np.arange(10)

    f = {
        "complete": mds,
        "data/complete": mdsc,
    }

    data_paths = ["data/complete"]
    key_paths = ["complete"]
    df = DataSource(f, key_paths, data_paths, timeout=0.1)

    val = 0
    for dset in df:
        d = dset["data/complete"]
        assert dset.maxshape == [10]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val,val+1,None),)
        print(dset.slice_metadata)
        assert d == val
        val = val + 1


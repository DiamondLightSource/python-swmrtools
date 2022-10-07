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
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        print(dset.slice_metadata)
        assert d == val
        val = val + 1


def test_iterates_complete_interleaved_datasets():

    mdsc1 = utils.make_mock([11])
    mdsc2 = utils.make_mock([11])
    mdsc3 = utils.make_mock([10])
    mdsc4 = utils.make_mock([10])
    mds = utils.make_mock([42])
    mds.dataset[...] = 1
    mdsc1.dataset[...] = np.arange(0, 42, 4)
    mdsc2.dataset[...] = np.arange(1, 42, 4)
    mdsc3.dataset[...] = np.arange(2, 42, 4)
    mdsc4.dataset[...] = np.arange(3, 42, 4)

    f = {
        "complete": mds,
        "data/complete1": mdsc1,
        "data/complete2": mdsc2,
        "data/complete3": mdsc3,
        "data/complete4": mdsc4,
    }

    data_paths = {
        "data/all": [
            "data/complete1",
            "data/complete2",
            "data/complete3",
            "data/complete4",
        ]
    }
    key_paths = ["complete"]
    df = DataSource(f, key_paths, None, timeout=0.1, interleaved_paths=data_paths)

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

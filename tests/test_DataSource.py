import numpy as np
from swmr_tools import DataSource
import utils


def test_iterates_complete_dataset():

    mds = utils.make_mock([10])
    mdsc = utils.make_mock([10])
    finished = utils.make_mock([1])
    mds.dataset[...] = 1
    mdsc.dataset[...] = np.arange(10)

    f = {"data/complete": mdsc}
    df = DataSource([mds], f, timeout=0.1, finished_dataset=finished)

    val = 0
    for dset in df:
        d = dset["data/complete"]
        assert dset.maxshape == [10]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        print(dset.slice_metadata)
        assert d == val
        val = val + 1

    assert val == 10

    assert not df.is_scan_finished()
    assert df.has_timed_out()

    finished.dataset[0] = 1
    df.reset()

    val = 0
    for dset in df:
        d = dset["data/complete"]
        assert dset.maxshape == [10]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 10

    assert df.is_scan_finished()
    assert not df.has_timed_out()

    finished.dataset[0] = 0
    mds.dataset[-1] = 0
    df.reset()

    val = 0
    for dset in df:
        d = dset["data/complete"]
        assert dset.maxshape == [10]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 9

    assert df.is_scan_finished()
    assert not df.has_timed_out()


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

    inter = {"data/all": [mdsc1, mdsc2, mdsc3, mdsc4]}

    df = DataSource([mds], None, timeout=0.1, interleaved_datasets=inter)

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

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
    assert val == 10

    mds.dataset[-1] = 0
    df.reset()
    val = 0
    for dset in df:
        d = dset["data/complete"]
        assert dset.maxshape == [10]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        print(dset.slice_metadata)
        assert d == val
        val = val + 1

    assert val == 9


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

    assert val == 42


def test_iterates_complete_interleaved_keys():

    mdsc1 = utils.make_mock([11])
    mdsc2 = utils.make_mock([11])
    mdsc3 = utils.make_mock([10])
    mdsc4 = utils.make_mock([10])

    mdsk1 = utils.make_mock([11])
    mdsk2 = utils.make_mock([11])
    mdsk3 = utils.make_mock([10])
    mdsk4 = utils.make_mock([10])

    mds = utils.make_mock([42])

    mdsk1.dataset[...] = 1
    mdsk2.dataset[...] = 1
    mdsk3.dataset[...] = 1
    mdsk4.dataset[...] = 1
    mds.dataset[...] = 1
    # mds.dataset[-1] = 0
    # mdsk1.dataset[-1] = 0
    mdsc1.dataset[...] = np.arange(0, 42, 4)
    mdsc2.dataset[...] = np.arange(1, 42, 4)
    mdsc3.dataset[...] = np.arange(2, 42, 4)
    mdsc4.dataset[...] = np.arange(3, 42, 4)

    f = {
        "complete": mds,
        "key1": mdsk1,
        "key2": mdsk2,
        "key3": mdsk3,
        "key4": mdsk4,
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

    interleave_keys = [
        ["key1", "key2", "key3", "key4"],
    ]
    key_paths = ["complete"]
    df = DataSource(
        f,
        key_paths,
        None,
        timeout=0.1,
        interleaved_paths=data_paths,
        interleaved_keys=interleave_keys,
    )

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 42

    mdsk2.dataset[-1] = 0

    df.reset()

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 41


def test_iterates_complete_multiple_interleaved_keys():

    mdsc1 = utils.make_mock([11])
    mdsc2 = utils.make_mock([11])
    mdsc3 = utils.make_mock([10])
    mdsc4 = utils.make_mock([10])

    mdsk1 = utils.make_mock([11])
    mdsk2 = utils.make_mock([11])
    mdsk3 = utils.make_mock([10])
    mdsk4 = utils.make_mock([10])

    mdskx1 = utils.make_mock([15])
    mdskx2 = utils.make_mock([14])
    mdskx3 = utils.make_mock([14])

    mds = utils.make_mock([42])

    mdsk1.dataset[...] = 1
    mdsk2.dataset[...] = 1
    mdsk3.dataset[...] = 1
    mdsk4.dataset[...] = 1
    mdskx1.dataset[...] = 1
    mdskx2.dataset[...] = 1
    mdskx3.dataset[...] = 1
    mds.dataset[...] = 1

    mdsc1.dataset[...] = np.arange(0, 42, 4)
    mdsc2.dataset[...] = np.arange(1, 42, 4)
    mdsc3.dataset[...] = np.arange(2, 42, 4)
    mdsc4.dataset[...] = np.arange(3, 42, 4)

    f = {
        "complete": mds,
        "key1": mdsk1,
        "key2": mdsk2,
        "key3": mdsk3,
        "key4": mdsk4,
        "keyx1": mdskx1,
        "keyx2": mdskx2,
        "keyx3": mdskx3,
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

    interleave_keys = [["key1", "key2", "key3", "key4"], ["keyx1", "keyx2", "keyx3"]]
    key_paths = ["complete"]
    df = DataSource(
        f,
        key_paths,
        None,
        timeout=0.1,
        interleaved_paths=data_paths,
        interleaved_keys=interleave_keys,
    )

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 42

    mdsk2.dataset[-1] = 0

    df.reset()

    val = 0
    for dset in df:
        d = dset["data/all"]
        assert dset.maxshape == [42]
        assert dset.index == val
        assert dset.slice_metadata == (slice(val, val + 1, None),)
        assert d == val
        val = val + 1

    assert val == 41

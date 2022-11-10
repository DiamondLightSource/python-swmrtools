from swmr_tools import KeyFollower
import utils
import numpy as np


def test_first_frame():

    shape = [10]

    k1 = utils.make_mock(shape)
    k2 = utils.make_mock(shape)
    k3 = utils.make_mock(shape)

    ks = [k1, k2, k3]

    kf = KeyFollower(ks, timeout=0.1)
    kf.check_datasets()

    assert kf.scan_rank == 1
    assert kf.maxshape == [10]

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == -1

    kf.reset()

    k1.dataset[0] = 1
    k2.dataset[0] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == -1

    kf.reset()
    k3.dataset[0] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == 0

    kf.reset()

    k1.dataset[1] = 1
    k2.dataset[1] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == 0

    kf.reset()
    k3.dataset[1] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == 1


def test_first_frame_jagged():

    k1 = utils.make_mock([2])
    k2 = utils.make_mock([2])
    k3 = utils.make_mock([1], maxshape=[2])

    ks = [k1, k2, k3]

    kf = KeyFollower(ks, timeout=0.1)
    kf.check_datasets()

    assert kf.maxshape == [2]

    assert kf.scan_rank == 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == -1

    kf.reset()

    k1.dataset[0] = 1
    k2.dataset[0] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == -1

    kf.reset()
    k3.dataset[0] = 1

    current_key = -1
    for key in kf:
        current_key += 1

    assert current_key == 0


def test_iterates_complete_dataset():

    mds = utils.make_mock()
    mds.dataset = mds.dataset + 1

    kf = KeyFollower([mds], timeout=0.1)
    kf.check_datasets()

    assert kf.scan_rank == 2
    assert kf.maxshape == [5, 10]
    current_key = 0
    for key in kf:
        current_key += 1

    assert current_key == 50


def test_iterates_incomplete_dataset():

    mds = utils.make_mock()
    mds.dataset[:2, :, :, :] = 1
    mds.dataset[
        2,
        0:5,
        :,
    ] = 1

    kf = KeyFollower([mds], timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25


def test_iterates_multiple_incomplete_dataset():

    mds = utils.make_mock()
    mds.dataset[:, :, :, :] = 1
    mdsi = utils.make_mock()
    mdsi.dataset[:2, :, :, :] = 1
    mdsi.dataset[
        2,
        0:5,
        :,
    ] = 1

    kf = KeyFollower([mds,mdsi], timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25


def test_iterates_snake_scan():

    mds = utils.make_mock()
    mds.dataset[:2, :, :, :] = 1
    mds.dataset[2, 1:, :, :] = 1

    kf = KeyFollower([mds], timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 20


def test_reads_updates():

    mds = utils.make_mock()
    mds.dataset.reshape((-1))[:26] = 1

    kf = KeyFollower([mds], timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1

        if current_key == 25:
            mds.dataset[...] = 1

    assert current_key == 50


def test_refresh_max():

    mds = utils.make_mock()
    mds.dataset.reshape((-1))[:26] = 1

    kf = KeyFollower([mds], timeout=0.1)
    kf.check_datasets()
    current_key = 0

    max = kf.get_current_max()

    assert current_key == 0

    kf.refresh()

    max = kf.get_current_max()

    assert max == 25

    assert not kf.are_keys_complete()

    mds.dataset[...] = 1

    kf.refresh()

    max = kf.get_current_max()

    assert max == 49

    assert kf.are_keys_complete()


def test_update_changes_shape():

    mds = utils.make_mock(shape=[2, 10, 1, 1])
    mds.dataset[...] = 1

    kf = KeyFollower([mds], timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1

        if current_key == 20:
            mds.dataset.resize((5, 10, 1, 1), refcheck=False)
            mds.dataset[...] = 1

    assert current_key == 50


def test_multiple_keys_from_node():

    mds = utils.make_mock()
    mds.dataset[:, :, :, :] = 1
    mdsi = utils.make_mock()
    mdsi.dataset[:2, :, :, :] = 1
    mdsi.dataset[
        2,
        0:5,
        :,
    ] = 1

    kf = KeyFollower([mds,mdsi], timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25


def test_finished_dataset():


    mds = utils.make_mock()
    mds.dataset = mds.dataset + 1
    mfds = utils.make_mock(shape=[1])
    mfds.dataset = np.array([0])
 
    kf = KeyFollower([mds], timeout=0.1, finished_dataset=mfds)
    assert not kf.is_finished()

    mfds.dataset = np.array([1])
    assert kf.is_finished()

    # can't have a finished array with more than one element
    mfds = utils.make_mock(shape=[4])
    mfds.dataset = mfds.dataset + 1
    
    kf = KeyFollower([mds], timeout=0.1, finished_dataset=mfds)
    assert not kf.is_finished()

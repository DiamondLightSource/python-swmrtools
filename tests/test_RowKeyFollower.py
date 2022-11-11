from swmr_tools import RowKeyFollower
import utils


def test_iterates_complete_dataset():

    mds = utils.make_mock()
    mds.dataset = mds.dataset + 1

    kf = RowKeyFollower([mds], timeout=0.1)
    kf.check_datasets()

    assert kf.scan_rank == 2
    assert kf.maxshape == [5, 10]

    current_key = 0
    for key in kf:
        current_key += 1

    assert current_key == 5


def test_iterates_incomplete_dataset():

    mds = utils.make_mock()
    mds.dataset[:2, :, :, :] = 1
    mds.dataset[
        2,
        0:5,
        :,
    ] = 1

    kf = RowKeyFollower([mds], timeout=0.1)
    kf.check_datasets()

    keys = []
    for key in kf:
        keys.append(key)
    assert keys == [9, 19]


def test_iterates_complete_dataset_maxshape():

    mds = utils.make_mock(shape=[5, 10, 1, 1], maxshape=(None, None, 1, 1))
    mds.dataset = mds.dataset + 1
    kf = RowKeyFollower([mds], timeout=0.1, row_size=10)
    kf.check_datasets()

    assert kf.scan_rank == 2
    assert kf.maxshape == (None, None)

    keys = []

    for key in kf:
        keys.append(key)

    assert keys == [9, 19, 29, 39, 49]

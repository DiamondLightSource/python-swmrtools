from swmr_tools import utils
import numpy as np
import h5py

def test_row_slice():

    shape = [3,4,5,6,7,8]
    index = 5
    scan_rank = 4

    s = utils.get_row_slice(index, shape, scan_rank)

    assert s == (slice(0, 1, None),
                 slice(0, 1, None),
                 slice(0, 1, None),
                 slice(None, None, None))

    index = 359
    s = utils.get_row_slice(index, shape, scan_rank)

    assert s == (slice(2, 3, None),
                 slice(3, 4, None),
                 slice(4, 5, None),
                 slice(None, None, None))

    index = 4
    s = utils.get_row_slice(index, shape, scan_rank)

    assert s == (slice(0, 1, None),
                 slice(0, 1, None),
                 slice(0, 1, None),
                 slice(None, None, None))

    index = 354
    s = utils.get_row_slice(index, shape, scan_rank)

    assert s == (slice(2, 3, None),
                 slice(3, 4, None),
                 slice(4, 5, None),
                 slice(None, None, None))

def test_get_position():

    index = 0
    shape = [2,3,4,5]
    scan_rank = 2
    out = utils.get_position(index, shape, scan_rank)

    print(out)

    assert out[0] == (0, 0)
    assert out[1] == [slice(0, None, 1), slice(0, None, 1), slice(0, None, 1), slice(0, None, 1)]
    assert out[2] == slice(0, 2, 1)


def test_create_dataset(tmp_path):
    f = str(tmp_path / "scan.h5")

    scan_max = (40,60)
    data = np.ones((400,500))
    smalldata = np.ones((1))
    path = "dataset"
    smallpath = "small"

    with h5py.File(f,'w') as fh:
        utils.create_dataset(data, scan_max, fh, path)
        utils.create_dataset(smalldata, scan_max, fh, smallpath)

    with h5py.File(f,'r') as fh:
        ds = fh["/dataset"]
        smallds = fh["/small"]
        assert ds.shape == (1,1,400,500)
        assert ds.maxshape == (40,60,400,500)
        assert ds.chunks == (1,1,400,500)

        assert smallds.shape == (1, 1, 1)
        assert smallds.maxshape == (40,60,1)
        assert smallds.chunks == (40,60,1)

def test_append_data(tmp_path):
    f = str(tmp_path / "scan.h5")

    scan_max = (2,3)
    data = np.ones((4,5))
    path = "dataset"

    with h5py.File(f,'w') as fh:
        ds = utils.create_dataset(data, scan_max, fh, path)
        utils.append_data(data,(slice(1,2),slice(2,3)),ds)


    with h5py.File(f,'r') as fh:
        ds = fh["/dataset"]

        assert ds.shape == (2,3,4,5)
        assert ds.maxshape == (2,3,4,5)

def test_check_file_readable(tmp_path):
    f = str(tmp_path / "scan.h5")

    scan_max = (2,3)
    data = np.ones((4,5))
    path = "dataset"

    with h5py.File(f,'w') as fh:
        ds = utils.create_dataset(data, scan_max, fh, path)


    assert utils.check_file_readable(f,["/dataset"],timeout = 0.1) == True


    assert utils.check_file_readable(f,["/datase"],timeout = 0.1) == False



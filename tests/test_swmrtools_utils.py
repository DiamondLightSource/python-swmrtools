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

    assert out == (0, 0)


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


def test_nexus_utils(tmp_path):

    fin = str(tmp_path / "in.h5")
    fout = str(tmp_path / "out.h5")

    axes = ["y","x",".","."]
    shape = [2,3,4,5]
    signal = "data"

    with h5py.File(fin,'w') as fh:
        e = utils.create_nxentry(fh,"entry",default=True)
        d = utils.create_nxdata(e,"data",default=True)
        d.attrs["axes"] = axes
        d.attrs["signal"] = "data"
        d.attrs["x_indices"] = 1
        d.attrs["y_indices"] = 0
        d.create_dataset("data", shape)
        d.create_dataset("y", [shape[0]])
        d.create_dataset("x", [shape[1]])


    with h5py.File(fin,'r') as fhin, h5py.File(fout,'w') as fhout:
        e = utils.create_nxentry(fhout,"entry1",default=True)
        d1 = utils.create_nxdata(e,"data1",default=True)
        d2 = utils.create_nxdata(e,"data2",default=True)
        utils.copy_nexus_axes(fhin["/entry/data"], d1, 2)
        utils.copy_nexus_axes(fhin["/entry/data"], d2, 2, frame_axes=[".","z"])

    with h5py.File(fout,'r') as fhout:
        d1 = fhout["/entry1/data1"]
        d2 = fhout["/entry1/data2"]

        def check_group(g):
            assert "axes" in g.attrs
            assert "x_indices" in g.attrs
            assert g.attrs["x_indices"] == 1
            assert "y_indices" in g.attrs
            assert g.attrs["y_indices"] == 0
            assert "x" in g
            assert "y" in g

        check_group(d1)
        check_group(d2)

def test_check_file_readable(tmp_path):
    f = str(tmp_path / "scan.h5")

    scan_max = (2,3)
    data = np.ones((4,5))
    path = "dataset"

    with h5py.File(f,'w') as fh:
        ds = utils.create_dataset(data, scan_max, fh, path)


    assert utils.check_file_readable(f,["/dataset"],timeout = 0.1) == True
    assert utils.check_file_readable(f,["/datase"],timeout = 0.1) == False



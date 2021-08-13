import h5py
import numpy as np
from swmr_tools import KeyFollower, DataSource
from functools import reduce


def test_multiple_keys(tmp_path):

    f = str(tmp_path / "f.h5")

    with h5py.File(f, "w") as fh:
        fh.create_dataset("complete", data=np.ones((10)), maxshape=(10,))
        d1 = np.zeros((10))
        d1[0:5] = 1
        fh.create_dataset("incomplete", data=d1, maxshape=(10,))

    ps = ["/"]

    with h5py.File(f, "r") as fh:
        kf = KeyFollower(fh, ps, timeout=0.1)
        kf.check_datasets()

        assert kf.scan_rank == 1

        current_key = 0
        for key in kf:
            current_key += 1

        assert current_key == 5

def test_complete_keys(tmp_path):

    f = str(tmp_path / "f.h5")

    with h5py.File(f, "w") as fh:
        fh.create_dataset("complete", data=np.ones((10)), maxshape=(10,))
        d1 = np.ones((10))
        fh.create_dataset("complete2", data=d1, maxshape=(10,))

    ps = ["/"]

    with h5py.File(f, "r") as fh:
        kf = KeyFollower(fh, ps, timeout=0.1)
        kf.check_datasets()

        assert kf.scan_rank == 1

        current_key = 0
        for key in kf:
            current_key += 1

        assert current_key == 10

def test_data_read(tmp_path):

    f = str(tmp_path / "f.h5")

    create_test_file(f)

    with h5py.File(f, "r") as fh:

        data_paths = ["/data"]
        key_paths = ["/key"]
        df = DataSource(fh, key_paths, data_paths, timeout=1)

        count = 0
        base = np.arange(4 * 5)
        base = base.reshape((4, 5))
        for dset in df:
            d = dset["/data"]
            assert np.all(d == base + (20 * count))
            count = count + 1

def test_use_case_example(tmp_path):

    f = str(tmp_path / "f.h5")
    o = str(tmp_path / "o.h5")

    create_test_file(f)

    output_path = "result"

    with h5py.File(f, "r") as fh, h5py.File(o, "w") as oh:

        data_paths = ["/data"]
        key_paths = ["/key"]
        df = DataSource(fh, key_paths, data_paths, timeout=1)


        output = None

        for dset in df:
            d = dset["/data"]
            d = d.squeeze()
            r = d.sum(axis=1)
            assert dset.maxshape == (2,3)

            if output is None:
                maxshape = dset.maxshape + r.shape
                shape = ([1] * len(dset.maxshape) + list(r.shape))
                r = r.reshape(shape)
                output = oh.create_dataset(output_path, data=r, maxshape = maxshape)
            else:
                s = dset.slice_metadata
                ds = tuple(slice(0,s,1) for s in r.shape)
                fullslice = s + ds
                new_shape = tuple(s.stop for s in fullslice)
                if (np.any(new_shape > output.shape)):
                    output.resize(new_shape)
                output[fullslice] = r



    with h5py.File(o, "r") as oh:
        out = oh["/result"]
        assert out.shape == (2,3,4)
        assert out.maxshape == (2,3,4)
        print(out[1,2,:])
        assert 119+118+117+116+115 == out[1,2,3]


def create_test_file(path):

    with h5py.File(path, "w") as fh:
        shape = (2, 3, 4, 5)
        size = reduce(lambda x, y: x * y, shape)
        d = np.arange(size)
        d = d.reshape(shape)
        fh.create_dataset("data", data=d, maxshape=shape)
        k = np.ones(shape[:-2])
        fh.create_dataset("key", data=k, maxshape=shape[:-2])



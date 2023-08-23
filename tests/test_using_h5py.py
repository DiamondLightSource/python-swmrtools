import h5py
import numpy as np
import hdf5plugin
import time
import multiprocessing as mp
from swmr_tools import KeyFollower, DataSource, utils
from functools import reduce


def test_multiple_keys(tmp_path):
    f = str(tmp_path / "f.h5")

    with h5py.File(f, "w") as fh:
        fh.create_dataset("complete", data=np.ones((10)), maxshape=(10,))
        d1 = np.zeros((10))
        d1[0:5] = 1
        fh.create_dataset("incomplete", data=d1, maxshape=(10,))

    with h5py.File(f, "r") as fh:
        keys = [fh["complete"], fh["incomplete"]]

        kf = KeyFollower(keys, timeout=0.1)
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

    with h5py.File(f, "r") as fh:
        keys = [fh["complete"], fh["complete2"]]
        kf = KeyFollower(keys, timeout=0.1)
        kf.check_datasets()

        assert kf.scan_rank == 1

        current_key = 0
        for key in kf:
            current_key += 1

        assert current_key == 10


def test_data_read(tmp_path):
    inner_data_read(tmp_path, False)


def test_data_read_direct(tmp_path):
    inner_data_read(tmp_path, True)


def inner_data_read(tmp_path, direct):
    f = str(tmp_path / "f.h5")

    create_test_file(f)

    with h5py.File(f, "r") as fh:
        keys = [
            fh["/key"],
        ]
        data = {"/data": fh["/data"]}

        df = DataSource(
            keys,
            data,
            timeout=1,
            use_direct_chunk=direct,
        )

        count = 0
        base = np.arange(4 * 5)
        base = base.reshape((4, 5))
        for dset in df:
            d = dset["/data"]
            assert d.shape == (1, 1, 4, 5)
            assert np.all(d == base + (20 * count))
            count = count + 1


def test_use_case_example(tmp_path):
    f = str(tmp_path / "f.h5")
    o = str(tmp_path / "o.h5")

    create_test_file(f)

    output_path = "result"

    with h5py.File(f, "r") as fh, h5py.File(o, "w") as oh:
        keys = [
            fh["/key"],
        ]
        data = {"/data": fh["/data"]}
        df = DataSource(keys, data, timeout=1)

        output = None

        for dset in df:
            d = dset["/data"]
            d = d.squeeze()
            r = d.sum(axis=1)
            assert dset.maxshape == (2, 3)

            if output is None:
                output = df.create_dataset(r, oh, output_path)
            else:
                df.append_data(r, dset.slice_metadata, output)

    with h5py.File(o, "r") as oh:
        out = oh["/result"]
        assert out.shape == (2, 3, 4)
        assert out.maxshape == (2, 3, 4)
        assert 119 + 118 + 117 + 116 + 115 == out[1, 2, 3]
        assert out[0, 1, 0] != 0


def test_mock_scan(tmp_path):
    f = str(tmp_path / "scan.h5")

    p = mp.Process(target=mock_scan, args=(f,))
    p.start()

    utils.check_file_readable(f, ["/data", "/key"], timeout=5)

    with h5py.File(f, "r", libver="latest", swmr=True) as fh:
        keys = [
            fh["/key"],
        ]
        data = {"/data": fh["/data"]}
        finished = fh["finished"]
        df = DataSource(keys, data, timeout=1, finished_dataset=finished)

        count = 1

        assert p.is_alive()
        for dset in df:
            d = dset["/data"]
            assert d[0, 0, 0].item() == count
            count = count + 1

    p.join()


def test_mock_grid_scan(tmp_path):
    f = str(tmp_path / "scan.h5")

    p = mp.Process(target=mock_grid_scan, args=(f,))
    p.start()

    utils.check_file_readable(f, ["/data", "/key"], timeout=5)

    with h5py.File(f, "r", libver="latest", swmr=True) as fh:
        keys = [
            fh["/key"],
        ]
        data = {"/data": fh["/data"]}
        finished = fh["finished"]
        df = DataSource(keys, data, timeout=3, finished_dataset=finished)

        count = 1

        assert p.is_alive()
        for dset in df:
            d = dset["/data"]
            assert d[0, 0, 0, 0].item() == count
            count = count + 1

    p.join()


def create_test_file(path):
    with h5py.File(path, "w") as fh:
        shape = (2, 3, 4, 5)
        size = reduce(lambda x, y: x * y, shape)
        d = np.arange(size)
        d = d.reshape(shape)
        fh.create_dataset(
            "data",
            data=d,
            maxshape=shape,
            chunks=(1, 1, 4, 5),
            **hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
            )
        )

        k = np.ones(shape[:-2])
        fh.create_dataset("key", data=k, maxshape=shape[:-2])


def mock_scan(path):
    with h5py.File(path, "w", libver="latest") as fh:
        maxn = 10
        maxshape = (maxn, 9, 10)
        shape = (1, 9, 10)

        d = np.zeros(shape)

        ds = fh.create_dataset("data", data=d, maxshape=maxshape)
        k = np.zeros((1,))
        ks = fh.create_dataset("key", data=k, maxshape=(20,))
        finished = fh.create_dataset("finished", data=np.zeros((1,)), maxshape=(1,))
        fh.swmr_mode = True
        for i in range(maxn):
            s = (i + 1, 9, 10)
            if i != 0:
                ds.resize(s)
                ks.resize((i + 1,))
            ds[i, :, :] = np.ones(shape) + i
            ds.flush()
            ks[i] = 1
            ks.flush()
            time.sleep(1)
        finished[0] = 1
        finished.flush()


def mock_grid_scan(path):
    with h5py.File(path, "w", libver="latest") as fh:
        maxshape = (3, 4, 9, 10)
        shape = (1, 1, 9, 10)

        d = np.zeros(shape)

        ds = fh.create_dataset("data", data=d, maxshape=maxshape)
        k = np.zeros((1, 1))
        ks = fh.create_dataset("key", data=k, maxshape=(3, 4))
        finished = fh.create_dataset("finished", data=np.zeros((1,)), maxshape=(1,))
        fh.swmr_mode = True
        count = 1
        for j in range(3):
            for i in range(4):
                # s = (j + 1, i + 1, 9, 10)
                if i == 0:
                    ds.resize((j + 1, 4, 9, 10))
                    ks.resize((j + 1, 4))
                ds[j, i, :, :] = np.ones(shape) * count
                count += 1
                ds.flush()
                ks[j, i] = 1
                ks.flush()
                time.sleep(1)

        finished[0] = 1
        finished.flush()

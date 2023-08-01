import h5py
import numpy as np
import hdf5plugin
import math
from swmr_tools import ChunkSource, chunk_utils, utils
import time
import multiprocessing as mp


def test_chunk_source_static(tmp_path):
    f = str(tmp_path / "chunk.h5")
    create_test_file(f)

    with h5py.File(f, "r") as fh:
        ds = fh["/data"]

        dd = {"data": ds}

        cs = ChunkSource(dd, timeout=0.5)

        counter = 0
        for c in cs:
            assert c is not None
            assert len(c) == 1
            assert c.index == counter * 10
            assert "data" in c

            if counter != 2:
                assert c["data"].shape == (10, 4, 5)
            else:
                assert c["data"].shape == (5, 4, 5)

            counter += 1

    assert counter == 3


def test_mock_scan(tmp_path):
    f = str(tmp_path / "scan.h5")

    p = mp.Process(target=mock_scan, args=(f,))
    p.start()

    utils.check_file_readable(f, "/data", timeout=2, retrys=10)

    with h5py.File(f, "r", libver="latest", swmr=True) as fh:
        ds = fh["/data"]

        dd = {"data": ds}

        cs = ChunkSource(dd)

        counter = 0
        for c in cs:
            assert c is not None
            assert len(c) == 1
            assert c.index == counter * 10
            assert "data" in c
            counter += 1
            print(f"Done {c.index}")

    assert counter == 25

    p.join()

def create_test_file(path):
    with h5py.File(path, "w") as fh:
        shape = (25, 4, 5)
        size = math.prod(shape)
        d = np.arange(size)
        d = d.reshape(shape)
        print(f"Max val is {d.max()}")
        fh.create_dataset(
            "data",
            data=d,
            shape=shape,
            maxshape=shape,
            chunks=(10, 4, 5),
            **hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )


def mock_scan(path):
    with h5py.File(path, "w", libver="latest") as fh:
        shape = (250, 4, 5)
        cshape = (10, 4, 5)
        size = math.prod(cshape)
        d = np.arange(size)
        d = d.reshape(cshape)
        print(f"Max val is {d.max()}")
        ds = fh.create_dataset(
            "data",
            shape=shape,
            maxshape=shape,
            chunks=cshape,
            **hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )

        fh.swmr_mode = True
        count = 0
        for i in range(0, shape[0], cshape[0]):
            print(i)
            ds[i : i + (cshape[0]), :, :] = d + count
            ds.flush()
            count += 1
            time.sleep(0.15)


def test_chunk_write_raster(tmp_path):
    f = str(tmp_path / "output.h5")
    shape = [35, 28]
    chunk_size = 7
    run_raster_write(f, shape, chunk_size)


def test_chunk_write_raster_smallx(tmp_path):
    f = str(tmp_path / "output.h5")
    shape = [35, 28]
    chunk_size = 50
    run_raster_write(f, shape, chunk_size)


def run_raster_write(f, shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)
    dsname = "output"

    chunks = (1, chunk_size) if chunk_size < shape[-1] else (1, shape[-1])

    with h5py.File(f, "w") as ofh:
        output = ofh.create_dataset(dsname, shape=shape, chunks=chunks)

        for i in range(0, npoints, chunk_size):
            ss = chunk_utils.get_slice_structure(i, chunk_size, shape, False)
            chunk_utils.write_data(ss, data, last_data, output)

            last_data = data
            data = data.copy()
            data += chunk_size

    expected = np.arange(npoints).reshape(shape)

    with h5py.File(f, "r") as fh:
        assert np.all(expected == fh[dsname][...])


def test_chunk_write_snake(tmp_path):
    f = str(tmp_path / "output.h5")
    shape = [35, 28]
    chunk_size = 7

    run_snake_write(f, shape, chunk_size)


def test_chunk_write_snake_smallx(tmp_path):
    f = str(tmp_path / "output.h5")
    shape = [35, 28]
    chunk_size = 50

    run_snake_write(f, shape, chunk_size)


def run_snake_write(f, shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)
    dsname = "output"

    chunks = (1, chunk_size) if chunk_size < shape[-1] else (1, shape[-1])

    with h5py.File(f, "w") as ofh:
        output = ofh.create_dataset(dsname, shape=shape, chunks=chunks)

        for i in range(0, npoints, chunk_size):
            ss = chunk_utils.get_slice_structure(i, chunk_size, shape, True)
            chunk_utils.write_data(ss, data, last_data, output)

            last_data = data
            data = data.copy()
            data += chunk_size

    expected = np.arange(npoints).reshape(shape)

    for i in range(npoints):
        position = utils.get_position_snake(i, shape, len(shape))
        s = [slice(p, p + 1) for p in position]
        expected[tuple(s)] = i

    with h5py.File(f, "r") as fh:
        assert np.all(expected == fh[dsname][...])

import h5py
import numpy as np
import hdf5plugin
import math
from swmr_tools import ChunkSource, utils
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


# def test_real_file():
#     fp = "/dls/i18/data/2023/cm33872-3/i18-205537.nxs"
#     dp = "/entry/instrument/Xspress3A/raw_mca0"
#     fo = "/scratch/data/odin_test_swmrtools.nxs"

#     snake_key = '"alternating":true'
#     model = "/entry/diamond_scan/scan_models"
#     scan_shape = "/entry/diamond_scan/scan_shape"

#     with h5py.File(fp,'r') as fh, h5py.File(fo, 'w', libver='latest') as ofh:
#         ds = fh[dp]
#         cshape = ds.chunks
#         is_snake = snake_key in fh[model][...].item().decode()
#         sshape = fh[scan_shape][...]

#         nxe = ofh.create_group("entry")
#         nxe.attrs["NX_class"] = "NXentry"

#         mcag = nxe.create_group("mca")
#         mcag.attrs["NX_class"] = "NXdata"
#         mcag.attrs["signal"] = "data"

#         sumg = nxe.create_group("sum")
#         sumg.attrs["NX_class"] = "NXdata"
#         sumg.attrs["signal"] = "data"

#         xaxis = fh["/entry/Xspress3A/t1x_value_set"][:sshape[1]]
#         yaxis = fh["/entry/Xspress3A/t1y_value_set"][::sshape[1]]

#         mcag.attrs["axes"] = ["t1y_value_set", "t1x_value_set", "."]
#         mcag.create_dataset("t1x_value_set",data = xaxis)
#         mcag.create_dataset("t1y_value_set",data = yaxis)

#         sumg.attrs["axes"] = ["t1y_value_set", "t1x_value_set"]
#         sumg.create_dataset("t1x_value_set",data = xaxis)
#         sumg.create_dataset("t1y_value_set",data = yaxis)

#         output = np.zeros(sshape)
#         omca = mcag.create_dataset("data", shape = (sshape[0],sshape[1], cshape[2]), chunks=(1,cshape[0],cshape[2]), dtype = ds.dtype, compression="lzf")
#         osum = sumg.create_dataset("data", data = output)


#         dd = {"data" : ds}
#         cs = ChunkSource(dd)
#         last_mca = None
#         last_ff = None
#         for c in cs:
#             mcas = c["data"].squeeze()
#             ff = mcas.sum(axis = 1)
#             pos = utils.get_position_snake(c.index, sshape, len(sshape))
#             ss = chunk_utils.get_slice_structure(pos,cshape[0],sshape, True)
#             chunk_utils.write_data(ss, pos, mcas, last_mca, omca)
#             chunk_utils.write_data(ss, pos, ff, last_ff, osum)
#             last_ff = ff
#             last_mca = mcas


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

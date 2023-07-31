from swmr_tools import utils
from swmr_tools import chunk_utils
import math
import numpy as np
import pytest


shapes = [
    ([5, 15], 32),
    ([7, 15], 46),
    ([3, 51], 100),
    ([4, 4, 3], 10),
    ([4, 5, 3], 10),
    ([2, 2], 2),
    ([3, 3], 2),
    ([4, 4], 2),
    ([5, 5], 2),
    ([6, 6], 2),
    ([3, 3], 3),
    ([4, 4], 3),
    ([5, 5], 5),
    ([6, 6], 5),
    ([10, 10], 5),
    ([35, 28], 7),
    ([35, 29], 7),
    ([35, 30], 7),
    ([35, 31], 7),
    ([35, 32], 7),
    ([35, 33], 7),
    ([35, 34], 7),
    ([35, 35], 7),
    ([12, 29], 7),
    ([2, 10], 10),
    ([13, 13], 10),
    ([2, 2, 2], 2),
    ([3, 3, 3], 2),
    ([2, 13, 20], 7),
    ([2, 3, 7, 10], 7),
]


def check_start(ss, chunk_size):
    for s in ss:
        if s.type == "current":
            so = s.current.output
        elif s.type == "last":
            so = s.last.output
        else:
            so = s.intermediate.slice

        #h5py doesnt support -ve step
            

        # start of output write should always be aligned to chunk
        if so.step is None or so.step == 1:
            assert so.start % chunk_size == 0
        else:
            assert so.step >= 1
            # if so.stop is not None:
            #     assert (so.stop + 1) % chunk_size == 0


@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_raster_scan(shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        ss = chunk_utils.get_slice_structure(i, chunk_size, shape, False)
        check_start(ss, chunk_size)
        chunk_utils.write_data(ss, data, last_data, output)

        last_data = data
        data = data.copy()
        data += chunk_size

    expected = np.arange(npoints).reshape(shape)
    assert np.all(output == expected)


@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_snake_scan(shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        ss = chunk_utils.get_slice_structure(i, chunk_size, shape, True)
        check_start(ss, chunk_size)

        chunk_utils.write_data(ss, data, last_data, output)

        last_data = data
        data = data.copy()
        data += chunk_size

    expected = np.arange(npoints).reshape(shape)

    for i in range(npoints):
        position = utils.get_position_snake(i, shape, len(shape))
        s = [slice(p, p + 1) for p in position]
        expected[tuple(s)] = i

    print("OUTPUT")
    print(output)
    print("EXPECTED")
    print(expected)
    print("DIFF")
    print(output-expected)
    assert np.all(output == expected)


def test_write():
    shape = [11, 23]
    chunk_size = 10

    scalars = np.arange(1, chunk_size + 1)

    scalar_out = np.zeros(shape)

    vectors = np.vstack([scalars] * 6).T
    vector_shape = shape + [vectors.shape[1]]
    vector_out = np.zeros(vector_shape)

    ss = chunk_utils.get_slice_structure(10, chunk_size, shape, False)

    chunk_utils.write_data(ss, scalars, scalars, scalar_out)

    assert scalar_out[0, 10] == 1
    assert scalar_out[0, 19] == 10

    chunk_utils.write_data(ss, vectors, vectors, vector_out)

    assert vector_out[0, 10, 0] == 1
    assert vector_out[0, 19, 0] == 10
    assert vector_out[0, 10, 5] == 1
    assert vector_out[0, 19, 5] == 10

    ss = chunk_utils.get_slice_structure(30, chunk_size, shape, True)
    chunk_utils.write_data(ss, scalars, scalars, scalar_out)

    assert scalar_out[1, 22] == 4
    assert scalar_out[1, 19] == 7

    chunk_utils.write_data(ss, vectors, vectors, vector_out)

    assert vector_out[0, 10, 0] == 1
    assert vector_out[0, 19, 0] == 10
    assert vector_out[0, 10, 5] == 1
    assert vector_out[0, 19, 5] == 10

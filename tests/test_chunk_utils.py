from swmr_tools import utils
from swmr_tools import chunk_utils
import math
import numpy as np
import pytest


shapes = [
    ([5, 15], 32),
    ([7, 15], 46),
    ([11, 51], 100),
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
]


def check_start(ss, chunk_size):
    for s in ss:
        if s.type == "current":
            so = s.current.output
        elif s.type == "last":
            so = s.last.output
        else:
            so = s.intermediate.slice
        # start of output write should always be aligned to chunk
        assert so.start % chunk_size == 0


# def write_data(ss, spos, data, last_data, output, chunk_size):
#     for s in ss:
#         so = None
#         if s.type == "current":
#             si = s.current.input
#             so = s.current.output
#             output[spos[0], so] += data[si]
#         elif s.type == "last":
#             si = s.last.input
#             so = s.last.output
#             output[spos[0], so] += last_data[si]

#         else:
#             # Combined
#             intermediate = np.zeros(s.intermediate.size)
#             ls = s.last.input
#             li = s.last.output
#             intermediate[li] += last_data[ls]

#             cs = s.current.input
#             ci = s.current.output
#             intermediate[ci] += data[cs]

#             so = s.intermediate.slice

#             output[spos[0], so] += intermediate

#         # start of output write should always be aligned to chunk
#         assert so.start % chunk_size == 0


@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_raster_scan(shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        spos = utils.get_position(i, shape, len(shape))

        ss = chunk_utils.get_slice_structure(spos, chunk_size, shape, False)
        print(ss)
        check_start(ss, chunk_size)

        chunk_utils.write_data(ss, data, last_data, output)

        last_data = data
        data = data.copy()
        data += chunk_size

    expected = np.arange(npoints).reshape(shape)
    print("OUTPUT")
    print(output)
    print("EXPECTED")
    print(expected)
    print("DIFF")
    print(output - expected)
    assert np.all(output == expected)


@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_snake_scan(shape, chunk_size):
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        spos = utils.get_position_snake(i, shape, len(shape))

        ss = chunk_utils.get_slice_structure(spos, chunk_size, shape, True)
        print(ss)
        check_start(ss, chunk_size)
        
        chunk_utils.write_data(ss, data, last_data, output)

        last_data = data
        data = data.copy()
        data += chunk_size

    expected = np.arange(npoints).reshape(shape)

    # step_slice = slice(1,None,2)
    # flip_slice = slice(None,None,-1)
    # all = slice(0,None)

    # full_slice = [all] * len(expected.shape)
    
    for i in range(npoints):
        position= utils.get_position_snake(i, shape, len(shape))
        s = [slice(p,p+1) for p in position]
        expected[tuple(s)] = i

    # expected[1::2, :] = expected[1::2, ::-1]
    # expected[ :, :] = expected[:, :]
    print("OUTPUT")
    print(output)
    print("EXPECTED")
    print(expected)
    print("DIFF")
    print(output - expected)
    assert np.all(output == expected)


def test_write():
    print("GO")
    shape = [11, 23]
    chunk_size = 10

    scalars = np.arange(1, chunk_size + 1)

    scalar_out = np.zeros(shape)

    vectors = np.vstack([scalars] * 6).T
    vector_shape = shape + [vectors.shape[1]]
    vector_out = np.zeros(vector_shape)

    # current
    spos = utils.get_position(10, shape, len(shape))
    ss = chunk_utils.get_slice_structure(spos, chunk_size, shape, False)

    chunk_utils.write_data(ss, scalars, scalars, scalar_out)

    assert scalar_out[0, 10] == 1
    assert scalar_out[0, 19] == 10

    chunk_utils.write_data(ss, vectors, vectors, vector_out)

    assert vector_out[0, 10, 0] == 1
    assert vector_out[0, 19, 0] == 10
    assert vector_out[0, 10, 5] == 1
    assert vector_out[0, 19, 5] == 10

    # last + combined
    spos = utils.get_position_snake(30, shape, len(shape))
    ss = chunk_utils.get_slice_structure(spos, chunk_size, shape, True)
    chunk_utils.write_data(ss, scalars, scalars, scalar_out)

    assert scalar_out[1, 22] == 4
    assert scalar_out[1, 19] == 7

    chunk_utils.write_data(ss, vectors, vectors, vector_out)

    assert vector_out[0, 10, 0] == 1
    assert vector_out[0, 19, 0] == 10
    assert vector_out[0, 10, 5] == 1
    assert vector_out[0, 19, 5] == 10

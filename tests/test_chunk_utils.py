from swmr_tools import utils
from swmr_tools import chunk_utils
import math
import numpy as np
import pytest


shapes = [ ([5,13], 10),
           ([10,15], 10),
           ([10,10], 10),
           ([10,33], 10),
           ([7,33], 7),]



def write_data(ss, spos, data, last_data, output, chunk_size):
    for s in ss:
        so = None
        if s.type == "current":
            si = s.current.input
            so = s.current.output
            output[spos[0],so] += data[si]
            print(f"Write current {so}")
        elif s.type == "last":
            si = s.last.input
            so = s.last.output
            output[spos[0],so] += last_data[si]
            print(f"Write last {so}")

        else:
            #Combined
            intermediate = np.zeros(s.intermediate.size)
            ls = s.last.input
            li = s.last.output
            intermediate[li] += last_data[ls]

            cs = s.current.input
            ci = s.current.output
            intermediate[ci] += data[cs]
            
            so = s.intermediate.slice

            output[spos[0],so] += intermediate
            print(f"Write combined {so}")

        assert so.start % chunk_size == 0


@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_raster_scan(shape, chunk_size):

    cshape = [chunk_size]
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        spos = utils.get_position(i, shape, 2)

        print(f"run for {spos} in {shape}")

        ss = chunk_utils.non_snake_routine(spos, chunk_size,shape)

        write_data(ss, spos, data, last_data, output, chunk_size)
            
        last_data = data
        data = data.copy()
        data += chunk_size
    print(output)

    expected = np.arange(npoints).reshape(shape)
    print(expected)

    assert np.all(output == expected)

@pytest.mark.parametrize("shape, chunk_size", shapes)
def test_snake_scan(shape, chunk_size):

    cshape = [chunk_size]
    npoints = math.prod(shape)

    data = np.arange(chunk_size)
    last_data = None

    output = np.zeros(shape)

    for i in range(0, npoints, chunk_size):
        spos = utils.get_position_snake(i, shape, 2)

        if spos[0] % 2 == 1:
            ss = chunk_utils.snake_routine(spos,chunk_size,shape)
        else:
            ss = chunk_utils.non_snake_routine(spos,chunk_size,shape)

        try:
            write_data(ss, spos, data, last_data, output, chunk_size)
        except ValueError as t:
            print(output)
            raise t
 
        last_data = data
        data = data.copy()
        data += chunk_size
    print(output)

    expected = np.arange(npoints).reshape(shape)

    expected[1::2,:] = expected[1::2,::-1]
    print(expected)
    assert np.all(output == expected)

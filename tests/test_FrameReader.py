from swmr_tools.datasource import FrameReader
import numpy as np


def test_framereader_scalar():
    
    r = np.arange(10)
    fh = {"ds": r}
    dataset = "ds"

    fr = FrameReader(dataset, fh, 1)
    
    for i in range(10):
        val = fr.read_frame(i)
        assert(val == i)



def test_framereader_linear():
    
    r = np.arange(100)
    r = r.reshape((10,10))

    fh = {"ds": r}
    dataset = "ds"

    fr = FrameReader(dataset, fh, 1)
    
    base = np.arange(10)

    for i in range(10):
        val = fr.read_frame(i)
        print(val)
        assert(np.all(val == base + (10*i)))


def test_framereader_image():
    
    r = np.arange(2000)
    r = r.reshape((10,10,20))

    fh = {"ds": r}
    dataset = "ds"

    fr = FrameReader(dataset, fh, 1)
    
    base = np.arange(200)
    base = base.reshape(10,20)

    for i in range(10):
        val = fr.read_frame(i)
        print(val)
        assert(np.all(val == base + (200*i)))


def test_framereader_image_grid():
    
    r = np.arange(6000)
    r = r.reshape((3,10,10,20))

    fh = {"ds": r}
    dataset = "ds"

    fr = FrameReader(dataset, fh, 2)
    
    base = np.arange(200)
    base = base.reshape(10,20)

    for i in range(30):
        val = fr.read_frame(i)
        print(val)
        assert(np.all(val == base + (200*i)))

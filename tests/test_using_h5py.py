import h5py
import numpy as np
from swmr_tools import KeyFollower, DataSource
from functools import reduce 

def test_multiple_keys(tmp_path):

    f = str(tmp_path / "f.h5")

    with h5py.File(f,'w') as fh:
        fh.create_dataset("complete", data=np.ones((10)), maxshape=(10,))
        d1 = np.zeros((10))
        d1[0:5] = 1
        fh.create_dataset("incomplete", data=d1, maxshape = (10,))

    ps = ["/"]
    
    with h5py.File(f,'r') as fh:
        kf = KeyFollower(fh, ps, timeout=0.1)
        kf.check_datasets()

        assert kf.scan_rank == 1

        current_key = 0
        for key in kf:
            current_key += 1

        assert current_key == 5


def test_data_read(tmp_path):

    f = str(tmp_path / "f.h5")

    with h5py.File(f,'w') as fh:
        shape = (2,3,4,5)
        size = reduce(lambda x, y: x * y, shape)
        d = np.arange(size)
        d = d.reshape(shape)
        fh.create_dataset("data", data=d, maxshape = shape)
        k =np.ones(shape[:-2])
        fh.create_dataset("key", data=k, maxshape=shape[:-2])

    ps = ["key"]
    
    with h5py.File(f,'r') as fh:

        data_paths = ["/data"]
        key_paths = ["/key"]
        df = DataSource(fh, key_paths, data_paths, timeout=1)
        
        count = 0
        base = np.arange(4*5)
        base = base.reshape((4,5))
        for dset in df:
            d = dset["/data"]
            assert np.all(d == base + (20 * count))
            count = count + 1



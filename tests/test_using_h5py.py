import h5py
import numpy as np
from swmr_tools import Follower


def test_multiple_keys(tmp_path):

    f = str(tmp_path / "f.h5")

    with h5py.File(f,'w') as fh:
        fh.create_dataset("complete", data=np.ones((10)), maxshape=(10,))
        d1 = np.zeros((10))
        d1[0:5] = 1
        fh.create_dataset("incomplete", data=d1, maxshape = (10,))

    ps = ["/complete","/incomplete"]
    
    with h5py.File(f,'r') as fh:
        kf = Follower(fh, ps, timeout=0.1)
        kf.check_datasets()

        assert kf.scan_rank == 1

        current_key = 0
        for key in kf:
            current_key += 1

        assert current_key == 5



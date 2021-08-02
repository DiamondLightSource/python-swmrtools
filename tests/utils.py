from mock import Mock
import numpy as np


def make_mock(shape=[5, 10, 1, 1]):

    mds = Mock()
    mds.dataset = np.zeros(shape)
    mds.refresh = lambda: print("Refresh")
    mds.maxshape = shape
    mds.shape = shape

    def slicemock(value):
        return mds.dataset[value]

    mds.__getitem__ = Mock(side_effect=slicemock)
    return mds

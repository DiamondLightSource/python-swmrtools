from mock import Mock
import numpy as np


def make_mock(shape=[5, 10, 1, 1], maxshape=None):

    mds = Mock()
    mds.dataset = np.zeros(shape)

    if maxshape is None:
        mds.maxshape = shape
    else:
        mds.maxshape = maxshape
    mds.shape = shape

    def slicemock(value):
        return mds.dataset[value]

    mds.__getitem__ = Mock(side_effect=slicemock)

    mds.size = mds.dataset.size
    return mds

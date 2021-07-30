from swmr_tools import KeyFollower
from mock import Mock
import numpy as np
import pytest
import utils

def test_iterates_complete_dataset():

    key_paths = ["complete"]

    mds = utils.make_mock()
    mds.dataset = mds.dataset + 1

    f = {"complete": mds}
    kf = KeyFollower(f, key_paths, timeout=0.1)
    kf.check_datasets()

    assert kf.scan_rank == 2

    current_key = 0
    for key in kf:
        current_key += 1

    assert current_key == 50


def test_iterates_incomplete_dataset():

    mds = utils.make_mock()
    mds.dataset[:2,:,:,:] = 1
    mds.dataset[2,0:5,:,] = 1

    key_paths = ["incomplete"]
    f = {"incomplete": mds}
    kf = KeyFollower(f, key_paths, timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25


def test_iterates_multiple_incomplete_dataset():

    mds = utils.make_mock()
    mds.dataset[:,:,:,:] = 1
    mdsi = utils.make_mock()
    mdsi.dataset[:2,:,:,:] = 1
    mdsi.dataset[2,0:5,:,] = 1

    key_paths = ["complete", "incomplete"]
    f = {
            "complete": mds,
            "incomplete": mdsi
    }
    kf = KeyFollower(f, key_paths, timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25


def test_iterates_snake_scan():


    mds = utils.make_mock()
    mds.dataset[:2,:,:,:] = 1
    mds.dataset[2,1:,:,:] = 1

    key_paths = ["incomplete"]
    f = {"incomplete": mds}
    kf = KeyFollower(f, key_paths, timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 20


def test_reads_updates():

    mds = utils.make_mock()
    mds.dataset.reshape((-1))[:26] = 1


    key_paths = ["incomplete"]
    f = {"incomplete": mds}
    kf = KeyFollower(f, key_paths, timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1

        if (current_key == 25):
            mds.dataset[...] = 1

    assert current_key == 50


def test_update_changes_shape():


    mds = utils.make_mock(shape=[2,10,1,1])
    mds.dataset[...] = 1

    key_paths = ["incomplete"]
    f = {"incomplete": mds}
    kf = KeyFollower(f, key_paths, timeout=0.1)
    current_key = 0
    for key in kf:
        current_key += 1

        if (current_key == 20):
            mds.dataset.resize((5,10,1,1),refcheck=False)
            mds.dataset[...] = 1

    assert current_key == 50


def test_multiple_keys_from_node():


    mds = utils.make_mock()
    mds.dataset[:,:,:,:] = 1
    mdsi = utils.make_mock()
    mdsi.dataset[:2,:,:,:] = 1
    mdsi.dataset[2,0:5,:,] = 1

    key_paths = ["keys"]
    f = {
            "keys": ["a", "b"],
            "keys/a": mds,
            "keys/b":mdsi
    }
    kf = KeyFollower(f, key_paths, timeout=0.1)
    kf.check_datasets()
    current_key = 0
    for key in kf:
        current_key += 1
    assert current_key == 25

# Test and Feature to be added
# Given array of this form[..., 30, 0, 32, ...] if iterator was at the 30th index
# It should be able to detect that there are non-zero keys ahead of the 0 key and infer that it should
# Skip this key and return the index of the next non-zero key
def test_skip_dead_frame():
    pass


# FrameGrabber Tests
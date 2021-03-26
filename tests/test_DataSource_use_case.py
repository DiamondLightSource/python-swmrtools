import numpy as np
from swmr_tools import DataSource
import DataSourceDatasets as Dataset


def test_iterates_complete_dataset():

    f = {
        "keys": {"complete": Dataset.complete_dataset_keys()},
        "data/complete": Dataset.complete_dataset_data(),
    }

    data_paths = ["data/complete"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    current_key = 0
    for dset in df:
        current_key += 1
    assert current_key == 50


def test_iterates_incomplete_dataset():

    f = {
        "keys": {"incomplete": Dataset.incomplete_dataset_keys()},
        "data/incomplete": Dataset.incomplete_dataset_data(),
    }

    data_paths = ["data/incomplete"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    current_key = 0
    for dset in df:
        current_key += 1
    assert current_key == 40


def test_iterates_multiple_incomplete_dataset():

    f = {
        "keys": {
            "complete": Dataset.complete_dataset_keys(),
            "incomplete": Dataset.incomplete_dataset_keys(),
        },
        "data/complete": Dataset.complete_dataset_data(),
        "data/incomplete": Dataset.incomplete_dataset_data(),
    }

    data_paths = ["data/complete", "data/incomplete"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    current_key = 0
    for dset in df:
        current_key += 1
    assert current_key == 40


# Check that the correct dataset is returned ignoring shapes
def test_correct_return_data_complete():
    f = {
        "keys": {"complete": Dataset.complete_dataset_keys()},
        "data/complete": Dataset.complete_dataset_data(),
    }
    data_paths = ["data/complete"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    full_dataset = np.array([])
    for dset in df:
        full_dataset = np.concatenate((full_dataset, dset[0].flatten()))
    assert (Dataset.complete_dataset_data().flatten() == full_dataset.flatten()).all()


# Test correct shapes are returned


def test_correct_return_shape():
    f = {
        "keys": {
            "four_dimensional": Dataset.four_dimensional_dataset_keys(),
            "three_dimensional": Dataset.three_dimensional_dataset_keys(),
        },
        "data/four_dimensional": Dataset.four_dimensional_dataset_data(),
        "data/three_dimensional": Dataset.three_dimensional_dataset_data(),
    }

    data_paths = ["data/four_dimensional", "data/three_dimensional"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    for dset in df:
        assert dset[0].shape == (1, 1, 1, 10) and dset[1].shape == (1, 1, 10)


def test_reset_method_iterates_correct_length():
    f = {
        "keys": {"complete": Dataset.complete_dataset_keys()},
        "data/complete": Dataset.complete_dataset_data(),
    }

    data_paths = ["data/complete"]
    key_paths = ["keys"]
    df = DataSource.DataFollower(f, key_paths, data_paths, timeout=0.1)
    current_key = 0
    for dset in df:
        current_key += 1

    df.reset()
    for dset in df:
        current_key += 1
    assert current_key == 100

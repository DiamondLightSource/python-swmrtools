import numpy as np

# Subclass numpy array
class DataSet(np.ndarray):
    pass


def create_dataset_from_numpy_array(numpy_array):
    ds = DataSet(numpy_array.shape)
    ds[:] = numpy_array[:]
    ds.refresh = lambda: None
    return ds


# Create a dataset of keys that is completely filled with non-zero values
# Number of iterations = 50
def complete_dataset():
    return create_dataset_from_numpy_array(np.arange(50).reshape(5, 10, 1, 1) + 1)


# Create a dataset of keys that is half filled with non-zero values
# Number of iterations = 25
def incomplete_dataset():
    incomplete_array = np.arange(50) + 1
    incomplete_array[25:] = 0
    incomplete_array = incomplete_array.reshape(5, 10, 1, 1)
    return create_dataset_from_numpy_array(incomplete_array)


# Create a dataset of keys that is partially filled, writing row by row
# Number of iterations = 26
def incomplete_row_by_row_dataset():
    incomplete_row_by_row_array = np.arange(50) + 1
    incomplete_row_by_row_array[26:] = 0
    incomplete_row_by_row_array = incomplete_row_by_row_array.reshape(5, 10, 1, 1)
    return create_dataset_from_numpy_array(incomplete_row_by_row_array)


# Create a dataset of keys that is partially filled, writing with a snake scan
# Number of iterations = 20
def incomplete_snake_scan_dataset():
    incomplete_snake_scan_array = np.arange(50) + 1
    incomplete_snake_scan_array = incomplete_snake_scan_array.reshape(5, 10, 1, 1)
    incomplete_snake_scan_array[3:] = 0
    incomplete_snake_scan_array[2][:-3] = 0
    return create_dataset_from_numpy_array(incomplete_snake_scan_array)


# Create a small dataset that can be updated to the full size
# Number of iterations = 25
def small_incomplete_dataset():
    small_incomplete_array = np.arange(25) + 1
    small_incomplete_array = small_incomplete_array.reshape(5, 5, 1, 1)
    return create_dataset_from_numpy_array(small_incomplete_array)


# The KeyFollower.Follower should iterate 50 times and the values produced should be 0-49
def complete_dataset_random_integers():
    return create_dataset_from_numpy_array(
        np.random.randint(10, 5000 + 1, size=(5, 10, 1, 1))
    )

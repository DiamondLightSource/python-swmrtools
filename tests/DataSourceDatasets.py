import numpy as np

# Creating datasets from numpy arrays
class DataSet(np.ndarray):
    pass


def create_dataset_from_numpy_array(numpy_array):
    ds = DataSet(numpy_array.shape)
    ds[:] = numpy_array[:]
    ds.refresh = lambda: None
    return ds


# Number of iterations = 50
def complete_dataset_data():
    return create_dataset_from_numpy_array(np.arange(50 * 10).reshape(5, 10, 1, 10))


def complete_dataset_keys():
    return create_dataset_from_numpy_array(np.arange(50).reshape(5, 10, 1, 1) + 1)


# Number of iterations = 40
def incomplete_dataset_data():
    incomplete_dataset_data = np.arange(50 * 10).reshape(5, 10, 1, 10)
    incomplete_dataset_data[-1] = 0
    return create_dataset_from_numpy_array(incomplete_dataset_data)


def incomplete_dataset_keys():
    incomplete_dataset_keys = np.arange(50).reshape(5, 10, 1, 1) + 1
    incomplete_dataset_keys[-1] = 0
    incomplete_dataset_keys = incomplete_dataset_keys.reshape(
        complete_dataset_keys().shape
    )
    return create_dataset_from_numpy_array(incomplete_dataset_keys)


def four_dimensional_dataset_data():
    four_dimensional_dataset_data = np.arange(50 * 10).reshape(5, 10, 1, 10)
    return create_dataset_from_numpy_array(four_dimensional_dataset_data)


def four_dimensional_dataset_keys():
    four_dimensional_dataset_keys = np.arange(50).reshape(5, 10, 1, 1) + 1
    return create_dataset_from_numpy_array(four_dimensional_dataset_keys)


def three_dimensional_dataset_data():
    three_dimensional_dataset_data = np.arange(50 * 10).reshape(5, 10, 1, 10)
    three_dimensional_dataset_data = three_dimensional_dataset_data.reshape(50, 1, 10)
    return create_dataset_from_numpy_array(three_dimensional_dataset_data)


def three_dimensional_dataset_keys():
    three_dimensional_dataset_keys = np.arange(50).reshape(5, 10, 1, 1) + 1
    three_dimensional_dataset_keys = three_dimensional_dataset_keys.reshape(50, 1, 1)
    return create_dataset_from_numpy_array(three_dimensional_dataset_keys)

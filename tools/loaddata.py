import glob
import numpy as np


__all__ = ['load_data', 'load_all_data', 'reshape',
           'reshape_all_datasets', 'make_sequential']


def load_data(path):
    """
    Given a path, to the folder that contains only .npy files
    This function returns a numpy array that merges all these files
    together

    Parameters
    ----------
    path : string

    Returns
    ----------
    numpy array that merges all these files
    together

    Examples
    --------
    >>> a = 'data/background/'
    >>> model_data = load_data(a)
    >>> np.shape(model_data)
    (5, 871, 913)
    >>> b = 'data/satellite/'
    >>> satellite_data = load_data(b)
    >>> np.shape(satellite_data)
    (5, 871, 913)
    """

    npy_files = sorted(glob.glob(path + '/*.npy'))
    return np.array([np.load(npy) for npy in npy_files])


def load_all_data(path_train, path_test, path_back, path_obs):
    """
    Given paths to folders that contain only .npy files,
    this function loads 4 datasets (from the 4 different folders) and
    returns 4 numpy arrays which combine the .npy files - one numpy array
    associated with each folder.

    Parameters
    ----------
    path_train : string
    path_test : string
    path_back : string
    path_obs : string

    Returns
    ----------
    one numpy array associated with each folder.

    Examples
    --------
    >>> path1 = "data/train/"
    >>> path2 = "data/test/"
    >>> path3 = "data/background/"
    >>> path4 = "data/satellite/"
    >>> train, test, back, obs = load_all_data(path1, path2, path3, path4)
    >>> np.shape(train)
    (1200, 871, 913)
    """
    train_full = load_data(path_train)
    test = load_data(path_test)
    model_data = load_data(path_back)
    satellite_data = load_data(path_obs)

    return train_full, test, model_data, satellite_data


def reshape(arr):
    """
    Given a three dimensional numpy array, this function
    reshapes it into a two dimensional numpy array.

    Parameters
    ----------
    arr : array-like/numpy array

    Returns
    ----------
    two dimensional numpy array.

    Examples
    --------
    >>> a = np.array([[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
    >>> a_new = reshape(a)
    >>> np.shape(a_new)
    (2, 6)

    >>> b = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    >>> b_new = reshape(b)
    >>> np.shape(b_new)
    (2, 4)

    >>> c = np.array([[[-4, 0.2], [0.5, 2.3]], [[5.6, 0], [0, 1.3]]])
    >>> c_new = reshape(c)
    >>> np.shape(c_new)
    (2, 4)
    """

    return np.reshape(arr, (np.shape(arr)[0],
                            np.shape(arr)[1]*np.shape(arr)[2]))


def reshape_all_datasets(a, b, c, d):
    """
    Reshapes four 3D numpy arrays into four 2D numpy arrays.

    Parameters
    ----------
    a, b, c, d : array-like/numpy arrays

    Returns
    ----------
    four 2D numpy arrays.

    Examples
    --------
    >>> a = np.array([[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
    >>> b = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    >>> c = np.array([[[-4, 0.2], [0.5, 2.3]], [[5.6, 0], [0, 1.3]]])
    >>> d = np.array([[[-2, 4], [1, 0]], [[0, 0], [0, 0]]])
    >>> a_new, b_new, c_new, d_new = reshape_all_datasets(a, b, c, d)
    >>> np.shape(b_new)
    (2, 4)
    """

    return reshape(a), reshape(b), reshape(c), reshape(d)


def make_sequential(data):
    '''
    Takes as input the raw wildfire data and returns the data in a sequential
    format, in the form of two arrays. The first is an array of "previous
    days", the second is an array of "next days". This is used in the
    prediction part of the project to train our models

    Usage:
    -----
    The function returns two arrays, so in the case of test data, the
    following is the usage:

    train_X, train_y = make_sequential(train_data)

    where train_data is the raw wildfire data as loaded by the loaddata
    function

    ***
    Important to look at required data shape in data parameter below
    ***

    Parameters:
            data (np.array): A numpy array fo the wildfire data. In our case
                             this would need to be either train or test data.
                             ***
                             --Required shape--
                             * (simulations, days, x, y)
                             ***

    Returns:
            data_X (np.array): An array of all previous days. The shape of this
                               array will be (n_sims * (days-1))

            data_y (np.array): An array of all next days. The shape of this
                               array will be (n_sims * (days-1))
    '''

    # Calculating the number of "next days", i.e. how
    # many T and T+1 pairs we have
    n_nextdays = data.shape[0] * (data.shape[1] - 1)

    # Reshaping the data to flatten it, as is needed by our basic
    # neural network
    data = data.reshape((data.shape[0],
                         data.shape[1],
                         data.shape[2] * data.shape[3]))

    # Creating empty arrays to put our sequential data in to
    data_X = np.empty((n_nextdays, data.shape[2]))
    data_y = np.empty((n_nextdays, data.shape[2]))

    # Looping through raw data and adding it new sequential arrays
    for i in range(data.shape[0]):
        for next_day in range(data.shape[1] - 1):
            data_X[i*3 + next_day] = data[i][next_day]
            data_y[i*3 + next_day] = data[i][next_day+1]

    return data_X, data_y


if __name__ == "__main__":
    import doctest
    doctest.testmod()

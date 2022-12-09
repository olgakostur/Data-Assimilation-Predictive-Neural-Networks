from numpy.linalg import inv
import numpy as np

__all__ = ['update_prediction', 'KalmanGain', 'mse',
           'assimilate', 'covariance_diagonal_only']


def update_prediction(x, K, H, y):
    """
    Given K and H an n x n matrix and x, y a numpy array of size n
    and returns an array of size n with the updated prediction

    Parameters
    ----------
    x : np.array
        'n' array
    y : np.array
        'n' array
    K : np.array or list of lists
        'n x n' array
    H : np.array or list of lists
        'n x n' array

    Returns
    ----------
    np.array of size n

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 3, 9, 5])
    >>> K = np.identity(5)
    >>> H = np.identity(5)
    >>> updated_data = update_prediction(x, K, H, y)
    >>> updated_data
    array([1., 2., 3., 9., 5.])
    """
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    """
    Given B, H and R which are symmetric n x n matrix where
    n is the number of latent space, return
    kalman gain n x n matrix

    Parameters
    ----------
    B : np.array or list of lists
        'n x n' array
    H : np.array or list of lists
        'n x n' array
    R : np.array or list of lists
        'n x n' array

    Returns
    ----------
    np.array 'n x n' (matrix)

    Examples
    --------
    >>> nNodes = 3
    >>> B = np.identity(nNodes)
    >>> H = np.identity(nNodes)
    >>> R = np.identity(nNodes)
    >>> k1 = KalmanGain(B, H, R)
    >>> k1
    array([[0.5, 0. , 0. ],
           [0. , 0.5, 0. ],
           [0. , 0. , 0.5]])
    >>> nNodes = 99
    >>> B = np.identity(nNodes)
    >>> H = np.identity(nNodes)
    >>> R = np.identity(nNodes)
    >>> k2 = KalmanGain(B, H, R)
    >>> np.shape(k2)
    (99, 99)
    """
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def mse(y_obs, y_pred):
    """
    Given y_obs, y_pred a numpy array of size n
    return the mse between y_obs and y_pred

    Parameters
    ----------
    y_obs : np.array
            'n' array
    y_pred : np.array
            'n' array

    Returns
    ----------
    mse value if the dimension of input matches

    Examples
    --------
    >>> y_obs = np.array([1,2,3,4,5])
    >>> y_pred = np.array([1,2,3,9,5])
    >>> mse_val = mse(y_obs, y_pred)
    >>> mse_val
    5.0
    """
    if np.shape(y_obs) == np.shape(y_pred):
        return np.square(np.subtract(y_obs, y_pred)).mean()
    else:
        return "incorrect dimensions"


def assimilate(B, H, R, mod_comp, sat_comp):
    """
    Given an n x n matrix of B, H, R and
    model/satellite compressed data as a numpy array of size n
    return the updated prediction as an array of size n

    Parameters
    ----------
    mod_comp : np.array
            'n' array
    sat_comp : np.array
            'n' array
    B : np.array or list of lists
        'n x n' array
    H : np.array or list of lists
        'n x n' array
    R : np.array or list of lists
        'n x n' array

    Returns
    ----------
    np.array of size n

    Examples
    --------
    >>> B = np.identity(1)
    >>> H = np.identity(1)
    >>> R = np.identity(1)
    >>> mod_comp = np.array(list(range(1)))
    >>> sat_comp = np.array(list(range(1)))
    >>> assim = assimilate(B, H, R, mod_comp, sat_comp)
    >>> assim
    array([[[0.]]])
    """
    K = KalmanGain(B, H, R)
    updated_data_list = []
    for i in range(len(mod_comp)):
        # compute only the analysis
        updated_data = update_prediction(mod_comp[i], K, H, sat_comp[i])
        updated_data_list.append(updated_data)
    return np.array(updated_data_list)


def covariance_diagonal_only(matrix, latent_space):
    """
    Given an n x n matrix and a value for latent_space
    return the diagonal covariance matrix with entries
    ofdiagonals as 0

    Parameters
    ----------
    latent_space : integer
    matrix : np.array or list of lists
        'n x n' array

    Returns
    ----------
    np.array 'n x n' (matrix)

    Examples
    --------
    >>> mat = np.identity(4)
    >>> lat = 4
    >>> cov_d = covariance_diagonal_only(mat, lat)
    >>> cov_d
    array([[0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    """
    diag_arr = []
    for i in range(latent_space):
        diag_arr.append(np.cov(matrix[i]))
    return np.diag(diag_arr)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

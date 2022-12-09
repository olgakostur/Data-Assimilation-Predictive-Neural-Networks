import tools.dataassimilation as da
import numpy as np
import pytest


@pytest.mark.parametrize('x, K, H, y, res', [
    ([1, 2, 2, 4, 5], np.identity(5), np.identity(5),
     [1, 4, 3, 9, 5], [1, 4, 3, 9, 5]),
    (list(range(999)), np.identity(999), np.identity(999), list(range(999)),
     list(range(999))),
])
def test_update_prediction(x, K, H, y, res):
    """ Test the update_prediction function """
    result = da.update_prediction(x, K, H, y)
    assert np.isclose(res, result).all()


@pytest.mark.parametrize('B, H, R, res', [
    (np.identity(4), np.identity(4), np.identity(4), np.identity(4)*0.5),
    (np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
               [-6, 7, 11, 19], [-9, 7, 11, 19]]),
        np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
                  [-6, 7, 11, 19], [-10, 7, 11, 19]]),
        np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
                  [-6, 7, 11, 19], [-15, 7, 11, 19]]),
        np.array([[5.85425787, 0.17963744, -8.58438652, 5.12635517],
                  [-8.02207105, -0.10610285, 12.06770482, -7.35981733],
                  [4.75697065, 0.12818798, -7.01683206, 4.23641443],
                  [1.63345534, 0.00985118, -2.40919644, 1.49053122]])),
    # (np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
    #            [-6, 7, 11, 19], [-6, 7, 11, 19]]),
    #  np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
    #            [-6, 7, 11, 19], [-6, 7, 11, 19]]),
    #  np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
    #            [-6, 7, 11, 19], [-6, 7, 11, 19]]),
    #  "singular matrix")
])
def test_KalmanGain(B, H, R, res):
    """ Test the KalmanGain function """
    result = da.KalmanGain(B, H, R)
    assert np.isclose(res, result).all()


@pytest.mark.parametrize('y_obs, y_pred, res', [
    (np.array(list(range(99, -1, -1))),
        np.array(list(range(100))), 3333.0),
    (np.array(list(range(50))),
        np.array(list(range(50))), 0),
])
def test_mse(y_obs, y_pred, res):
    """ Test the mse function """
    result = da.mse(y_obs, y_pred)
    assert np.isclose(res, result).all()


@pytest.mark.parametrize('matrix, latent_space, res', [
    (np.identity(5), 5, np.identity(5)*0.2),
    (np.array([[1, 4, 5, 12], [-5, 8, 9, 0],
               [-6, 7, 11, 19], [-6, 7, 11, 19]]), 4,
        np.array([[21.66666667, 0, 0, 0], [0, 44.66666667, 0, 0],
                  [0, 0, 108.91666667, 0], [0, 0, 0, 108.91666667]])),
])
def test_covariance_diagonal_only(matrix, latent_space, res):
    """ Test the covariance_diagonal_only function """
    result = da.covariance_diagonal_only(matrix, latent_space)
    assert np.isclose(res, result).all()


@pytest.mark.parametrize('B, H, R, mod_comp, sat_comp, res', [
    (np.identity(2), np.identity(2), np.identity(2),
        np.array(list(range(2))), np.array(list(range(2))),
        np.array([[[0, 0],
                   [0, 0]],
                 [[1, 1.5],
                  [1.5, 1]]])),
    (np.identity(3), np.identity(3), np.identity(3),
        np.array(list(range(3))), np.array(list(range(3))),
        np.array([[[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]],
                 [[1, 1.5, 1.5],
                  [1.5, 1, 1.5],
                  [1.5, 1.5, 1]],
                 [[2, 3, 3],
                  [3, 2, 3],
                  [3, 3, 2]]])),
])
def test_assimilate(B, H, R, mod_comp, sat_comp, res):
    """ Test the covariance_diagonal_only function """
    result = da.assimilate(B, H, R, mod_comp, sat_comp)
    assert np.isclose(res, result).all()

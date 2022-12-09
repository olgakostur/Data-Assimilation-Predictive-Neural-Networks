import tools.loaddata as loaddata
import numpy as np
import pytest


@pytest.mark.parametrize('path, res', [
    ('tools/tests/testdata', np.array([np.array(list(range(99, -1, -1))),
                                       np.array(list(range(4, -1, -1)))],
                                      dtype=object)),
])
def test_load_data(path, res):
    """ Test the load data function """
    data = loaddata.load_data(path)
    assert np.isclose(data[0], res[0]).all()
    assert np.isclose(data[1], res[1]).all()


path1 = 'tools/tests/testdata/test1'
path2 = 'tools/tests/testdata/test2'
path3 = 'tools/tests/testdata/test3'
path4 = 'tools/tests/testdata/test4'


@pytest.mark.parametrize('path_train, path_test, path_back, path_obs, res', [
    (path1, path2, path3, path4,
     [loaddata.load_data(path1), loaddata.load_data(path2),
      loaddata.load_data(path3), loaddata.load_data(path4)]),
])
def test_load_all_data(path_train, path_test, path_back, path_obs, res):
    """ Test the load all data function """
    data = loaddata.load_all_data(path_train, path_test, path_back, path_obs)
    assert np.isclose(data[0], res[0]).all()
    assert np.isclose(data[1], res[1]).all()
    assert np.isclose(data[1], res[1]).all()
    assert np.isclose(data[1], res[1]).all()


@pytest.mark.parametrize('arr, res', [
    (np.array([[[-4, 0.2], [0.5, 2.3]], [[5.6, 0], [0, 1.3]]]),
     np.array([[-4, 0.2, 0.5, 2.3], [5.6, 0, 0, 1.3]])),
    (np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
     np.array([[1, 0, 0, 1], [1, 0, 0, 1]])),
])
def test_reshape(arr, res):
    """ Test the load data function """
    data = loaddata.reshape(arr)
    assert np.isclose(data, res).all()


a = np.array([[[4, 99], [2, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
b = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
c = np.array([[[-4, 0.2], [0.5, 2.3]], [[5.6, 0], [0, 1.3]]])
d = np.array([[[-2, 4], [1, 0]], [[0, 0], [0, 0]], [[2, 3], [4, 5]]])

a_res = np.array([[4, 99, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
b_res = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
c_res = np.array([[-4, 0.2, 0.5, 2.3], [5.6, 0, 0, 1.3]])
d_res = np.array([[-2, 4, 1, 0], [0, 0, 0, 0], [2, 3, 4, 5]])


@pytest.mark.parametrize('a, b, c, d, res', [
    (a, b, c, d, [a_res, b_res, c_res, d_res]),
])
def test_reshape_all_datasets(a, b, c, d, res):
    """ Test the load data function """
    data = loaddata.reshape_all_datasets(a, b, c, d)
    assert np.isclose(data[0], res[0]).all()
    assert np.isclose(data[1], res[1]).all()
    assert np.isclose(data[2], res[2]).all()
    assert np.isclose(data[3], res[3]).all()


# Preparing data to test the make_sequential function

input_data_a = np.array([[[[7, 9],
                           [4, 2]],
                          [[1, 3],
                           [9, 9]],
                          [[7, 5],
                           [8, 9]],
                          [[5, 0],
                           [6, 4]]],
                         [[[5, 6],
                           [9, 1]],
                          [[9, 3],
                           [7, 1]],
                          [[5, 5],
                           [6, 4]],
                          [[6, 4],
                           [6, 8]]]])

input_data_b = np.array([[[[0.78021524, 0.11179494],
                           [0.57067647, 0.56661968]],
                          [[0.14209513, 0.98555792],
                           [0.59661704, 0.00323865]],
                          [[0.25913959, 0.21823557],
                           [0.20189582, 0.84043542]],
                          [[0.7928326, 0.99321167],
                           [0.34414127, 0.34451034]]],
                         [[[0.25784196, 0.53364292],
                           [0.17069344, 0.98421263]],
                          [[0.63197071, 0.86528964],
                             [0.42380213, 0.29000369]],
                          [[0.8123267, 0.30441817],
                             [0.09070643, 0.61544304]],
                          [[0.25751664, 0.55248369],
                             [0.65524213, 0.31038634]]]])

data_X_e_a = np.array([[7., 9., 4., 2.],
                       [1., 3., 9., 9.],
                       [7., 5., 8., 9.],
                       [5., 6., 9., 1.],
                       [9., 3., 7., 1.],
                       [5., 5., 6., 4.]])

data_X_e_b = np.array([[0.78021524, 0.11179494, 0.57067647, 0.56661968],
                       [0.14209513, 0.98555792, 0.59661704, 0.00323865],
                       [0.25913959, 0.21823557, 0.20189582, 0.84043542],
                       [0.25784196, 0.53364292, 0.17069344, 0.98421263],
                       [0.63197071, 0.86528964, 0.42380213, 0.29000369],
                       [0.8123267, 0.30441817, 0.09070643, 0.61544304]])

data_y_e_a = np.array([[1., 3., 9., 9.],
                      [7., 5., 8., 9.],
                      [5., 0., 6., 4.],
                      [9., 3., 7., 1.],
                      [5., 5., 6., 4.],
                      [6., 4., 6., 8.]])

data_y_e_b = np.array([[0.14209513, 0.98555792, 0.59661704, 0.00323865],
                       [0.25913959, 0.21823557, 0.20189582, 0.84043542],
                       [0.7928326, 0.99321167, 0.34414127, 0.34451034],
                       [0.63197071, 0.86528964, 0.42380213, 0.29000369],
                       [0.8123267, 0.30441817, 0.09070643, 0.61544304],
                       [0.25751664, 0.55248369, 0.65524213, 0.31038634]])


@pytest.mark.parametrize('input_data, data_X_e, data_y_e', [
    ([input_data_a, data_X_e_a, data_y_e_a]),
    ([input_data_b, data_X_e_b, data_y_e_b]),
])
def test_make_sequential(input_data, data_X_e, data_y_e):
    """ Test the make sequential function """
    data_X, data_y = loaddata.make_sequential(input_data)
    assert np.isclose(data_X, data_X_e).all()
    assert np.isclose(data_y, data_y_e).all()

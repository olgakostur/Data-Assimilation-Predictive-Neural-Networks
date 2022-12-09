import tools.visualisation as visualisation
import numpy as np
import pytest

# Testing with array containing only one simulation
input_data_a = np.array([[[[0.93836153, 0.25310585],
                           [0.84305321, 0.06995121]],
                          [[0.6884795, 0.46378882],
                           [0.06515325, 0.74538271]],
                          [[0.34811392, 0.4970604],
                           [0.99769378, 0.45690089]],
                          [[0.79056227, 0.75337038],
                           [0.19168457, 0.55392034]]]])

# Testing with array that has more than one sim, and choosing
# another colour map
input_data_b = np.array([[[[6, 7],
                           [7, 8]],
                          [[9, 0],
                           [7, 8]],
                          [[7, 0],
                           [6, 7]],
                          [[4, 2],
                           [9, 1]]],
                         [[[7, 9],
                           [7, 7]],
                          [[3, 1],
                             [7, 8]],
                          [[8, 3],
                             [0, 3]],
                          [[8, 8],
                             [0, 8]]],
                         [[[7, 5],
                           [8, 7]],
                          [[4, 7],
                             [6, 6]],
                          [[2, 4],
                             [7, 8]],
                          [[7, 7],
                             [3, 1]]]])


@pytest.mark.parametrize('data, simulation, cmap', [
    (input_data_a, 0, 'viridis'),
    (input_data_b, 1, 'hot'),
])
def test_create_slider(data, simulation, cmap):
    """ Test the load data function """
    visualisation.create_slider(data, simulation, cmap)


# Testing with sequential data
input_data_c = np.array([[[[5, 8],
                           [6, 3]],
                          [[3, 6],
                           [3, 8]],
                          [[2, 4],
                           [0, 6]]]])


@pytest.mark.parametrize('data, simulation, title, sequential', [
    (input_data_a, 0, '', False),
    (input_data_b, 1, 'adding title', False),
    (input_data_c, 0, '', True),
])
def test_plot_simulation_data(data, simulation, title, sequential):
    """ Test the load data function """
    visualisation.plot_simulation_data(data, simulation, title, sequential)

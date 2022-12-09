#######
creek
#######

A dataassimilation routine
------------------------------

This package implements dataassimilation [1]_ for :obj:`numpy.ndarray` objects, along with hand-written matrix multiplication.

See :`tools` folder for more information.

loaddata.py
------------------------------
.. automodule:: tools
  :members: loaddata

.. automodule:: tools.loaddata
  :members: load_data, load_all_data, reshape, reshape_all_datasets, make_sequential
  :noindex: loaddata

dataassimilation.py
------------------------------
.. automodule:: tools
  :members: dataassimilation

.. automodule:: tools.dataassimilation
  :members: update_prediction, KalmanGain, mse, assimilate, covariance_diagonal_only
  :noindex: dataassimilation

visualisation.py
------------------------------
.. automodule:: tools
  :members: visualisation

.. automodule:: tools.visualisation
  :members: create_slider, plot_data, plot_simulation_data, plot_pca_variance
  :noindex: visualisation

  
.. rubric:: References


import numpy as np
import matplotlib.pyplot as plt

import holoviews as hv
import hvplot.pandas  # noqa F401

__all__ = ['create_slider',
           'plot_data',
           'plot_simulation_data',
           'plot_pca_variance']


def create_slider(data, simulation=0, cmap='viridis'):
    '''
    Returns a slider holoview object which allows the user to view the
    progression of the wildfire spread.

    Usage:
    Simply calling the function will return the slider plot.

    ***
    Important to look at required data shape in data parameter below
    ***

    Parameters:
            data (np.array): A numpy array for the wildfire data. In our case
                             this would need to be either train, test,
                             observation or background data.
                             ***
                             --Required shape--
                             * For train and testdata: (sims, days, x, y)
                             * For obs or background data: (days, x, y)
                             ***

            simulation (int): Only needs to be provided if using train or test
                              data. It selects which simulation to run (of
                              which there are 300 in the train data, and 75 in
                              the test data). Defaulted to 0.

            cmap (str): Used to specify the colour used in the wildfire plot
                        images. This has to be a string that matches one of
                        the values listed here:
                        https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html
                        Default value is 'viridis' - the same is the matplotlib
                        imshow default.

    Returns:
            hmap (HoloMap object): A Holomap object of the slider. Needs to be
                                   assigned to a variable and run separatelky
                                   in order to be displayed in Jupyter
                                   Notebooks - see the above usage.
    '''

    # Labelling the slider variable
    kdims = 'Day'

    if len(data.shape) > 3:
        days = [2, 3, 4, 5]
        image_dict = {d: hv.Image(data[simulation][d-2],
                                  group='Simulation: {},'.format(simulation),
                                  label='Wildfire') for d in days}
        hmap = hv.HoloMap(image_dict,
                          kdims=kdims).opts(cmap=cmap)
    else:
        days = [2, 3, 4, 5, 6]
        image_dict = {d: hv.Image(data[d-2],
                                  label='Wildfire,') for d in days}
        hmap = hv.HoloMap(image_dict,
                          kdims=kdims).opts(cmap=cmap)
    return hmap


def plot_data(data_arr):
    '''
    This function produces a figure of 4 subplots, one for each
    day of a wildfire progression.

    Usage:
    Simply calling the function will return the slider plot. E.g:

        plot_data(data_arr)

    Parameters:
            data_arr (np.array): A numpy array fo the wildfire data. This
                                 will need to be either the background or
                                 satellite data.
    '''

    # Setting up figure for plots
    fig, axes = plt.subplots(1, 5, figsize=[20, 4])

    # Setting up list of days for iterating
    ts = [0, 1, 2, 3, 4, 5]

    # Looping through days and plotting each one
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(data_arr[ts[i]], vmin=0, vmax=1)
        ax.set_title("day" + str(ts[i] + 2))
        fig.colorbar(im, ax=ax)

    # No return from this function
    return None


def plot_simulation_data(data, simulation=0, title='', sequential=False):
    '''
    This function produces a figure of 4 subplots, one for each
    day of a wildfire simulation. The user can specify what data set to
    use (train or test), and which simulation from these datasets
    to show.

    Usage:
    Simply calling the function will return the slider plot. E.g:

        plot_simulation_data(train_data, simulation=42)

    ***
    Important to look at required data shape in data parameter below
    ***

    Parameters:
            data (np.array): A numpy array fo the wildfire data. In our case
                             this would need to be either train or test.
                             ***
                             --Required shape--
                             * For train and testdata: (sims, days, x, y)
                             ***

            simulation (int): Selects which simulation to run (of
                              which there are 300 in the train data, and 75 in
                              the test data). Defaulted to 0.

            title (str): Option to add a string to augment the title of the
                         resultant figure

            sequential (bool): Let the function know whether you are plotting
                               from a dataset that has a reduced amout of days.
                               I.e. sequential datasets that have been reshaped
                               for model training, or have been predicted using
                               our model.
    '''

    # Creating a list of the simulation days to iterate through

    # This logic gate takes as input the boolean "sequential", which lets
    # us know how many days are gonna be in the data set
    if sequential:
        days = [3, 4, 5]
    else:
        days = [2, 3, 4, 5]

    # Setting up the figure to add the plots to
    fig, axes = plt.subplots(1, len(days), figsize=[20, 4])

    # Creating a super title for the whole figure
    fig.suptitle(str(title) + " Wildfire spread for simulation "
                 + str(simulation),
                 fontsize=20)

    # Adjusting the positioning of the suptitle so it doesn't clash
    # with the subplot titles
    fig.subplots_adjust(top=0.8)

    # Looping through the days and plotting each one
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(data[simulation][days[i]-days[0]], vmin=0, vmax=1)
        ax.set_title("Day" + str(days[i]))

    return None


def plot_pca_variance(pca, n_components):
    '''
    Function that plots explained variance vs number of components for PCA
    analysis, with annotation of explaned variance for a provided number
    of components.

    Usage:
    Simply calling the function with the correct args will
    return the PCA variance plot. E.g.:

        plot_pca_variance(pca=pca, n_components=60)

    Parameters:
            pca (PCA object): An sklearn PCA object. The result of using
                              sklearn.decomposition.PCA on a fitting dataset.

            n_components (int): The number of components for the annotation
                                on the plot to show the explained variance for.
                                This can be a maximum of the total number of
                                components available in the PCA. This max can
                                be found using PCA.components_.shape[0].

    Returns:
            No return variable, just the displaying of the plot.
    '''

    p = pca.explained_variance_ratio_.cumsum()[n_components]
    cumsum_eig = np.cumsum(pca.explained_variance_ratio_)
    d = pca.components_.shape[0]

    fig, axes = plt.subplots(figsize=(20, 10))

    axes.plot(np.arange(1, 1+len(cumsum_eig)), cumsum_eig, linewidth=3)
    axes.set_xlabel("Dimensions",
                    fontsize=24)
    axes.set_ylabel("Explained Variance",
                    fontsize=20)
    axes.set_title("PCA Analysis: Plot of explained variance vs number of components",  # noqa E501
                   fontsize=24)
    axes.set_ylim([cumsum_eig[0], 1.05])
    axes.plot([n_components, n_components], [0, p], "k:")
    axes.plot([0, n_components], [p, p], "k:")
    axes.plot(n_components, p, "ko")
    axes.annotate(f'Keeping {n_components} features \n retains {p*100:.2f}% of the variance',  # noqa E501
                  xy=(n_components, p),
                  xytext=(0.5*d, 0.7),
                  ha='center',
                  va='center',
                  arrowprops=dict(arrowstyle="->"),
                  fontsize=20)

    axes.grid(True)

    return None

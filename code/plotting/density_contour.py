import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, uselog=False, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.6827))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.9545))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.9973))
    four_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.9999366575))
    levels = np.sort(np.array([one_sigma, two_sigma, three_sigma, four_sigma]))

    #print(levels)
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    if (uselog):
        X_plot = 10**X
        Y_plot = 10**Y
    else:
        X_plot = X
        Y_plot = Y
    
    if ax == None:
        contour = plt.contour(X_plot, Y_plot, Z, levels=levels, origin="lower", **contour_kwargs)
    else:
        contour = ax.contour(X_plot, Y_plot, Z, levels=levels, origin="lower", **contour_kwargs)

    return contour

def test_density_contour():
    norm = np.random.normal(10., 15., size=(12540035, 2))
    density_contour(norm[:,0], norm[:,1], 100, 100)
    plt.show()

#test_density_contour()
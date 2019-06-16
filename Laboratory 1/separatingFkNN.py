import matplotlib.pyplot as plt
import numpy as np
from kNNClassify import *
from matplotlib.mlab import griddata


def separatingFkNN(Xtr, Ytr, k):
    """Plots seprating function of a kNN classifier

    usage: separatingFkNN(Xtr=X, Ytr=Y, k=3)

    Xtr: The training points
    Ytr: The labels of the training points
    k : How many neighbours to use for classification

    Returns:
    nothing"""

    Ypred = kNNClassify(Xtr=Xtr, Ytr=Ytr, k=k, Xte=Xtr)

    x = Xtr[:, 0]
    y = Xtr[:, 1]
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    zi = griddata(x, y, Ypred, xi, yi, interp='linear')

    CS = plt.contour(xi, yi, zi, 15, linewidths=2, colors='k', levels=[0])
    # plot data points.
    plt.scatter(x, y, c=Ytr, marker='o', s=20, zorder=10)
    plt.xlim(x.min(), x.max())
    plt.ylim(x.min(), x.max())
    plt.title('Separating function')

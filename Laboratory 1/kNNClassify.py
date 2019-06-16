import numpy as np
from sqDist import *


def kNNClassify(Xtr, Ytr, k, Xte):
    """Classifies a set o test points given a training set and the number of neighbours to use

    usage:
    Ypred = kNNClassify(Xtr, Ytr, k=5, Xte)

    Arguments:
    Xtr: Training set matrix [nxd], each row is a point
    Ytr: Training set label array [n], values must be +1,-1
    k: Number of neighbours
    Xte: Test points

    Return
    Ypred: estimated test output
    """
    n_train = Xtr.shape[0]
    n_test = Xte.shape[0]

    if any(np.abs(Ytr) != 1):
        print("The values of Ytr should be +1 or -1.")
        return -1

    if k > n_train:
        print("k is greater than the number of points, setting k=n_train")
        k = n_train

    Ypred = np.zeros(n_test)

    dist = sqDist(Xte, Xtr)

    for idx in range(n_test):
        neigh_indexes = np.argsort(dist[idx, :])[:k]
        # print(Ytr[neigh_indexes])
        avg_neigh = np.mean(Ytr[neigh_indexes])
        # print(avg_neigh)
        Ypred[idx] = np.sign(avg_neigh)

    return Ypred

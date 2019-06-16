import numpy as np


def calcErr(T, Y):
    """Computes the average error given a true set of labels and computed labels

    usage: error = calcErr(T, Y)

    T: True labels of the test set
    Y: labels computed by the user, must belong to {-1,+1}
    """
    err = np.mean(np.sign(T) != np.sign(Y))
    return err

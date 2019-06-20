import numpy as np


def calcErr(T, Y):
    err = np.mean(np.sign(T) != np.sign(Y))
    return err

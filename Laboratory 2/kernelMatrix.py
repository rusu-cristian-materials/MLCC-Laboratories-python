import numpy as np
from sqDist import *


def kernelMatrix(x1, x2, param, kernel='linear'):
    '''Input:
    x1, x2: collections of points on which to compute the Gram matrix
    kernel: can be 'linear', 'polynomial' or 'gaussian'
    param: is [] for the linear kernel, the exponent of the polynomial kernel,
           or the variance for the gaussian kernel

    Output:
    k: Gram matrix'''
    if kernel == 'linear':
        k = np.dot(x1, np.transpose(x2))
    elif kernel == 'polynomial':
        k = np.power((1 + np.dot(x1, np.transpose(x2))), param)
    elif kernel == 'gaussian':
        k = np.exp(float(-1) / float((2 * param ** 2)) * sqDist(x1, x2))
    return k

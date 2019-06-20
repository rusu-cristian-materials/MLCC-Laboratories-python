import numpy as np
import numpy.linalg as la


def PCA(X, k):
    """Computes the first k eigenvectors, eigenvalues and projections of the
    data matrix X
    usage: V, d, X_proj = PCA(X, k)

    X is the dataset
    k is the number of components

    V is a matrix of the form [v_1, ..., v_k] where v_i is the i-th
    eigenvector
    d is the list of the first k eigenvalues
    X_proj is the projection of X on the linear space spanned by the
    eigenvectors in V"""

    mean = X.mean(axis=0)
    X_z = X - mean
    cov_mat = X_z.T.dot(X_z)
    U, d, V = la.svd(cov_mat)
    X_proj = X_z.dot(V[:, :k])

    return V, d, X_proj

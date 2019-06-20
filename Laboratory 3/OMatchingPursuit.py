import numpy as np


def OMatchingPursuit(X, Y, T):
    """ Computes a sparse representation of the signal using Orthogonal Matching Pursuit algorithm

    usage: w, r, I = OMatchingPursuit( X, Y, T)

    Arguments:
    X: input data
    Y: output labels
    T: number of iterations

    Returns:
    w: estimated coefficients
    r: residuals
    I: indices"""

    N, D = np.shape(X)

    # Initialization of residual, coefficient vector and index set I
    r = Y
    w = np.zeros(D)
    I = []

    for i in range(T):
        I_tmp = range(D)

        # Select the column of X which most "explains" the residual
        a_max = -1

        for j in I_tmp:
            a_tmp = ((r.T.dot(X[:, j])) ** 2) / (X[:, j].T.dot(X[:, j]))

            if a_tmp > a_max:
                a_max = a_tmp
                j_max = j

        # Add the index to the set of indexes
        if np.sum(I == j_max) == 0:
            I.append(j_max)

        # Compute the M matrix
        M_I = np.zeros((D, D))

        for j in I:
            M_I[j, j] = 1

        A = M_I.dot(X.T).dot(X).dot(M_I)
        B = M_I.dot(X.T).dot(Y)

        # Update estimated coefficients
        w = np.linalg.pinv(A).dot(B)

        # Update the residual
        r = Y - X.dot(w)

    return w, r, I

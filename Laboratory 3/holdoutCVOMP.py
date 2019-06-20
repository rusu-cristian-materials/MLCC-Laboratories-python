import numpy as np
from OMatchingPursuit import *
from calcErr import *


def holdoutCVOMP(X, Y, perc, nrip, intIter):
    """l, s, Vm, Vs, Tm, Ts = holdoutCVOMP(algorithm, X, Y, kernel, perc, nrip, intRegPar, intKerPar)
    X: the training examples
    Y: the training labels
    perc: fraction of the dataset to be used for validation
    nrip: number of repetitions of the test for each couple of parameters
    intIter: range of iteration for the Orthogonal Matching Pursuit

    Output:
    it: the number of iterations of OMP that minimize the classification
    error on the validation set
    Vm, Vs: median and variance of the validation error for each couple of parameters
    Tm, Ts: median and variance of the error computed on the training set for each couple
          of parameters

    intIter = 1:50;
    Xtr, Ytr = MixGauss(np.matrix([[0,1],[0,1]]),np.array([[0.5],[0.25]]),100);
    Xtr_noise =  0.01 * np.random.randn(200, 28);
    Xtr = np.concatenate((Xtr, Xtr_noise), axis=1)
    l, s, Vm, Vs, Tm, Ts = holdoutCVOMP(Xtr, Ytr, 0.5, 5, intIter);"""

    nIter = np.size(intIter)

    n = X.shape[0]
    ntr = int(np.ceil(n * (1 - perc)))

    tmn = np.zeros((nIter, nrip))
    vmn = np.zeros((nIter, nrip))

    for rip in range(nrip):
        I = np.random.permutation(n)
        Xtr = X[I[:ntr]]
        Ytr = Y[I[:ntr]]
        Xvl = X[I[ntr:]]
        Yvl = Y[I[ntr:]]

        iit = -1

        newIntIter = [x + 1 for x in intIter]
        for it in newIntIter:
            iit = iit + 1;
            w, r, I = OMatchingPursuit(Xtr, Ytr, it)
            tmn[iit, rip] = calcErr(Xtr.dot(w), Ytr)
            vmn[iit, rip] = calcErr(Xvl.dot(w), Yvl)

            #print('%-12s%-12s%-12s%-12s' % ('rip', 'Iter', 'valErr', 'trErr'))
            #print('%-12i%-12i%-12f%-12f' % (rip, it, vmn[iit, rip], tmn[iit, rip]))

    Tm = np.median(tmn, axis=1)
    Ts = np.std(tmn, axis=1)
    Vm = np.median(vmn, axis=1)
    Vs = np.std(vmn, axis=1)

    # one of the min removed to make it iterable
    # nonzero returns the indices of the elements that are non-zero
    row = np.nonzero(Vm <= min(Vm))
    # added to solve last index problem
    row = row[0]

    it = intIter[row[0]]

    return it, Vm, Vs, Tm, Ts

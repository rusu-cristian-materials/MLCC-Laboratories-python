from kNNClassify import *
from calcErr import *


def holdoutCVkNN(Xtr, Ytr, perc, n_rep, kernel, lam_list, kerpar_list=[]):
    """Performs holdout cross-validation for Kernel Ridge regression

    Arguments:
    Xtr: Training set matrix [nxd], each row is a point
    Ytr: Training set label array [n], values must be +1,-1
    perc: percentage of training set ot be used for validation
    n_rep: number of repetitions of the test for each couple of parameters
    kernel: the chosen kernel
    lam_list: list/array of regularization parameters to try
    kerpar_list: st/array of regularization parameters to try

    Returns:
    k: the value k in k_list that minimizes the mean of the validation error
    Vm, Vs: mean and variance of the validation error for each couple of parameters
    Tm, Tx: mean and variance of the error computed on the training set for each couple of parameters
    """

    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1

    if n_rep <= 0:
        print("Please supply a positive number of repetitions")
        return -1

    # Ensures that parameters are in a numpy array
    lam_list = np.array(k_list)
    num_lam = k_list.size

    if not kerpar_list:
        kerpar_list = [0]
    kerpar_list = np.array(kerpar_list)
    num_kerpar = kerpar_list.size

    n_tot = Xtr.shape[0]
    n_train = int(np.ceil(n_tot * (1 - perc / 100)))

    Tm = np.zeros(num_k)
    Ts = np.zeros(num_k)
    Vm = np.zeros(num_k)
    Vs = np.zeros(num_k)

    for lamdx, lam in enumerate(lam_list):
        for kerpardx, kerpar in enumerate(kerpar_list):
            for rip in range(n_rep):
                # Randombly select a subset of the training set
                rand_idx = np.random.choice(n_tot, size=n_tot, replace=False)

                X = Xtr[rand_idx[:n_train]]
                Y = Ytr[rand_idx[:n_train]]
                X_val = Xtr[rand_idx[n_train:]]
                Y_val = Ytr[rand_idx[n_train:]]

                # Compute the training error of the kNN classifier for the given value of k
                trError = calcErr(kNNClassify(X, Y, k, X), Y)
                Tm[kdx] = Tm[kdx] + trError
                Ts[kdx] = Ts[kdx] + trError ** 2

                # Compute the validation error of the kNN classifier for the given value of k
                valError = calcErr(kNNClassify(X, Y, k, X_val), Y_val)
                Vm[kdx] = Vm[kdx] + valError
                Vs[kdx] = Vs[kdx] + valError ** 2

    Tm = Tm / n_rep
    Ts = Ts / n_rep - Tm ** 2

    Vm = Vm / n_rep
    Vs = Vs / n_rep - Vm ** 2

    best_k_idx = np.argmin(Vm)
    k = k_list[best_k_idx]

    return k, Vm, Vs, Tm, Ts

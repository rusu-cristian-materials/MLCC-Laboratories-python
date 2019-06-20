from MixGauss import *
from separatingFkNN import *
from holdoutCVkNN import *
from flipLabels import *

np.random.seed(42)

######################### Section 1 #########################

## 1.A
help(MixGauss)

## 1.B
X, Y = MixGauss([[0, 0], [1, 1]], [0.5, 0.25], 50)

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("dataset 1")
axs[0, 0].scatter(X[:, 0], X[:, 1], s=70, c=Y, alpha=0.8)

## 1.C
Xtr, Ytr = MixGauss(means=[[0, 0], [0, 1], [1, 1], [1, 0]], sigmas=[0.3, 0.3, 0.3, 0.3], n=100)
axs[0, 1].set_title("dataset 2")
axs[0, 1].scatter(Xtr[:, 0], Xtr[:, 1], s=40, c=Ytr, alpha=0.8)

Ytr = 2*np.mod(Ytr, 2) - 1
axs[1, 0].set_title("dataset 3 (train dataset)")
axs[1, 0].scatter(Xtr[:, 0], Xtr[:, 1], s=40, c=Ytr, alpha=0.8)

## 1.D
Xts, Yts = MixGauss(means=[[0,0], [0,1], [1,1], [1,0]], sigmas=[0.3, 0.3, 0.3, 0.3], n=30)
Yts = 2*np.mod(Yts, 2) - 1
axs[1, 1].set_title("dataset 4 (test dataset)")
axs[1, 1].scatter(Xts[:, 0], Xts[:, 1], s=40, c=Yts, alpha=0.8)

plt.tight_layout()
plt.savefig('figure_1.png', dpi=100)

######################### Section 2 #########################

## 2.A
help(kNNClassify)

## 2.B
# number of neighbours to consider
k = 3

## 2.C1-2.C3
# perform kNN classification
Ypred = kNNClassify(Xtr, Ytr, k, Xte=Xts)

fig, axs = plt.subplots(2, 1)
axs[0].set_title('kNN prediction with k = 3')
axs[0].scatter(Xts[:, 0], Xts[:, 1], s=100, c=Yts, alpha=0.3, marker='o', edgecolor='black')
axs[0].scatter(Xts[:, 0], Xts[:, 1], s=30, c=Ypred, alpha=1, marker='^')

error = np.mean(Ypred != Yts)
print('Error rate for kNN, with k = ' + str(k) + ' is ' + str(error) + '.')

help(separatingFkNN)

separatingFkNN(Xtr, Ytr, k=10)

plt.tight_layout()
plt.savefig('figure_2.png', dpi=100)

######################### Section 3 #########################

help(holdoutCVkNN)

fig, axs = plt.subplots(2, 2)

## 3.A
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
k_best, Vm, Vs, Tm, Ts = holdoutCVkNN(Xtr=Xtr, Ytr=Ytr, perc=20, n_rep=10, k_list=k_list)
axs[0, 0].set_title('hold out, initial dataset')
axs[0, 0].set_xlabel('k')
axs[0, 0].set_ylabel('error')
axs[0, 0].plot(k_list, Tm)
axs[0, 0].plot(k_list, Vm)
axs[0, 0].legend(['Test', 'Validation'])

## 3.B
p = 5
perc = 20
Ytr_noisy = flipLabels(Ytr, p)
k_best_noisy, Vm, Vs, Tm, Ts = holdoutCVkNN(Xtr=Xtr, Ytr=Ytr_noisy, perc=perc, n_rep=10, k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9])
axs[0, 1].set_title('hold out, noisy p = ' + str(p) + ', perc = ' + str(perc))
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('error')
axs[0, 1].plot(k_list, Tm)
axs[0, 1].plot(k_list, Vm)
axs[0, 1].legend(['Test', 'Validation'])

## 3.C
k = [5]
p_list = [1, 5, 10, 15, 20, 25]
perc = 20
n_rep = 10
errors_training = np.zeros(len(p_list))
errors_validation = np.zeros(len(p_list))
for pdx, p in enumerate(p_list):
    Ytr_noisy = flipLabels(Ytr, p)
    k_best_noisy, Vm, Vs, Tm, Ts = holdoutCVkNN(Xtr=Xtr, Ytr=Ytr_noisy, perc=perc, n_rep=n_rep, k_list=k)
    errors_training[pdx] = float(Tm)
    errors_validation[pdx] = float(Vm)

axs[1, 0].set_title('hold out, noisy, k = ' + str(k[0]) + ', nrep = ' + str(n_rep) + ', perc = ' + str(perc))
axs[1, 0].set_xlabel('p')
axs[1, 0].set_ylabel('error')
axs[1, 0].plot(p_list, errors_training)
axs[1, 0].plot(p_list, errors_validation)
axs[1, 0].legend(['Test', 'Validation'])

### do the same for perc and n_rep

## 3.D
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
p  = 5
Yts_noisy = flipLabels(Yts, p)

errors = np.zeros(len(k_list))
for kdx, k in enumerate(k_list):
    Ypred = kNNClassify(Xtr, Ytr, k, Xte=Xts)
    errors[kdx] = calcErr(Yts_noisy, Ypred)

axs[1, 1].set_title('test error, p = ' + str(p))
axs[1, 1].set_xlabel('k')
axs[1, 1].set_ylabel('error')
axs[1, 1].plot(k_list, errors)

plt.tight_layout()
plt.savefig('figure_3.png', dpi=100)

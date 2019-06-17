from MixGauss import *
from separatingFKernRLS import *
import matplotlib.pyplot as plt
from flipLabels import *
from two_moons import *
from holdoutCVKernRLS import *

np.random.seed(42)

######################### Section 1 #########################

## 1.A
Xtr, Ytr = MixGauss(means=[[0,0],[1,1]], sigmas=[0.5, 0.3], n=100)
Xts, Yts = MixGauss(means=[[0,0],[1,1]], sigmas=[0.5, 0.3], n=100)
Ytr = 2*np.mod(Ytr, 2)-1
Yts = 2*np.mod(Yts, 2)-1

## 1.B
fig, axs = plt.subplots(2, 1)
sigma = 0.8
lam = 1E-5
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma=sigma, lam=lam)
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma=sigma, Xte=Xts, axs=axs[0])
axs[0].set_title('kNN prediction with sigma = ' + str(sigma) + ', lambda = ' + str(lam))

sigma = 0.05
lam = 1E-5
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma=sigma, lam=lam)
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma=sigma, Xte=Xts, axs=axs[1])
axs[1].set_title('kNN prediction with sigma = ' + str(sigma) + ', lambda = ' + str(lam))

plt.tight_layout()
plt.savefig('figure_1.png', dpi=100)

## 1.C and 1.D
fig, axs = plt.subplots(1,  1)
p = 10
Ytr_noisy = flipLabels(Ytr, p)
sigma = 0.1
lam = 1E-1
c = regularizedKernLSTrain(Xtr, Ytr_noisy, 'gaussian', sigma=sigma, lam=lam)
separatingFKernRLS(c, Xtr, Ytr_noisy, 'gaussian', sigma=sigma, Xte=Xts, axs=axs)
axs.set_title('kNN prediction, noisy, with sigma = ' + str(sigma) + ', lambda = ' + str(lam))

plt.tight_layout()
plt.savefig('figure_2.png', dpi=100)

## 1.D
fig, axs = plt.subplots(2,  1)
p = 10 # percentage to flip
Xtr, Ytr, Xts, Yts = two_moons(100, p)
axs[0].scatter(Xtr[:, 0], Xtr[:, 1], s=50, c=Ytr)
axs[0].set_title('noisy train dataset')
axs[1].scatter(Xts[:, 0], Xts[:, 1], s=50, c=Yts)
axs[1].set_title('noisy test dataset')

plt.tight_layout()
plt.savefig('figure_3.png', dpi=100)

## 1.E
fig, axs = plt.subplots(1,  1)
sigma = 0.1
lam = 1E-5
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma=sigma, lam=lam)
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma=sigma, Xte=Xts, axs=axs)
axs.set_title('kNN prediction with sigma = ' + str(sigma) + ', lambda = ' + str(lam))

plt.tight_layout()
plt.savefig('figure_4.png', dpi=100)

######################### Section 2 #########################

## 2.C
kerpar_list = [0.5]
lam_list = [10, 7, 5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00001, 0.000001]
nrip = 5
perc = 50

l, s, vm, vs, tm, ts = holdoutCVKernRLS(Xtr, Ytr, perc, nrip, 'gaussian', lam_list, kerpar_list)

fig, axs = plt.subplots(1,  1)
plt.semilogx(lam_list, tm, 'r')
plt.semilogx(lam_list, vm, 'b')

plt.legend(['Training error', 'Validation error'])
plt.grid()

plt.tight_layout()
plt.savefig('figure_5.png', dpi=100)

## 2.E
fig, axs = plt.subplots(1,  1)
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma=s[0], lam=l[0])
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma=s[0], Xte=Xts, axs=axs)
axs.set_title('kNN prediction with best sigma = ' + str(s[0]) + ', best lambda = ' + str(l[0]))
sigma_best_gaus = s[0]

plt.tight_layout()
plt.savefig('figure_6.png', dpi=100)

######################### Section 3 #########################

## 3.C
kerpar_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
lam_list = [5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001]
nrip = 5
perc = 50

l, s, vm, vs, tm, ts = holdoutCVKernRLS(Xtr, Ytr, perc, nrip, 'polynomial', lam_list, kerpar_list)

fig, axs = plt.subplots(1,  1)
c = regularizedKernLSTrain(Xtr, Ytr, 'polynomial', sigma=s[0], lam=l[0])
separatingFKernRLS(c, Xtr, Ytr, 'polynomial', sigma=s[0], Xte=Xts, axs=axs)
axs.set_title('kNN prediction with poly best degree = ' + str(s[0]) + ', best lambda = ' + str(l[0]))
sigma_best_poly = s[0]
print('Best kernel polynomial has degree ' + str(sigma_best_poly))

plt.tight_layout()
plt.savefig('figure_7.png', dpi=100)

## 3.D
K_gaussian = kernelMatrix(Xtr, Xtr, sigma_best_gaus, 'gaussian')
K_polynomial = kernelMatrix(Xtr, Xtr, sigma_best_poly, 'polynomial')

D_gaus = np.linalg.eig(K_gaussian)
D_poly = np.linalg.eig(K_polynomial)

fig, axs = plt.subplots(1,  1)
xx = np.linspace(0, 100, 100)
axs.semilogy(xx, np.sort(D_gaus[0]))
axs.semilogy(xx, np.sort(D_poly[0]))
axs.legend(['Gaussian eigenvalues', 'Polynomial eigenvalues'])

plt.tight_layout()
plt.savefig('figure_8.png', dpi=100)

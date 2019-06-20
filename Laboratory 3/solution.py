import numpy as np
from MixGauss import *
import matplotlib.pyplot as plt
from PCA import *
from holdoutCVOMP import *

np.random.seed(42)

######################### Section 1 #########################

n = 100
d = 30

## 1.A
Xtr, Ytr = MixGauss(means=[[0, 0], [1, 1]], sigmas=[[0.7], [0.7]], n=n)
Ytr = 2*np.mod(Ytr, 2)-1

Xts, Yts = MixGauss(means=[[0, 0], [1, 1]], sigmas=[[0.7], [0.7]], n=n)
Yts = 2*np.mod(Yts, 2)-1

## 1.B
fig, axs = plt.subplots(1, 1)
plt.scatter(Xtr[:, 0], Xtr[:, 1], s=30, c=np.squeeze(Ytr), alpha=0.8)
plt.scatter(Xts[:, 0], Xts[:, 1], s=30, c=np.squeeze(Ytr), alpha=0.1)
plt.title('train and test datasets')

plt.tight_layout()
plt.savefig('figure_1.png', dpi=100)

## 1.C
sigma_noise = 0.01

Xtr_noise = sigma_noise * np.random.randn(2*n, d-2)
Xts_noise = sigma_noise * np.random.randn(2*n, d-2)

Xtr = np.concatenate((Xtr, Xtr_noise), axis=1)
Xts = np.concatenate((Xts, Xts_noise), axis=1)

######################### Section 2 #########################

## 2.A
V, D, X_proj = PCA(Xtr, 3)

## 2.B
fig, axs = plt.subplots(1, 1)
plt.scatter(X_proj[:, 0], X_proj[:, 1], s=30, c=np.squeeze(Ytr), alpha=0.8)
plt.title('train dataset projected on first 2 components')

plt.tight_layout()
plt.savefig('figure_2.png', dpi=100)

## 2.D
print('Eigenvalues: ' + str(np.sqrt(D[:10])))

fig, axs = plt.subplots(1, 1)
plt.scatter(range(d), abs(V[:, 0]), s=30, alpha=0.8)
plt.title('Eigenvector of highest eigenvalue')

plt.tight_layout()
plt.savefig('figure_3.png', dpi=100)

######################### Section 3 #########################

## 3.A normalize train and test datasets
m = np.mean(Xtr, axis=0)
s = np.std(Xtr, axis=0)

Xtr = (Xtr - m) / s
Xts = (Xts - m) / s

## 3.B and 3.C
intIter = range(2, 10)
perc = 0.80
nrip = 20
it_best, Vm, Vs, Tm, Ts = holdoutCVOMP(Xtr, Ytr, perc, nrip, intIter)

w, r, I = OMatchingPursuit(Xtr, Ytr, it_best)
Ypred = np.sign(Xts.dot(w))
error = calcErr(Yts, Ypred)
print('Classification error with OMP is ' + str(error))

fig, axs = plt.subplots(1, 1)
plt.scatter(range(d), w, s=30, alpha=0.8)
plt.title('the w vector');

plt.tight_layout()
plt.savefig('figure_4.png', dpi=100)

## 3.D
fig, axs = plt.subplots(1, 1)
plt.plot(intIter, Tm)
plt.plot(intIter, Vm)
plt.legend(['Training error', 'Validation error'])
plt.xlabel('number of iterations for OMP')
plt.ylabel('error')

plt.tight_layout()
plt.savefig('figure_5.png', dpi=100)

######################### Section 4 #########################

## 4.B
# generate a fresh dataset
Xtr, Ytr = MixGauss([[0, 0], [1, 1]], [0.7, 0.7], 100)
Xts, Yts = MixGauss([[0, 0], [1, 1]], [0.7, 0.7], 100)
Ytr[Ytr==0] = -1
Yts[Yts==0] = -1

# perform the singular value decomposition, i.e., dimensionality reduction
V, D, X_proj = PCA(Xtr, 1)

Z = np.zeros((2*n, 2))

# compute the projections, one by one
for i in range(2*n):
    Z[i, :] = V[0, :].dot(float(X_proj[i]))

fig, axs = plt.subplots(1, 1)
plt.scatter(Xtr[:, 0], Xtr[:, 1], s=30, c=np.squeeze(Ytr), alpha=0.8)
plt.scatter(Z[:, 0], Z[:, 1], s=30, c=np.squeeze(Ytr), alpha=0.2)
plt.title('original and projected datasets')

plt.tight_layout()
plt.savefig('figure_6.png', dpi=100)

### task: pick a point and its projection and draw a line between them
### task: perform kNN (from Lab 1) in the projected space

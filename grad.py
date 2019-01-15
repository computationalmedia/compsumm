import scipy as sp
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Tuple, Callable, Any, List
from functools import partial
from profilehooks import profile
from math import ceil
import time

### MMD-Grad ###
# @profile
def ekxs_cost_grad(A: np.ndarray, X: np.ndarray, gamma: float, **kwargs: Any) -> Tuple[float, np.ndarray]:
	n, d = X.shape
	m = A.shape[0] // d
	A = A.reshape(m, d)
	Kxa = pairwise_kernels(X, A, metric = "rbf", gamma = gamma)
	cost = -2 * Kxa.mean()
	Grad = np.zeros((m, d))
	for l in range(A.shape[0]):
		Grad[l] -= ((X - A[l]).T * Kxa[:,l]).T.mean(axis = 0)
	return cost, 4 * gamma / m * Grad.flatten()

def mmd_cost_grad(A: np.ndarray, X: np.ndarray, gamma: float, **kwargs: Any) -> Tuple[float, np.ndarray]:
	n, d = X.shape
	m = A.shape[0] // d
	A = A.reshape(m, d)
	Kxa = pairwise_kernels(X, A, metric = "rbf", gamma = gamma)
	Kaa = pairwise_kernels(A, A, metric = "rbf", gamma = gamma)
	cost = -2 * Kxa.mean() + Kaa.mean()
	Grad = np.zeros((m, d))
	for l in range(A.shape[0]):
		Grad[l] -= ((X - A[l]).T * Kxa[:,l]).T.mean(axis = 0)
		Grad[l] += ((A - A[l]).T * Kaa[:,l]).T.mean(axis = 0)
	return cost, 4 * gamma / m * Grad.flatten()

def ekxs_cost(A: np.ndarray, X: np.ndarray, gamma: float, **kwargs: Any) -> Tuple[float, np.ndarray]:
	n, d = X.shape
	return -2 * pairwise_kernels(X, A.reshape(A.shape[0] // d, d), metric = "rbf", gamma = gamma).mean()

def mmd_cost(A: np.ndarray, X: np.ndarray, gamma: float, **kwargs: Any) -> Tuple[float, np.ndarray]:
	n, d = X.shape
	m = A.shape[0] // d
	A = A.reshape(m, d)
	Kxa = pairwise_kernels(X, A, metric = "rbf", gamma = gamma)
	Kaa = pairwise_kernels(A, A, metric = "rbf", gamma = gamma)
	return -2 * Kxa.mean() + Kaa.mean()

#### MMD_grad with labels: different thank other data
# @profile
def mmd_cost_grad_labels(A: np.ndarray, X: np.ndarray, y: np.ndarray, 
					gamma: float, gamma2: float, lambdaa: float = 0.0, diff = "data", **kwargs: Any) -> Tuple[float, np.ndarray]:
	cost = 0.0
	n, d = X.shape
	m = A.shape[0] // d
	A = A.reshape(m, d)
	Grad = np.zeros((m, d))
	classes = sorted(set(y))
	perclass = m // len(classes)

	for c, k in enumerate(classes):
		Ac = A[c*perclass:(c+1)*perclass, :].flatten()
		Xc = X[np.where(y == k)[0], :]
		Xk = X[np.where(y != k)[0], :]
		cost_c, Grad_c = mmd_cost_grad(Ac, Xc, gamma)
		cost = cost + cost_c
		Grad[c*perclass:(c+1)*perclass, :] = Grad_c.reshape(perclass, d)
		if lambdaa > 0:
			cost_k, Grad_k = 0.0, 0.0
			if diff == "data":
				cost_k, Grad_k = mmd_cost_grad(Ac, Xk, gamma2)
				Grad_k = Grad_k.reshape(perclass, d)
			elif diff == "EKxs":
				cost_k, Grad_k = ekxs_cost_grad(Ac, Xk, gamma2)
				Grad_k = Grad_k.reshape(perclass, d)
			cost -= lambdaa * cost_k
			Grad[c*perclass:(c+1)*perclass, :] -= lambdaa * Grad_k
	if diff == "params" and lambdaa > 0:
		cost_k, Grad_k = mmd_cost_grad_params(A.reshape(-1), d, len(classes), gamma2)
		cost -= lambdaa * cost_k
		Grad -= lambdaa * Grad_k.reshape(m, d)
	return cost, Grad.flatten()

### cost only
def mmd_cost_params(A: np.ndarray, d: int, K: int, gamma: float) -> float:
	# print(A.shape, d, K, gamma)
	m = A.shape[0] // d
	A = A.reshape(m, d)
	mk = m // K
	cost = 0
	Kaa = pairwise_kernels(A, metric = "rbf", gamma = gamma)

	for k in np.arange(K):
		r = np.arange(k*mk,(k+1)*mk,1)
		msk = np.zeros(m, dtype = np.bool)
		msk[r] = True
		cost_k =  Kaa[msk, :][:, msk].mean() + Kaa[~msk, :][:, ~msk].mean() -2 * Kaa[msk, :][:, ~msk].mean()
		cost += cost_k
	return cost

def mmd_cost_labels(A: np.ndarray, X: np.ndarray, y: np.ndarray, 
					gamma: float, gamma2: float, lambdaa: float = 0.0, diff = "data", **kwargs: Any) -> Tuple[float, np.ndarray]:
	cost = 0.0
	_, d = X.shape
	m = A.shape[0] // d
	A = A.reshape(m, d)
	classes = sorted(set(y))
	perclass = m // len(classes)

	for c, k in enumerate(classes):
		Ac = A[c*perclass:(c+1)*perclass, :].flatten()
		Xc = X[np.where(y == k)[0], :]
		Xk = X[np.where(y != k)[0], :]
		cost_c, Grad_c = mmd_cost_grad(Ac, Xc, gamma)
		cost = cost + cost_c
		cost_k = 0.0
		if diff == "data" and lambdaa > 0:
			cost_k = mmd_cost(Ac, Xk, gamma2)
		elif diff == "EKxs":
			cost_k = ekxs_cost(Ac, Xk, gamma2)
		cost -= lambdaa * cost_k
	if diff == "params" and lambdaa > 0:
		cost_k = mmd_cost_params(A.reshape(-1), d, len(classes), gamma2)
		cost -= lambdaa * cost_k
	return cost

## MMD with labels: different prototypes
# @profile
def mmd_cost_grad_params(A: np.ndarray, d: int, K: int, gamma: float) -> float:
	# print(A.shape, d, K, gamma)
	m = A.shape[0] // d
	A = A.reshape(m, d)
	mk = m // K
	cost = 0
	Grad = np.zeros((m, d))
	Kaa = pairwise_kernels(A, metric = "rbf", gamma = gamma)

	for k in np.arange(K):
		r = np.arange(k*mk,(k+1)*mk,1)
		msk = np.zeros(m, dtype = np.bool)
		msk[r] = True
		cost_k =  Kaa[msk, :][:, msk].mean() + Kaa[~msk, :][:, ~msk].mean() -2 * Kaa[msk, :][:, ~msk].mean()
		cost += cost_k

		for l in range(m):
			if l in r:
				Grad[l] += (4 * gamma / mk * ((A[msk, :] - A[l]).T * Kaa[msk, :][:,l]).T.mean(axis = 0) )
				Grad[l] -= (4 * gamma / mk * ((A[~msk, :] - A[l]).T * Kaa[~msk, :][:,l]).T.mean(axis = 0) )
			if l not in r:
				Grad[l] += (4 * gamma / (m - mk) * ((A[~msk, :] - A[l]).T * Kaa[~msk, :][:,l]).T.mean(axis = 0) )
				Grad[l] -= (4 * gamma / (m - mk) * ((A[msk, :] - A[l]).T * Kaa[msk, :][:,l]).T.mean(axis = 0) )
	return cost, Grad.flatten()


####################################################################################################
########################################## OPTIMIZATION ############################################
def step_decay(epochs: int, drop: float = 0.5, epochs_drop: int = 10) -> float:
	'''
		step_decay of learning rate
	'''
	return drop ** ((1 + epochs) // epochs_drop)

def gd(func: Callable[[np.ndarray, Any], Tuple[np.ndarray, List[float]]], 
		param0: np.ndarray, lr: float = 0.1, beta: float = 0.9, 
		decay: Callable[[int], float] = partial(step_decay, drop = 0.5, epochs_drop = 10),
		max_epochs: int = 100, tol: float = 1e-6, **kwargs) -> Tuple[np.ndarray, List[float]]:
	'''
		Gradient descent with momentum
		args:
			- func => f: (param, *args) -> cost, grad
			- param0 => initial guess
			- kwargs => optional arguments to the cost/grad function
			- lr => learning rate
			- beta => momentum parameter
			- max_epochs => maximum number of epochs
			- tol => tolerance of param for stopping criteria
		returns:
			- param: optimized parameter
			- cost: list of costs evaluated in each iterations
	'''
	costs = []
	V = np.zeros_like(param0)
	for epoch in range(max_epochs):
		cost, grad = func(param0, **kwargs)
		V = beta * V + grad
		lr_ = (lr * decay(epoch))
		param = param0 - lr_ * V
		costs.append(cost)
		if np.abs(param - param0).sum() <= tol:
		# if i >= 1 and np.abs(costs[-1] - costs[-2]) <= tol:
			break
		param0 = param
	return param, costs


def sgd(func, param0, X, y = None, batch_size = 100, lr = 0.1, beta = 0.9, 
		decay = partial(step_decay, drop = 0.5, epochs_drop = 5), tol = 1e-6,
		max_epochs = 100, **kwargs):
	'''
		Stochastic Gradient descent with momentum
		args:
			- func => f: (param, *args) -> cost, grad
			- param0 => initial guess
			- args => optional arguments to the cost/grad function
			- lr => learning rate
			- beta => momentum parameter
			- max_epochs => maximum number of epochs
			- tol => tolerance of param for stopping criteria
		returns:
			- param: optimized parameter
			- cost: list of costs evaluated in each iterations
	'''
	costs = []
	N, _ = X.shape
	V = np.zeros_like(param0)
	num_batches = ceil(N/batch_size)
	print("starting sgd with {} batches".format(num_batches))
	for epoch in range(max_epochs):
		for i in range(num_batches):
			if y is not None:
				cost, grad = func(param0.flatten(), 
					X = X[i * batch_size: (i+1) * batch_size],
					y = y[i * batch_size: (i+1) * batch_size],
					**kwargs
				)
			else:
				cost, grad = func(param0.flatten(), 
					X = X[i * batch_size: (i+1) * batch_size],
					**kwargs
				)

			V = beta * V + grad.reshape(param0.shape)
			lr_ = (lr * decay(epoch))
			param = param0 - lr_ * V
			costs.append(cost)
			if np.abs(param - param0).sum() <= tol:
				break
			param0 = param
		# print("epoch {} => {:.4f}".format(epoch +1, cost))
	# print("sgd costs:", costs)
	return param, costs


#############################################################################
#################### Just some tests from here #########################
import h5py
def hdf5(path):

    with h5py.File(path, 'r') as hf:
        X = hf.get("data_pca85")[:]
        y = np.array(hf.get("target")[:], dtype = np.uint8)
        train_idxs = hf.get("train_idxs")[:]
        test_idxs = hf.get('test_idxs')[:]

    return X[train_idxs[0]], y[train_idxs[0]], X[test_idxs[0]], y[test_idxs[0]]


def eval_sgd():
	from scipy.optimize import check_grad, approx_fprime, minimize
	from sklearn.cluster import KMeans
	import os
	X_tr, y_tr, X_te, y_te  = hdf5(
		os.environ["HOME"] + "/Nextcloud/datasets/usps.h5"
	)

	m = 4
	gamma = 0.04
	A0 = []
	for c in sorted(set(y_tr)):
		kmeans = KMeans(n_clusters = 2, init = "k-means++", random_state = 29)
		kmeans.fit(X_tr)
		A0.append(kmeans.cluster_centers_)
	A0 = np.concatenate(A0, axis = 0)
	print("kmeans completed")
	A, costs = sgd(mmd_cost_grad_labels, A0, X_tr, y_tr, gamma = gamma, gamma2 = gamma,lambdaa = 0.1)
	print("sgd", costs)

def eval_lbfgs():
	from scipy.optimize import check_grad, approx_fprime, minimize
	from sklearn.cluster import KMeans
	import os
	X_tr, y_tr, X_te, y_te  = hdf5(
		os.environ["HOME"] + "/Nextcloud/datasets/usps.h5"
	)

	m = 4
	gamma = 0.04
	A0 = []
	for c in sorted(set(y_tr)):
		kmeans = KMeans(n_clusters = 2, init = "k-means++", random_state = 29)
		kmeans.fit(X_tr)
		A0.append(kmeans.cluster_centers_)
	A0 = np.concatenate(A0, axis = 0)

	print("kmeans completed")
	opt = sp.optimize.minimize(mmd_cost_grad_labels, A0.flatten(), args = (X_tr, y_tr, gamma, gamma, 0.1), 
	                    method='L-BFGS-B', jac = True, tol = 1e-6,
	                    options={'maxiter': 100, 'disp': True})
	# A = opt.x.reshape(m, X_tr.shape[1])
	print("LBFGS", opt.x.shape())

def main():
	eval_sgd()
	eval_lbfgs()

if __name__ == '__main__':
	main()
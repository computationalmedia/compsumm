import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import product, chain
from abc import ABC, abstractmethod
from kmedoids import KMedoids
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors as NN
import subm, grad

import scipy as sp
from sklearn.neighbors import KNeighborsClassifier as KNN
import os, gc
from sklearn.cluster import KMeans
from utils import Kernel, kmeanspp, bacc
from sklearn.metrics import confusion_matrix
from collections import Counter

scorer = make_scorer(bacc)
mmd	    = subm.MMD()
ekxs    = subm.EKxs()
kxs     = subm.Kxs()
dummy   = subm.Dummy()
nnsubm  = subm.NNSubm()
critic1 = subm.Critic(abs = True)
critic2 = subm.Critic(abs = False)
logdet1 = subm.LogDet(merge = False)
logdet2 = subm.LogDet(merge = True)

def kmeans_obj(A, X, y):
	clss = sorted(set(y))
	cost = 0.0
	clss = sorted(set(y))
	mk = A.shape[0] // len(clss)
	for c, k in enumerate(clss):
		idxs = np.where(y == k)[0]
		D = cdist(A[c*mk:(c+1)*mk], X[idxs], 'sqeuclidean').min(axis=0)
		# print(len(idxs), D.shape)
		cost += D.sum()
	return cost

class Summ(BaseEstimator):
	def __init__(self, mk = 4, C = 10.0, gamma = 0.01, lambdaa = 0.1, gamma2 = 0.1, seed = 0, **kwargs):
		self.mk = mk
		self.gamma = gamma
		self.gamma2 = gamma2
		self.C = C
		self.lambdaa = lambdaa
		self.verbose = kwargs.get("verbose", False)
		self.idxs = None
		self.A0 = None
		self.vecs = None
		self.vecs_not_snapped = None ## not snapped ones if applicable
		self.seed = seed
		
		# self.clf = KNN(n_neighbors=1, n_jobs = 1)
		self.clf = SVC(gamma = self.gamma, kernel = "rbf", C = self.C, 
			class_weight = "balanced", random_state = seed, verbose = False)

	def __str__(self):
		return "{}(gamma={}, gamma2={}, C={}, lambda={}, seed={}, m={})".format(
			self.__class__.__name__, self.gamma, self.gamma2, self.C, self.lambdaa, self.seed, self.mk)
	
	# @abstractmethod
	def obj(self, A, X, y):
		raise NotImplementedError("must implement")

	def _init(self, X, y, init = "greedy", **kwargs):
		clss = np.array(sorted(set(y)), dtype = np.uint8)
		idxs = []
		vecs = []
		if init == "kmeans":
			for c, k in enumerate(clss):
				idxs_c = np.where(y == k)[0]
				kmeans = KMeans(n_clusters = self.mk, init = "k-means++", random_state = self.seed)
				kmeans.fit(X[idxs_c])
				
				nn = NN(n_neighbors = 1, algorithm = 'auto', metric = "euclidean")
				nn.fit(X[idxs_c])
				vecs.append(kmeans.cluster_centers_)
				idxs.append(idxs_c[nn.kneighbors(kmeans.cluster_centers_)[1].squeeze().tolist()])
			vecs = np.concatenate(vecs, axis = 0)
		elif init == "greedy":
			K1 = Kernel.create(X, verbose = False, metric = 'rbf', gamma = self.gamma)
			# K2 = Kernel.create(X, verbose = False, metric = 'rbf', gamma = self.gamma2)
			assert K1().shape[0] == X.shape[0] == len(y), (K1().shape, X.shape, y.shape)
			# print("kernels", K1, K2)
			print(self.diff, self.gamma, self.lambdaa, self.C)
			diff = {"ekxs": ekxs, "dummy": dummy, "data": mmd, "kxs": kxs}.get(self.diff.lower(), dummy)
			# print(diff)
			S = subm.greedy_maximize_labels(mmd, diff, V = np.arange(len(y)).tolist(), y = y, 
				verbose = False, k = self.mk, lambdaa = -self.lambdaa, delF_args = {"K": K1}, delG_args = {"K": K1})
			for c in clss:
				idxs.append(S[c])
		idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()
		return idxs, vecs

	def predict(self, X):
		try:
			return self.clf.predict(X)
		except:
			return np.zeros(len(X))

	def score(self, X, y, **kwargs):
		return bacc(y, self.predict(X), **kwargs)

class KMeansSumm(Summ):	
	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self)
		self.idxs, self.vecs_not_snapped = self._init(X, y, "kmeans")
		self.vecs = X[self.idxs] 
		cc = Counter(y[self.idxs])
		assert set(cc.values()) == set({self.mk})
		if self.verbose:
			print("fitted KMeans: seed = {}, ".format(self.seed), cc)
		self.clf.fit(self.vecs, y[self.idxs])
		return self

	def obj(self, A, X, y):
		return kmeans_obj(A, X, y)

class KMedoidsSumm(Summ):
	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self)
		clss = sorted(set(y))
		meds = []
		for c, k in enumerate(clss):
			idxs_c = np.where(y == k)[0]
			kmpp_idxs = kmeanspp(X[idxs_c], self.mk, seed = self.seed)
			kmed = KMedoids(self.mk, init = kmpp_idxs)
			kmed.fit(X[idxs_c], dist = False)
			meds.append(idxs_c[kmed.medoids].tolist())

		self.idxs = np.concatenate(meds, axis = 0)
		self.vecs = X[self.idxs]
		cc = Counter(y[self.idxs])
		assert set(cc.values()) == set({self.mk})
		if self.verbose:
			print("fitted KMedoids seed={},".format(self.seed), cc)
		self.clf.fit(self.vecs, y[self.idxs])
		return self
	
	def obj(self, A, X, y):
		return kmeans_obj(A, X, y)

class MMDGreedySumm(Summ):
	def __init__(self, mk = 4, C = 10.0, gamma = 0.1, gamma2 = 0.01, lambdaa = 0.1, diff = "data", **kwargs):
		super().__init__(mk = mk, C = C, gamma = gamma, gamma2 = gamma2, lambdaa = lambdaa, **kwargs)
		self.diff = diff

	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self)
		self.idxs, _ = self._init(X, y, init = "greedy", diff = self.diff)
		cc = Counter(y[self.idxs])
		assert set(cc.values()) == set({self.mk}), self.idxs
		if self.verbose:
			print("fitted MMD-Greedy-", self.diff, cc)
		self.vecs = X[self.idxs]
		self.clf.fit(self.vecs, y[self.idxs])
		return self
	
	def obj(self, A, X, y):
		return grad.mmd_cost_labels(A.flatten(), X, y, self.gamma, self.gamma2, self.lambdaa, self.diff)

class NNSumm(Summ):

	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self)
		V = np.arange(len(y)).tolist()
		W = sp.spatial.distance.cdist(X, X, 'euclidean')
		W = W.max() - W

		S = subm.greedy_maximize_labels(nnsubm, dummy, 
			V = V, y = y, k = self.mk, lambdaa = 0.0, W = W)
		self.idxs = []
		# print(S)
		for k in sorted(set(y)):
			self.idxs.append(S[k])
		self.idxs = np.array(self.idxs).flatten().tolist()
		cc = Counter(y[self.idxs])
		assert set(cc.values()) == set({self.mk}), self.idxs
		if self.verbose:
			print("fitted NN-Subm", cc)
		self.vecs = X[self.idxs]
		self.clf.fit(self.vecs, y[self.idxs])
		return self
	
	def obj(self, A, X, y):
		W = sp.spatial.distance.cdist(X, A, 'euclidean')
		W = W.max() - W
		cost = 0.0
		for c, k in enumerate(sorted(set(y))):
			idxs = np.where(y == k)[0]
			# print(idxs)
			cost += W[idxs, :].max(axis = 1).sum()
		return cost

class MMDcSumm(Summ):
	def __init__(self, mk = 4, C = 10.0, gamma = 0.1, original_critic = True, **kwargs):
		super().__init__(mk = mk, C = C, gamma = gamma, **kwargs)
		self.original_critic = original_critic

	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self)
		clss = sorted(set(y))
		V = np.arange(len(y)).tolist()
		s = np.arange(len(y)).tolist()
		K = kwargs.get("K", Kernel.create(X, verbose = False, metric = 'rbf', gamma = self.gamma))
		K.regularize(1e-4)
		ps = mmd.greedy_maximize(candidates = s, V = V, k = (self.mk * len(clss)) // 2, verbose = False, K = K)
		s = [i for i in V if i not in ps]
		reg_critic = (critic1 + logdet1) if self.original_critic else (critic2 + logdet2)
		cs = reg_critic.greedy_maximize(V = V, candidates = s, k =  (self.mk * len(clss)) // 2, K = K, P = ps)

		self.idxs = ps + cs
		self.vecs = X[self.idxs]
		if self.verbose:
			print("fitted MMDCritic, ", Counter(y[self.idxs]))
		try:
			self.clf.fit(self.vecs, y[self.idxs])
		except Exception as ex:
			print(ex)
		return self

	def obj(self, A, X, y):
		return grad.mmd_cost(A.flatten(), X, self.gamma)


class MMDGradSumm(Summ):
	def __init__(self, mk = 4, C = 10.0, gamma = 0.1, gamma2 = 0.01, lambdaa = 0.1, seed = 0, 
							diff = "data", init = "greedy", **kwargs):
		super().__init__(mk = mk, C = C, gamma = gamma, gamma2 = gamma2, lambdaa = lambdaa, seed = seed, **kwargs)
		self.diff = diff
		self.init = init

	def fit(self, X, y, **kwargs):
		if self.verbose:
			print("fitting", self, self.diff)
		n, d = X.shape
		clss = np.array(sorted(set(y)), dtype = np.uint8)
		
		## optimize this later
		idxs, _ = self._init(X, y, init = self.init, **kwargs)
		A0 = X[idxs]
		
		opt = sp.optimize.minimize(grad.mmd_cost_grad_labels, A0.flatten(), 
										args = (X, y, self.gamma, self.gamma, self.lambdaa, self.diff), 
										method='L-BFGS-B', jac = True, tol = 1e-6,
									 	options={'maxiter': 100, 'disp': False})
		self.vecs_not_snapped = opt.x.reshape(self.mk * len(clss), d)
		# print(self.vecs_not_snapped)
		idxs = []
		for c, k in enumerate(clss):
			ixs_c = np.where(y == k)[0]
			nn = NN(n_neighbors = 1, algorithm = 'auto', metric = "euclidean")
			nn.fit(X[ixs_c])
			idxs.append(ixs_c[nn.kneighbors(self.vecs_not_snapped[c*self.mk: (c+1)*self.mk])[1].squeeze().tolist()])
		self.idxs = np.array(idxs).flatten().tolist()
		cc = Counter(y[self.idxs])
		assert set(cc.values()) == set({self.mk})
		if self.verbose:
			print("fitted MMD-Grad-",self.diff, cc, self.idxs)
		self.vecs = X[self.idxs]
		self.A0 = A0
		self.clf.fit(self.vecs, y[self.idxs])
		return self
	
	def obj(self, A, X, y):
		return grad.mmd_cost_labels(A.flatten(), X, y, self.gamma, self.gamma, self.lambdaa, self.diff)


def get_model(name, **kwargs):
	MODELS = {  "kmeans": KMeansSumm(**kwargs), 
				"kmedoids": KMedoidsSumm(**kwargs),
				"mmd-critic": MMDcSumm(original_critic = True, **kwargs),
				"mmd-critic+": MMDcSumm(original_critic = False, **kwargs), ## not used in paper
				"greedy": MMDGreedySumm(diff = "dummy", **kwargs), ## not used in paper
				"mmd-diff-greedy": MMDGreedySumm(diff = "data", **kwargs),
				"mmd-exks-greedy": MMDGreedySumm(diff = "EKxs", **kwargs), ## not used in paper
				"mmd-div-greedy": MMDGreedySumm(diff = "Kxs", **kwargs),
				"mmd-diff-grad": MMDGradSumm(diff = "data", **kwargs),
				"mmd-params-grad": MMDGradSumm(diff = "params", **kwargs), ## not used in paper
				"mmd-div-grad": MMDGradSumm(diff = "EKxs", **kwargs),
				"nn-comp": NNSumm(**kwargs)
	}	
	return MODELS[name]

from sklearn.utils.estimator_checks import check_estimator
def main():
	for M in [MMDcSumm, MMDGreedySumm, KMeansSumm, KMedoidsSumm, MMDGradSumm]:
		check_estimator(M) 
if __name__ == '__main__':
	main()

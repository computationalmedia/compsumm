import numpy as np
import scipy as sp
from collections import Counter
from sklearn.metrics import confusion_matrix
from functools import lru_cache
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels

## imbalanced classification
## balanced accuracy
def bacc(y_true, y_pred, verbose = False):
	C = confusion_matrix(y_true, y_pred)
	if verbose:
		print("bacc", Counter(y_true), Counter(y_pred))
		print(C)
	counts = C.sum(axis=1)
	positives = C.diagonal()
	res = np.mean(positives / counts)
	return res

## kmeans++ initialisations
def kmeanspp(X, n_clusters, dist = 'euclidean', seed = 0):
	c_idxs = [np.random.RandomState(seed).randint(X.shape[0])]
	for i in range(1, n_clusters):
		D_x2 = cdist(X, X[c_idxs], metric = dist).min(axis = 1) ** 2
		p_x = D_x2 / np.sum(D_x2)
		c_idxs.append(p_x.argmax())
	return c_idxs

## wrapper for kernel matrix
## nothing special, helps in modular code
class Kernel(object):
	"""
		A kernel matrix that memorizes the mean of rows
	"""
	def __init__(self, K, verbose = False):
		self.__data = K
		self.__sum = K.sum()
		self.verbose = verbose

	def __repr__(self):
		return "[{:.4f}, {:.4f}, {:4f}]".format(self.__data.min(), np.median(self.__data), self.__data.max())
	
	def __getitem__(self, index):
		return Kernel(self.__data[index])

	def __call__(self, index = None):
		if index is None: 
			return self.__data
		else: 
			return self[index]

	def __len__(self):
		return len(self.__data)

	@property 
	def sum(self):
		return self.__sum

	@property
	def avg(self):
		return 1.0 * self.__sum /  ( len(self) * len(self) )

	def __del__(self):
		del self.__data

	def regularize(self, alpha):
		self.__data = self.__data + alpha * np.eye(self.__data.shape[0])

	@lru_cache(maxsize=10)
	def __compute__mean(self, rows = []):
		N = len(rows)
		if self.verbose:
			print("caching mean for {} rows".format(len(self) if N == 0 else N))
		if N == 0:
			return self.__data.mean(axis = 0)
		else:
			return self.__data[rows, :].mean(axis = 0)
	
	def mean(self, rows = []):
		return self.__compute__mean(tuple(rows))

	def diagonal(self):
		return self.__data.diagonal()

	@staticmethod
	def create(X, verbose = False, **kernelargs):
		# temp = np.exp(-kernelargs.get("gamma") * sp.spatial.distance.cdist(X, X, 'euclidean'))
		# return Kernel(temp)
		return Kernel(pairwise_kernels(X, **kernelargs), verbose)


def main():
	np.random.seed(31)

	from itertools import chain
	ns = [30, 10]
	data = np.vstack([
		np.random.normal(loc = [0, 0], scale = 6.0, size = (ns[0], 2)),
		np.random.normal(loc = [10, 10], scale = 4.0, size = (ns[1], 2))
	])
	print(data.sum(axis=0))
	labels = np.array(list(chain(*[[i +1] * n for i, n in enumerate(ns)])))
	K1 = Kernel.create(data, metric = 'rbf', gamma = 0.1)
	K2 = Kernel.create(data, metric = 'rbf', gamma = 0.01)
	V = list(range(sum(ns)))
	print(K1.mean(rows = V))
	print(kmeanspp(data, n_clusters = 4))
	
	y1 = np.random.randint(0, 2, 10)
	y2 = np.random.randint(0, 2, 10)
	print(bacc(y1, y2))

if __name__ == "__main__":
	main()
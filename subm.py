# from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from math import ceil
from scipy.linalg import solve
from utils import Kernel
from copy import deepcopy, copy
from itertools import chain
import scipy as sp
import types
# from profilehooks import profile

EPS = 1e-12

class DelF(ABC):
	'''
		A class for computing delF (change in submodular function) over canidates
			s: candidates
			S: the set already computed
	'''	

	def greedy_maximize(self, candidates, k = 5, verbose = False, **kwargs):
		'''
			greedily maximizes
		'''
		S = []
		while k>0:
			dF = self(S, candidates, **kwargs) 
			ix = np.argmax(dF)
			item = candidates[ix]
			k -= 1
			S.append(item)
			candidates.remove(item)
			if verbose:
				print("%s(s=%d, |S|=%d) = %+.4f, |s|=%d"%(
					self.__class__.__name__, item, len(S), dF[ix], len(candidates)))
		return S

	@abstractmethod
	def f(self, S, **kwargs):
		'''
			for debugging (empericaly checking submodularity)
			F(S)
		'''
		raise NotImplementedError("Must override function")

	@abstractmethod
	def apply(self, S, s, **kwargs):
		'''
			args:
				V: is the entire set of Unverse	
			returns: 
				F(S+s) -F(S) for each s (aka discrete derivative \\del_s F(S))
		'''
		raise NotImplementedError("Must override function")

	def __call__(self, *args, **kwargs):
		return self.apply(*args, **kwargs)

	def __add__(self, other):
		'''
			Addition of 2 submodular functions is a submodular function
			F(S+s) - F(S) + G(S+s) - G(S)
		'''
		print("+", end = "")
		new = deepcopy(self)
		old_apply1, old_f1 = self.apply, self.f
		old_apply2, old_f2 = other.apply, other.f
		def new_apply(self, *args, **kwargs):
			return old_apply1(*args, **kwargs) + old_apply2(*args, **kwargs)
		def new_f(self, *args, **kwargs):
			return old_f1(*args, **kwargs) + old_f2(*args, **kwargs)
		new.apply = types.MethodType(new_apply, new)
		new.f = types.MethodType(new_f, new)
		return new

	def __sub__(self, other):
		'''
			Subtraction of modular function from submodular is a submodular function
			F(S+s) - F(S) - ( G(S+s) - G(S) )
		'''
		print("-", end = "")
		return self + (-1) * other

	def __rmul__(self, alpha):
		'''
			Allows multiplication of a scaler with delF
			alpha * (F(S+s) - F(S))
		'''
		print("*", end = "")
		new = deepcopy(self)
		old_apply, old_f = self.apply, self.f
		def new_apply(self, *args, **kwargs):
			return alpha * old_apply(*args, **kwargs)
		def new_f(self, *args, **kwargs):
			return alpha * old_f(*args, **kwargs)
		new.apply = types.MethodType(new_apply, new)
		new.f = types.MethodType(new_f, new)
		return new


# @profile
def greedy_maximize_labels(delF, delG, V, y, k = 5, lambdaa = 0.0, verbose = False, 
				delF_args = {}, delG_args = {}, **kwargs):
	'''
		Greedily maximizes the \\sum_g [delF(S_g, s_g, V_g) + \\lambda delG(S_g, s_g, V_~g)]
		i.e. greedy optimization of partition matroids
	'''
	
	classes = sorted(set(y))
	V = np.array(list(V)).flatten()
	assert len(V) == len(y)

	S = {g: [] for g in classes}
	for g in classes:
		idxs = np.where(y == g)[0]
		Vk, sk = V[idxs], V[idxs]
		Vk_ = np.array(np.setdiff1d(V, Vk, assume_unique = True)) ## not in class 
		mk = k
		while mk > 0:
			dFk = delF(S = S[g], s = sk, V = Vk, **delF_args, **kwargs)
			if lambdaa != 0.0:
				dG = delG(S = S[g], s = sk, V = Vk_ , **delG_args, **kwargs)
				dFk += (lambdaa * dG)
			ixk = np.argmax(dFk)
			sk_next = int(sk[ixk])
			S[g].append(sk_next)
			sk = np.delete(sk, ixk)
			if verbose:
				print("dF_%d(s=%d, |s|=%d, |S|=%d)= %+.4f, k=%d"%(
					g, sk_next, len(sk), len(S[g]), dFk[ixk], mk))
			mk -= 1
	# print(S_)
	return S

class MMD(DelF):
	'''
		F(S+s) -F(s) for  Maximum mean discrepency
		- is submodular
	'''
	def f(self, S, V, K, **kwargs):
		return 2 * K.mean(rows = V)[S].mean()- np.mean( K[S, :][:, S]())

	def apply(self, S, s, V, K, **kwargs):
		
		s1len = len(S) + 1
		s1 = 2 * K.mean(rows = V)[s] - 1./s1len * K.diagonal()[s] ## cached ##this first
		s2 = 0
		if s1len > 1:
			s2 -= 2 * K.mean(rows = V)[S].mean()## this
			s2 += (2 * (s1len - 1) / s1len + 1./s1len) * np.mean(K[S, :][:, S]())
			s2 -= 2 * (s1len - 1) / s1len * np.mean( K[S, :][:, s](), axis = 0)
		# print(len(V), s1.mean(), len(s))
		return 1./s1len * (s1 + s2) ## normalizer removed

class Dummy(DelF):
	'''
		Does nothing
	'''
	def f(self, S, **kwargs):
		return 0.0

	def apply(self, S, s, **kwargs):
		return np.zeros(len(s))

class EKxs(DelF):
	'''
		F(S+s) -F(s) for  Exp[Sum(Sum(k(x,x_s)))]
		- is submodular
	'''
	def f(self, S, V, K, **kwargs):
		return K.mean(rows = V)[S].mean()

	def apply(self, S, s, V, K, **kwargs):
		s1len = len(S) + 1
		s1 = K.mean(rows = V)[s] ## cached
		s2 = 0
		if s1len > 1:
			s2 -= K.mean(rows = V)[S].mean()
		# print(s1, s2)
		return 1./s1len * (s1 + s2) ## normalizer removed

class Div(DelF):
	'''
		F(S+s) -F(s) for  Exp[Sum(Sum(k(x,x_s)))]
		- is submodular
	'''
	def f(self, S, V, K, **kwargs):
		return - np.mean( K[S, :][:, S]())

	def apply(self, S, s, V, K, **kwargs):
		s1len = len(S) + 1
		s1 =  - 1./s1len * K.diagonal()[s] ## cached ##this first
		s2 = 0
		if s1len > 1:
			s2 += (2 * (s1len - 1) / s1len + 1./s1len) * np.mean(K[S, :][:, S]())
			s2 -= 2 * (s1len - 1) / s1len * np.mean( K[S, :][:, s](), axis = 0)
		# print(len(V), s1.mean(), len(s))
		return 1./s1len * (s1 + s2) ## normalizer removed

class Kxs(DelF):
	'''
		F(S+s) -F(s) for  Exp[Sum(Sum(k(x,x_s)))]
		- is modular
	'''
	def f(self, S, V, K, **kwargs):
		return K.mean(rows = V)[S].sum() ## mean to normalize

	def apply(self, S, s, V, K, **kwargs):
		# print(K.mean(rows = V)[s])
		return K.mean(rows = V)[s] ## mean to normalize, doesn't matter if sum is used

class NNSubm(DelF):
	'''
		F(S+s) - F(S) for nearest neighbor objective from Blimes paper
	'''
	def f(self, S, V, W, **kwargs):
		return np.max(W[V, :][:, S], axis = 1).sum()

	def apply(self, S, s, V, W, **kwargs):
		
		if len(S) > 0:
			WiS = W[V, :][:, S].max(axis = 1)
			return (np.maximum(W[V, :][:, s].T, WiS) - WiS).sum(axis = 1)
		else:
			return W[V, :][:, s].sum(axis = 0)

class Critic(DelF):
	'''
		F(S+s) -F(s) for  witness function
		Witness function optimizes set S over s where S is similar to V and different than W
		- is modular
	'''
	def __init__(self, abs = True, **kwargs):
		self.__abs = np.abs if abs else lambda x: x

	def f(self, S, V, K, P, **kwargs):
		Lc1 = K.mean(rows = V)[S] #cached
		Lc2 = K.mean(rows = P)[S] ## cached
		return self.__abs(Lc1 - Lc2).sum()

	def apply(self, S, s, V, K, P, **kwargs):
		Lc1 = K.mean(rows = V)[s] #cached
		Lc2 = K.mean(rows = P)[s] ## cached
		return self.__abs(Lc1 - Lc2)

class LogDet(DelF):
	'''
		F(S+s) -F(s) for  log-determinant
		- is submodular
	'''
	def __init__(self, merge = True, **kwargs):
		self.__merge  = merge

	def f(self, S, V, K, P, **kwargs):
		S1 = S + P if self.__merge else S
		eigvals = sp.linalg.eigvalsh(K[S1, :][:, S1]())
		eigvals[eigvals < EPS] = EPS
		return np.log(eigvals).sum()

	def apply(self, S, s, V, K, P, **kwargs):
		S1 = S + P if self.__merge else S
		if len(S1) == 0:
			det = K.diagonal()[s]
		else:
			K_CC = K[S1, :][:, S1]()
			K_Cc = K[S1, :][:, s]()
			k_cc = K.diagonal()[s] 
			# solve(A, b) == dot(inv(A), b)
			term = solve(K_CC, K_Cc, assume_a = "pos")
			# term = np.dot(npla.inv(K_CC), K_Cc)
			det = np.abs(k_cc - (term * K_Cc).sum(axis = 0))
		det[det < EPS] = EPS
		return np.log(det)

if __name__ == '__main__':
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
	candidates = copy(V)
	mmd = MMD()
	
	ps = mmd.greedy_maximize(candidates, V = V, k =5, verbose = True, K = K1)
	candidates = [_s for _s in candidates if _s not in ps]
	
	reg_critic = Critic(abs = False) + 0.1 * LogDet(merge = False)
	cs = reg_critic.greedy_maximize(candidates, V = V, k =5, verbose = True, K = K1, P = ps)
	print("unlabelled", ps, cs)

	y = labels == 1
	psk = greedy_maximize_labels(MMD(),  -1 * Kxs(), 
						V, y, k = 5, lambdaa = 0.1, 
						verbose = True, delF_args = {"K": K1}, delG_args = {"K": K2})
	print("labelled", psk)
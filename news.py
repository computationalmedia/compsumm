import json, sys, os, re, traceback
from itertools import product, chain
from collections import Counter
# import multiprocessing as mp
import h5py
import numpy as np
N_JOBS = 4
runs = 5
SUBM_PARALLEL = False

class NewsH5(object):
	def __init__(self, name):
		self.name = name
		# 
		with h5py.File("datasets/%s.h5"%self.name, 'r') as hf:
			self.X = hf.get("data")[:]
			self.y = np.array(hf.get("target")[:], dtype = np.uint8)
			try:
				self.yn = np.array(hf.get("target-month")[:], dtype = np.uint8)
			except:
				try:
					self.yn = np.array(hf.get("target-week")[:], dtype = np.uint8)
				except:
					self.yn = self.y

			self.title = hf.get("title")[:].view(np.chararray).decode('utf-8')
			self.text = hf.get("text")[:].view(np.chararray).decode('utf-8')
			self.datetime = hf.get("datetime")[:].view(np.chararray).decode('utf-8')
			self.train_idxs = hf.get("train_idxs")[:]
			self.test_idxs = hf.get('test_idxs')[:]
			self.val_idxs = hf.get('val_idxs')[:]
			# self.D = sp.spatial.distance.cdist(self.X, self.X, 'euclidean')

		self.classes = sorted(set(self.y))

	def get_idxs(self, i, j, split = 0):
		
		tr_idxs = set(self.train_idxs[split].tolist())
		te_idxs = set(self.test_idxs[split].tolist())
		val_idxs = set(self.val_idxs[split].tolist())

		ixs = np.where(np.logical_or(self.yn == i, self.yn == j))[0]
		ixs_tr = [i for i in ixs if i in tr_idxs]
		ixs_te = [i for i in ixs if i in te_idxs]
		ixs_val = [i for i in ixs if i in val_idxs]
		np.random.shuffle(ixs_tr)
		np.random.shuffle(ixs_te)
		np.random.shuffle(ixs_val)
		return ixs_tr, ixs_te, ixs_val
import numpy as np
from sklearn.base import ClusterMixin
from typing import List, Optional
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter

class KMedoids(ClusterMixin):
    def __init__(self, n_clusters: int, 
            init: Optional[List[int]] = None, verbose = False) -> None:
        self.__K = n_clusters
        self.__medoids = None
        if init is not None:
            assert len(init) == self.__K
            self.__medoids = np.array(init)
        self.__medoids_points = None
        self.__assignments = None
        self.__verbose = verbose

    @property
    def medoids(self) -> List[int]:
        return self.__medoids.tolist()

    @property
    def centroids(self) -> np.ndarray:
        return self.__medoids_points

    @centroids.setter
    def centroids(self, X: np.ndarray) -> None:
        self.__medoids_points = X

    @property
    def assignments(self) -> List[int]:
        if self.__assignments is not None:
            return self.__assignments.tolist()
        else: 
            raise Exception("Model not trained yet")

    def __reassign(self, D: np.ndarray) -> np.ndarray:
        return np.argmin(D[:, self.__medoids], axis = 1)

    def __recompute(self, D: np.ndarray) -> np.ndarray:
        medoids = np.zeros_like(self.__medoids, dtype = np.int)
        for i, med in enumerate(self.__medoids):
            idxs = np.where(self.__medoids[self.__assignments] == med)[0]
            try:
                m_ = np.argmin(D[idxs,:][:,idxs].sum(axis = 0))
            except:
                print(self.medoids, idxs)
                print(len(Counter(self.__assignments)))
            medoids[i] = idxs[m_]
        return medoids

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, dist = True) -> 'KMedoids':
        '''
            X is pairwise distance measure if dist is True (default)
            else, X is the data matrix
        '''
        ## computer pairwise distance matrix
        D = pairwise_distances(X, n_jobs = 1) if not dist else X
        
        ## initialize initial medoids
        if self.__medoids is None:
            self.__medoids = np.random.choice(D.shape[0], self.__K, replace = False)
        assert D.shape[0] == D.shape[1]
        iters = 0
        
        ## repeat till convergence
        while True:
            self.__assignments = self.__reassign(D)
            new_medoids = self.__recompute(D)
            iters += 1
            if np.all(self.__medoids == new_medoids):
                if self.__verbose:
                    print("Converged in %d iters"%iters)
                break
            self.__medoids = new_medoids
        
        ## store medoids points
        if not dist:
            self.__medoids_points = X[self.__medoids, ]
        del D
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.__medoids_points is None:
            raise Exception("Centroids not set, use centroids setter.")
        D = pairwise_distances(X, self.__medoids_points)
        return np.argmin(D, axis = 1)
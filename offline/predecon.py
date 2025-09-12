import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def _predecon(X: np.ndarray, metric: str, epsilon: float, min_samples: int, max_pref_dimensionality: int,
              variance_threshold: float, k: float) -> (int, list, list, list, np.ndarray, np.ndarray):
    dim = len(X[0])
    w = np.zeros((len(X),dim))
    pdim = np.zeros(len(X))
    if max_pref_dimensionality == None:
        max_pref_dimensionality = dim

    neighbors = NearestNeighbors(radius=epsilon, metric=metric).fit(X).radius_neighbors(X, return_distance=False)
    for i in range(len(X)):
        for a in range(dim):
            ai = np.ones((1,1)) * X[i][a]
            nn_ai = X[neighbors[i],a].reshape(-1,1)
            #print(ai.shape, nn_ai.shape)
            adists = cdist(ai, XB=nn_ai, metric=metric)
            adist_square_sum = np.sum(adists**2, axis=1)
            vara = adist_square_sum/len(neighbors[i])
            pdim[i] += 1 if vara <= variance_threshold else 0
            w[i,a] = k if vara <= variance_threshold else 1
            print(vara, pdim[i], w[i,a])

    nn_w = []
    cores = []
    for ip in range(len(X)):
        nn_w.append([ip]) # self always included
    for ip in range(len(X)):
        #for iq in neighbors[ip]:
        for iq in range(len(X)):
            if iq > ip: # only calc once since symmetric
                if general_pref_weighted_similarity(X, w, ip, iq) <= epsilon:
                    nn_w[ip].append(int(iq))
                    nn_w[iq].append(int(ip))
        if len(nn_w[ip]) >= min_samples and pdim[ip] <= max_pref_dimensionality:
            cores.append(ip)
    clustering = [-1]* len(X)
    curClu = -1
    for o in cores: # no need to iterate over all, since we already know core points from above
        if clustering[o] == -1:
            curClu += 1
            clustering[o] = curClu
            queue = set()
            for neighbor in nn_w[o]:
                queue.add(neighbor)
            while(len(queue) > 0):
                curi = queue.pop()
                clustering[curi] = curClu
                r = []
                if curi in cores:
                    for neighbor in nn_w[curi]:
                        if pdim[neighbor] <= max_pref_dimensionality:
                            r.append(neighbor)
                for x in r:
                    if clustering[x] == -1:
                        clustering[x] = curClu
                        if x in cores:
                            queue.add(x)
    return len(np.unique(clustering)), clustering, cores, nn_w, pdim, w







def pref_weighted_similarity(X, w, ip, iq):
    dim = len(X[0])
    adist_sum = 0
    for a in range(dim):
        adist_sum += w[ip,a] * ((X[ip][a] - X[iq][a]) ** 2)
    adist = adist_sum ** 0.5
    return adist

def general_pref_weighted_similarity(X, w, ip, iq):
    return max(pref_weighted_similarity(X, w, ip, iq), pref_weighted_similarity(X, w, iq, ip))




class PreDeCon(BaseEstimator, ClusterMixin):

    """
    epsilon: epsilon
    min_samples: mu
    pref_dimensioanlity: lambda (upper limit for low variance, not necessarily needed)
    variance_threshold: delta (5 for range 100 -> 0.05 for range 1)
    """
    def __init__(self, metric: str = 'euclidean', eps: float=0.5, min_samples: int=5, max_pref_dimensionality: int=None, variance_threshold: float=0.05, k: float=10) -> None:
        self.epsilon = eps
        self.min_samples = min_samples
        self.max_pref_dimensionality = max_pref_dimensionality
        self.variance_threshold = variance_threshold
        self.metric = metric
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreDeCon":
        n_clusters, labels, cores, nn_w, pdim, w = _predecon(X, self.metric, self.epsilon, self.min_samples, self.max_pref_dimensionality,
                                       self.variance_threshold, self.k)
        self.n_clusters = n_clusters
        self.labels = labels
        self.weight_vectors = w
        self.cores = cores
        self.weighted_nearest_neighbors = nn_w
        self.pref_dimensionality = pdim

        return self




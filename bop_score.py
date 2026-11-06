import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import utils
from datahandler import load_data

def pca_part_otsu(data, n, l):
    clusters = [data]
    for i in range(n-1):
        largest_sse = -1*np.inf
        largest_sse_j = None
        for j in range(len(clusters)):
            cluster_mean = np.mean(clusters[j], axis=0)
            sse = np.sum(utils.distance_calc(clusters[j], cluster_mean, dist="euclidean")**2)
            if sse > largest_sse:
                largest_sse = sse
                largest_sse_j = j

        cluster = clusters[largest_sse_j]

        cluster_projection = PCA(svd_solver="full", n_components=1).fit_transform(cluster)

        clu_min = np.min(cluster_projection)
        clu_max = np.max(cluster_projection)
        #print([l*(p-clu_min) / (clu_max-clu_min) for p in cluster_projection])
        bin_assign = [np.floor((l*(p-clu_min)) / (clu_max-clu_min)) for p in cluster_projection]

        bin_ids, bin_count = np.unique(bin_assign, return_counts=True)

        #https://www.geeksforgeeks.org/python/python-convert-two-lists-into-a-dictionary/
        bin_prob = dict(zip(bin_ids, bin_count/len(cluster_projection)))

        mu_l1 = 0
        for bin_id in range(l):
            #print(bin_id)
            if bin_id not in bin_ids:
                bin_prob[bin_id] = 0
            mu_l1 += bin_id * bin_prob[bin_id]


        #print(mu_l1)

        max_sigma = -1*np.inf
        best_border_bin_id = None
        for border_bin_id in range(l):
            prob_0 = 0
            mu_0 = 0
            for bin_id in range(border_bin_id):
                prob_0 += bin_prob[bin_id]
                mu_0 += bin_id * bin_prob[bin_id]
            prob_1 = 1 - prob_0

            sigma = ((mu_l1*prob_0 - mu_0)**2) / (prob_0 + prob_1)

            if sigma > max_sigma:
                max_sigma = sigma
                best_border_bin_id = border_bin_id

        #bin_pos = np.linspace(clu_min, clu_max, l-1)
        #border = bin_pos[best_border_bin_id]
        #print(clu_max, clu_min)
        #print(border)
        #print(np.floor((l*(border-clu_min)) / (clu_max-clu_min)), best_border_bin_id)


        border = ((best_border_bin_id/l) * (clu_max-clu_min)) + clu_min
        #print(border)
        #print(np.floor((l*(border-clu_min)) / (clu_max-clu_min)), best_border_bin_id)

        assigns_1 = np.where(cluster_projection <= border)[0]
        assigns_2 = np.where(cluster_projection > border)[0]

        if len(assigns_2) > 0:

            prob_sum = 0
            for bin_id in range(best_border_bin_id):
                prob_sum += bin_prob[bin_id]

            #print(len(assigns_1), len(assigns_2),  len(assigns_1)/len(cluster_projection), prob_sum, prob_sum+bin_prob[best_border_bin_id])

            clusters.append(cluster[assigns_2])
            clusters[largest_sse_j] = cluster[assigns_1]
        else:
            break



    centers = []
    for cluster in clusters:
        cluster_mean = np.mean(cluster, axis=0)
        centers.append(cluster_mean)

    centers = np.array(centers)
    #print(centers.shape)

    return centers

def PCAPartKMeans(data, n_clusters, l=256, max_iter=None, tol=None):
    if max_iter is None:
        max_iter = 300
    if tol is None:
        tol = 1e-4
    init_centroids = pca_part_otsu(data, n_clusters, l)
    kmeans = KMeans(n_clusters=len(init_centroids), max_iter=max_iter, tol=tol, init=init_centroids)
    return kmeans

def calculate_bop(data_ref, weights_ref, data_comp, weights_comp, k, l=256):
    k = min(len(data_ref), k)
    kmeans = PCAPartKMeans(data_ref, k, l)
    clustering = kmeans.fit_predict(data_ref, sample_weight=weights_ref)
    centroids = kmeans.cluster_centers_
    ref_histogram = np.zeros(len(np.unique(clustering)))
    for i in range(len(clustering)):
        ref_histogram[clustering[i]] += weights_ref[i]

    comp_histogram = np.zeros(len(np.unique(clustering)))

    for i in range(len(data_comp)):
        closest = np.argmin(utils.distance_calc(data_comp[i], centroids, dist="euclidean"))
        comp_histogram[closest] += weights_comp[i]

    score = jensenshannon(ref_histogram/sum(ref_histogram), comp_histogram/sum(comp_histogram))
    return score



if __name__ == "__main__":
    data, labels = load_data("stream_densired")

    data_weights = np.ones(len(data))
    data_weights[1000:2000] = 50

    data2, _ = load_data("rbf3")
    data_weights2 = np.ones(len(data2))
    #data_weights2[1000:2000] = 50
    bop_score = calculate_bop(data, np.ones(len(data)), data2, data_weights2, 100)
    print(bop_score)
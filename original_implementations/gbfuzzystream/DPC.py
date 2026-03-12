# Original code from xjnine and 920850946 (dmg), from https://github.com/xjnine/GBFuzzyStream/tree/master

# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/30 20:52
@Auth ： daiminggao
@File ：DPC.py
@IDE ：PyCharm
@Motto:咕咕咕
"""
import numpy as np

verbose=False


def get_cluster_num(datas):
	dists = getDistanceMatrix(datas)
	dc = select_dc(dists)
	rho = get_density(dists, dc)
	deltas, nearest_neiber = get_deltas(dists, rho)
	centers = find_centers_auto(rho, deltas)
	return len(centers)


def get_cluster_DPC(datas):
	dists = getDistanceMatrix(datas)
	percent = 2
	while True:
		dc = select_dc(dists, percent)
		rho = get_density(dists, dc)
		deltas, nearest_neiber = get_deltas(dists, rho)
		centers = find_centers_auto(rho, deltas)
		if len(centers) >= 2:
			labs = cluster_PD(rho, centers, nearest_neiber)
			return labs, len(centers)
		else:
			if percent <= 80:
				percent = percent + 1
			else:
				labs = cluster_PD(rho, centers, nearest_neiber)
				return labs, len(centers)


def getDistanceMatrix(datas):
	N, D = np.shape(datas)
	dists = np.zeros([N, N])
	for i in range(N):
		for j in range(N):
			vi = datas[i, :]
			vj = datas[j, :]
			dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
	return dists


# Find the threshold dc for density calculation
# Require that the number of points with a distance less than dc
# around each point should account for 1%-2% of the total number of points
def select_dc(dists, percent=2):
	'''Algorithm 1'''
	N = np.shape(dists)[0]
	tt = np.reshape(dists, N * N)
	position = int(N * (N - 1) * percent / 100)
	dc = np.sort(tt)[position + N]
	return dc


# Calculate the local density of each point
def get_density(dists, dc, method=None):
	'''
	Method 1: Truncated kernel
	Method 2: Gaussian kernel
	'''
	N = np.shape(dists)[0]
	rho = np.zeros(N)
	for i in range(N):
		if method == None:
			rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
		else:
			rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
	return rho


# Calculate the density distance of each data point
# That is, for each point, find all points with a greater density than it
# Then find the distance to the nearest point among these points
def get_deltas(dists, rho):
	N = np.shape(dists)[0]
	deltas = np.zeros(N)
	nearest_neiber = np.zeros(N)
	# Sort the density from large to small
	index_rho = np.argsort(-rho)
	for i, index in enumerate(index_rho):
		# For the point with the highest density
		if i == 0:
			continue
		# For other points
		# Find the number of the point with a greater density than it
		index_higher_rho = index_rho[:i]
		# Get the distance between these points and the current point, and find the minimum value
		deltas[index] = np.min(dists[index, index_higher_rho])
		# Save the nearest neighbor number
		index_nn = np.argmin(dists[index, index_higher_rho])
		nearest_neiber[index] = index_higher_rho[index_nn].astype(int)
	deltas[index_rho[0]] = np.max(deltas)
	return deltas, nearest_neiber


# Select points with large rho and delta through threshold value
# as cluster centers
def find_centers_auto(rho, deltas):
	rho_threshold = (np.min(rho) + np.max(rho)) * 0.12
	delta_threshold = (np.min(deltas) + np.max(deltas)) * 0.12
	N = np.shape(rho)[0]
	centers = []
	for i in range(N):
		if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
			centers.append(i)
	return centers


# Select the point with the largest product of rho and delta as the cluster center
def find_centers_K(rho, deltas, K):
	rho_delta = rho * deltas
	centers = np.argsort(-rho_delta)
	return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
	K = np.shape(centers)[0]
	if K == 0:
		if verbose:
			print("can not find centers")
		return
	N = np.shape(rho)[0]
	labs = -1 * np.ones(N).astype(int)
	# First, label several clusters
	for i, center in enumerate(centers):
		labs[center] = i
	# Sort the density from large to small
	index_rho = np.argsort(-rho)
	for i, index in enumerate(index_rho):
		# Label the points with high density
		if labs[index] == -1:
			# If it has not been labeled, then the cluster label is the same as the label
			# of the point closest to it and with greater density than it.
			labs[index] = labs[int(nearest_neiber[index])]
	return labs

# Original code by Alaettin Zubaroglu, https://gitlab.com/alaettinzubaroglu/emcstream

from copy import deepcopy

from numpy.random import PCG64
from river import metrics as rmetrics
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import time
import umap.umap_ as umap
import pylab
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix
from sklearn.cluster import KMeans
import datetime
from statistics import mean
from timeit import default_timer
import matplotlib.pyplot as plt


import logging

import warnings
warnings.filterwarnings('ignore')


class EmCDataManager:

    def __init__(self, data_store, label_store):
        self._X = []
        self._labels = []
        current_index = 0
        data_length = 0

        self._X = np.array(data_store)
        self._labels = np.array(label_store)
        self.current_index = 0
        self.data_length = len(self._X)

    def get_data(self, instance_count):
        #print self.current_index, instance_count, self.data_length
        if self.current_index + instance_count <= self.data_length:
            self.current_index = self.current_index + instance_count
            #print(self._X.shape, self._labels.shape)
            #print(self.current_index)
            #print(self.current_index-instance_count, self.current_index)
            if not self._labels is None:
                return True, self._X[self.current_index-instance_count:self.current_index], self._labels[self.current_index-instance_count:self.current_index]
            else:
                return True, self._X[self.current_index-instance_count:self.current_index]
        else:
            return False, self._X[self.current_index:], self._labels[self.current_index:]

    def iterate_index(self, value):
        self.current_index += value

    def data_shape(self):
        return self._X.shape

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    con_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return float(np.sum(np.amax(con_mat, axis=0))) / float(np.sum(con_mat))


class OriginalEmcStream:

    def __init__(self, k, horizon, ari_threshold, ari_threshold_step, seed, init_num_factor, drift_num_factor, drift_inc_delay, consistency_delay, verbose=False):
        self.k = k
        self.horizon = horizon
        self.ari_threshold = ari_threshold
        self.ari_threshold_step = ari_threshold_step
        self.init_num_factor = init_num_factor
        self.drift_num_factor = drift_num_factor
        self.drift_inc_delay = drift_inc_delay
        self.consistency_delay = consistency_delay
        self.generator = np.random.Generator(PCG64(seed))
        self.label_store = None
        self._tlabels = None
        self._X = None
        self._total_embedding = None
        self._total_klabels = None
        self.verbose = verbose

    def learn(self, data, timestamps):         # turn true_labels into indices/timestamps
        verbose = True
        randomState1 = round(self.generator.random()*10000)
        randomState2 = round(self.generator.random()*20000)

        start = default_timer()

        #print('----- new run at : {} -----'.format(datetime.datetime.now().isoformat()))
        #print("Running EmCStream")

        dataManager = EmCDataManager(data,timestamps)
        data_length, feature_cnt = dataManager.data_shape()

        cluster_cnt = self.k

        ari_drift_threshold = self.ari_threshold
        initial_ari_drift_threshold = ari_drift_threshold

        consecutiveDriftCnt = 0
        consecutiveNodriftCnt = 0

        increaseDCPCnt = self.drift_inc_delay # increase the drift check period when $increaseDCPCnt censuctive NOconcept drifts occur, and increase the ari drift threshold

        decreaseThreshold_DriftCnt = self.consistency_delay # decrease the ari drift threshold when $decreaseThreshold_DriftCnt censuctive concept drifts occur

        init_size = self.horizon*self.init_num_factor
        drift_check_period = self.horizon*self.drift_num_factor

        initial_drift_check_period = drift_check_period

        if self.horizon > 5 * cluster_cnt:
            match_fnx_size = self.horizon
        else:
            match_fnx_size = 2*self.horizon

        detected_dirft_count = 0
        if self.verbose:
            print('init size: {}\nhorizon: {}'.format(init_size, self.horizon))    #
            print('drift check period : {}'.format(drift_check_period))
            print('overlap size : {}'.format(match_fnx_size))
            print('cluster count : {}'.format(cluster_cnt))
            print('initial ari_drift_threshold : {}'.format(ari_drift_threshold))
            print('\nExecuting EmCStream...\n')
        total_embedding = np.empty(shape=[0, 2])
        total_klabels = [] # kmeans labels
        total_tlabels = [] # true labels
        total_X = np.empty(shape=[0, feature_cnt])

        no_more_data = False
        match_fnx = None
        drift_occured = False
        very_first_loop = True
        in_init_cycle = True
        #aris = list()
        #purities = list()
        #siluets = list()

        max_ari_tobe_threshold = 0

        while not no_more_data:
            # reinit umap when a concept drift occurs
            embedding = None

            in_init_cycle = True
            not_empty, X, tlabels = dataManager.get_data(init_size)
            reducer = umap.UMAP(random_state = randomState1)
            if not len(X) > 0:
                if self.verbose:
                    print('breaking the loop because X is false, which means no more data.')
                break
            reducer = umap.UMAP(random_state = randomState1)
            randomState1 = randomState1 + 1
            embedding = reducer.fit_transform(X)
            added_X = X
            added_tlabels = tlabels

            drift_occured = False
            while not drift_occured and not no_more_data:
                # come back here when no concept drift occured
                added_instance_cnt = 0
                drift_check_time = False
                while not drift_check_time and not no_more_data:
                    not_empty, X, tlabels = dataManager.get_data(self.horizon)



                    if not not_empty:
                        no_more_data = True
                        if self.verbose:
                            print('setting no_more_data to true.')
                        if len(X) > 0:
                            embedding = np.append(embedding, reducer.transform(X), axis=0)
                            added_X = np.append(added_X, X, axis=0)
                            added_tlabels = np.append(added_tlabels, tlabels)
                        break
                    embedding = np.append(embedding, reducer.transform(X), axis = 0)
                    added_X = np.append(added_X, X, axis = 0)
                    added_tlabels = np.append(added_tlabels, tlabels)

                    added_instance_cnt += self.horizon
                    if added_instance_cnt >= drift_check_period:
                        drift_check_time = True

                if not no_more_data:
                    # it is drift check time
                    if self.verbose:
                        print('calculation kmeans on this window, with k={}'.format(cluster_cnt))
                        print('reinitialize umap with last [{}] data instances, to check for concept drift.'.format(added_instance_cnt))
                    kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(embedding)
                    klabels = [x for x in kmeans.labels_]
                    new_reducer = umap.UMAP(random_state = randomState2)
                    randomState2 = randomState2 + 1
                    new_embedding = new_reducer.fit_transform(added_X)
                    new_kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(new_embedding)
                    new_klabels = [x for x in new_kmeans.labels_]



                    if not very_first_loop and len(total_klabels[-match_fnx_size:]) > 0:
                        # create the match fnx
                        prev_tomatch = total_klabels[-match_fnx_size:]
                        curr_tomatch = klabels[:match_fnx_size]
                        match = np.zeros(shape = [cluster_cnt, cluster_cnt])
                        match_fnx = np.zeros(shape = [cluster_cnt])
                        for i in range(match_fnx_size):
                            match[int(curr_tomatch[i]), int(prev_tomatch[i])] += 1
                        for i in range(cluster_cnt):
                            match_fnx[i] = np.argmax(match[i])
                        # match function is ready
                        if self.verbose:
                            print('---+---+---+---')
                            print(match)
                            print('---')
                            print(match_fnx)
                            print('---+---+---+---')
                        # convert labels to match previous labels
                        for i in range(len(klabels)):
                            klabels[i] = match_fnx[klabels[i]]

                    ari = adjusted_rand_score(klabels, new_klabels)
                    if ari >= ari_drift_threshold:
                        # no drift
                        max_ari_tobe_threshold = 0

                        #true_ari = adjusted_rand_score(klabels, added_tlabels)
                        #purity = purity_score(klabels, added_tlabels)
                        #siluet = silhouette_score(added_X, klabels)
                        #aris.append(true_ari)
                        #purities.append(purity)
                        #siluets.append(siluet)

                        if self.verbose:
                            print('no concept drift yet [{}]'.format(ari))
                        consecutiveDriftCnt = 0
                        consecutiveNodriftCnt = consecutiveNodriftCnt + 1
                        if consecutiveNodriftCnt >= increaseDCPCnt :
                            consecutiveNodriftCnt = 0
                            if drift_check_period < initial_drift_check_period :
                                drift_check_period = drift_check_period + self.horizon
                                if self.verbose:
                                    print('new drift check period (increased) is [{}]'.format(drift_check_period))

                        ari_drift_threshold = ari - self.ari_threshold_step
                        if self.verbose:
                            print('new ari drift threshold (increased) is [{}]'.format(ari_drift_threshold))

                        if very_first_loop:
                            total_embedding = np.append(total_embedding, embedding, axis = 0)
                            total_klabels = total_klabels + klabels
                            total_tlabels = np.append(total_tlabels, added_tlabels)
                            total_X = np.append(total_X, added_X, axis = 0)

                            very_first_loop = False
                        else:
                            total_embedding = np.append(total_embedding, embedding[match_fnx_size:], axis = 0)
                            total_klabels = total_klabels + klabels[match_fnx_size:]
                            total_tlabels = np.append(total_tlabels, added_tlabels[match_fnx_size:])
                            total_X = np.append(total_X, added_X[match_fnx_size:], axis = 0)

                        if self.verbose:
                            print('total embedding : [{}]'.format(total_embedding.shape))
                            print('total klabels : [{}]'.format(len(total_klabels)))
                            print('total tlabels : [{}]'.format(total_tlabels.shape))
                            print('total X : [{}]'.format(total_X.shape))
                    else:
                        drift_occured = True
                        consecutiveNodriftCnt = 0
                        if in_init_cycle:
                            consecutiveDriftCnt += 1
                            if ari > max_ari_tobe_threshold:
                                max_ari_tobe_threshold = ari
                            if self.verbose:
                                print('***** a consecutive concept drift detected.*****[{}]'.format(ari))
                            dataManager.iterate_index(-(added_instance_cnt+init_size))
                            if drift_check_period > match_fnx_size + self.horizon :
                                drift_check_period = drift_check_period - self.horizon
                                if self.verbose:
                                    print('new drift check period (decreased) is [{}]'.format(drift_check_period))
                            elif consecutiveDriftCnt >= decreaseThreshold_DriftCnt:
                                #ari_drift_threshold = ari_drift_threshold - ari_threshold_step
                                ari_drift_threshold = max_ari_tobe_threshold - self.ari_threshold_step
                                max_ari_tobe_threshold = 0
                                if self.verbose:
                                    print('new ari drift threshold (decreased) is [{}]'.format(ari_drift_threshold))
                                consecutiveDriftCnt = 0
                        else:
                            if self.verbose:
                                print('********** a new concept drift detected.**********[{}]'.format(ari))
                            dataManager.iterate_index(-added_instance_cnt)
                            detected_dirft_count += 1

                else:
                    # no more data
                    if self.verbose:
                        print('calculating kmeans on this window, with k={}'.format(cluster_cnt))
                    kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(embedding)
                    klabels = [x for x in kmeans.labels_]

                    # create the match fnx
                    prev_tomatch = total_klabels[-match_fnx_size:]
                    curr_tomatch = klabels[:match_fnx_size]
                    if self.verbose:
                        print('match_fnx_size',match_fnx_size, flush=True)
                        print('curr_tomatch', len(curr_tomatch), curr_tomatch, flush=True)
                        print('prev_tomatch', len(prev_tomatch), prev_tomatch, flush=True)

                    match = np.zeros(shape = [cluster_cnt, cluster_cnt])
                    match_fnx = np.zeros(shape = [cluster_cnt])
                    if self.verbose:
                        print('match', match.shape, match, flush=True)
                        print('match_fnx', match_fnx.shape, match_fnx, flush=True)
                    for i in range(match_fnx_size):
                        if self.verbose:
                            print(i, flush=True)
                        match[int(curr_tomatch[i]), int(prev_tomatch[i])] += 1
                    for i in range(cluster_cnt):
                        match_fnx[i] = np.argmax(match[i])
                    # match function is ready
                    if self.verbose:
                        print('---+---+---+---')
                        print(match)
                        print('---')
                        print(match_fnx)
                        print('---+---+---+---')
                    # convert labels to match previous labels
                    for i in range(len(klabels)):
                        klabels[i] = match_fnx[klabels[i]]
                    print(len(total_embedding))
                    total_embedding = np.append(total_embedding, embedding[match_fnx_size:], axis = 0)
                    total_klabels = total_klabels + klabels[match_fnx_size:]
                    total_tlabels = np.append(total_tlabels, added_tlabels[match_fnx_size:])
                    total_X = np.append(total_X, added_X[match_fnx_size:], axis = 0)

                    print(len(total_embedding))
                    break

                embedding = np.empty(shape=[0, 2])
                added_X = np.empty(shape=[0, feature_cnt])
                added_tlabels = np.empty(shape=[0, 1])
                klabels = []
                if not drift_occured:
                    dataManager.iterate_index(-match_fnx_size)

                in_init_cycle = False

        self._tlabels = total_tlabels
        self._X = total_X
        self._total_embedding = total_embedding
        self._total_klabels = total_klabels
        return total_klabels, []

    def get_clustered_data(self):
        return self._X, self._tlabels

    def get_summary(self):

        return {"embedding": deepcopy(self._total_embedding), "timestamps": deepcopy(self._tlabels), "labels": deepcopy(self._total_klabels)}

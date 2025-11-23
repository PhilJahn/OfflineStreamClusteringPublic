from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from umap import UMAP

from datahandler import load_data
from online.stream_clustering import StreamClusterer
import numpy as np


class EMCStream(StreamClusterer):

    #consistency_threshold handling based on original implementation
    # 1.0 to start, decrements if repeated

    #increaseDCPCnt = 4 # increase the drift check period when $increaseDCPCnt censuctive NOconcept drifts occur, and increase the ari drift threshold

    #decreaseThreshold_DriftCnt = 4 # decrease the ari drift threshold when $decreaseThreshold_DriftCnt censuctive concept drifts occur

    def __init__(self, k, horizon=1000, init_num_factor: int = 2, drift_num_factor: int = 5,
                 drift_num_factor_change: int = 1, drift_inc_delay: int = 2, consistency_threshold=1.0, consistency_drift=0.001,
                 consistency_delay=4, random_state=42, verbose=False):
        super().__init__()
        self.k = k
        self.horizon = horizon
        self.init_num = init_num_factor * horizon

        self.drift_num = drift_num_factor * horizon
        self.drift_num_change = drift_num_factor_change * horizon
        self.max_drift_num = drift_num_factor * horizon
        self.drift_inc_delay = drift_inc_delay

        self.initialized = False

        self.cur_data = []
        self.cur_timestamps = []
        self.past_data = []
        self.past_timestamps = []

        self.embedding = None
        self.embedded_data = []

        self.consistency_threshold = consistency_threshold
        self.consistency_drift = consistency_drift
        self.consistency_delay = consistency_delay
        self.max_consistency_threshold = consistency_threshold

        self.consec_drift = 0
        self.consec_nondrift = 0
        self.loop_depth = 0

        self.random_state = random_state

        self.actual_embedding = []
        self.actual_timestamps = []
        self.actual_labels = []

        self.new_consistency_threshold = -np.inf

        self.verbose = verbose

    def learn_one(self, x, t: int = None):
        event_log = ""
        x = np.array(x)
        self.cur_data.append(x)
        self.cur_timestamps.append(t)
        self.past_data.append(x)
        self.past_timestamps.append(t)
        if len(self.past_data) > self.drift_num:
            self.past_data = self.past_data[1:]
        if not self.initialized:
            if len(self.cur_data) == self.init_num:
                self.embedding = UMAP(random_state=self.loop_depth, n_jobs=1).fit(self.cur_data) # self.loop_depth to ensure different seeds for loops
                self.embedded_data = self.embedding.embedding_.tolist()
                if self.verbose:
                    print("init:", self.loop_depth)
                self.actual_embedding.extend(self.embedded_data)
                self.actual_timestamps.extend(self.cur_timestamps)
                self.initialized = True
                # print("init", self.cur_data[-5:])
                self.cur_data = []
                self.cur_timestamps = []
                event_log += "initialized;"
            else:
                return -1, ""
        else:
            if len(self.cur_data) == self.horizon:
                additional_embedded_data = self.embedding.transform(self.cur_data).tolist()
                self.embedded_data.extend(additional_embedded_data)
                self.actual_embedding.extend(additional_embedded_data)
                self.actual_timestamps.extend(self.cur_timestamps)
                event_log += "embedded;"
                #print("emb", self.cur_data[-5:])
                # print(self.cur_data)
                self.cur_data = []
                self.cur_timestamps = []
        if len(self.past_data) == self.drift_num:
            #print("check", self.past_data[-5:])
            embedding2 = UMAP(random_state=self.loop_depth+1, n_jobs=1).fit(self.past_data) # use different seed from init embedding to prevent trivial match and to properly calibrate threshold
            embedded_data2 = embedding2.embedding_.tolist()
            event_log += "drift check;"
            temp_data = self.past_data
            temp_timestamps = self.past_timestamps
            # plt.figure(figsize=(6, 6))
            # plt.scatter(np.array(self.embedded_data)[:, 0], np.array(self.embedded_data)[:, 1])
            # plt.show()
            # plt.figure(figsize=(6, 6))
            # plt.scatter(np.array(embedded_data2)[:, 0], np.array(embedded_data2)[:, 1])
            # plt.show()
            #print(len(self.embedded_data), len(embedded_data2))
            # print(self.past_data)
            clustering = KMeans(n_clusters=self.k, random_state=self.random_state).fit_predict(self.embedded_data)
            self.actual_labels.extend(clustering)
            clustering2 = KMeans(n_clusters=self.k, random_state=self.random_state).fit_predict(embedded_data2)
            consistency = adjusted_rand_score(clustering, clustering2)
            if self.verbose:
                print("consistency: ", consistency)
            # clear out stored samples/embeddings
            self.past_data = []
            self.past_timestamps = []
            self.embedded_data = []
            if consistency >= self.consistency_threshold:
                self.consec_drift = 0
                self.consec_nondrift += 1
                self.loop_depth = 0
                #print("consec no drift:", self.consec_nondrift)
                if self.drift_num < self.max_drift_num and self.consec_nondrift % self.drift_inc_delay == 0:
                    self.drift_num = min(self.drift_num + self.drift_num_change, self.max_drift_num)
                    if self.verbose:
                        print("drift num (+):", self.drift_num)
                    event_log += "driftnum+;"

                if self.consec_nondrift % self.consistency_delay == 0:
                    self.consistency_threshold = min(consistency - self.consistency_drift,
                                                     self.max_consistency_threshold)
                    self.new_consistency_threshold = -np.inf
                    event_log += "ct+;"
                    if self.verbose:
                        print("threshold changed (+):", self.consistency_threshold)
            else:
                self.consec_nondrift = 0
                self.consec_drift += 1
                if consistency > self.new_consistency_threshold:
                    self.new_consistency_threshold = consistency
                if self.drift_num > self.init_num:
                    self.drift_num = max(self.init_num, self.drift_num-self.drift_num_change)
                    event_log += "driftnum-;"
                    if self.verbose:
                        print("drift num (-):", self.drift_num)
                elif self.consec_drift >= self.consistency_delay:
                    self.consistency_threshold = self.new_consistency_threshold - self.consistency_drift
                    self.new_consistency_threshold = -np.inf
                    self.consec_drift = 0
                    event_log += "ct-;"
                    if self.verbose:
                        print("threshold changed (-):", self.consistency_threshold)
                self.loop_depth = self.loop_depth + 1
                self.initialized = False
                self.actual_embedding = self.actual_embedding[:-len(temp_data)]
                self.actual_timestamps = self.actual_timestamps[:-len(temp_data)]
                self.actual_labels = self.actual_labels[:-len(temp_data)]
                #print("reduced actual embedding data to", len(self.actual_embedding))
                ids, event_log_loop = self.learn(temp_data, temp_timestamps) # rewind
                #print(event_log_loop)
                event_log += str(event_log_loop).replace("'',", "").replace("\\", "").replace(" ", "")
        return -1, event_log

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs

    def get_summary(self):
        embeddings = deepcopy(self.actual_embedding)
        # plt.figure(figsize=(6, 6))
        # plt.scatter(np.array(embeddings)[-self.drift_num:, 0], np.array(embeddings)[-self.drift_num:, 1])
        # plt.title(f"old data {len(embeddings[-self.drift_num:])}, {-self.drift_num}")
        # plt.show()
        # plt.close()
        timestamps = deepcopy(self.actual_timestamps)
        labels = deepcopy(self.actual_labels)
        unlabeled = len(embeddings) - len(labels)
        if unlabeled > 0:
            embeddings2label = embeddings[-unlabeled:]
        else:
            embeddings2label = []
        if len(self.cur_data) > 0:
            additional_embedded_data = self.embedding.transform(self.cur_data).tolist()

            embeddings.extend(additional_embedded_data)
            embeddings2label.extend(additional_embedded_data)
            timestamps.extend(self.cur_timestamps)
        #print(len(embeddings2label))
        # plt.figure(figsize=(6, 6))
        # plt.scatter(np.array(embeddings2label)[:, 0], np.array(embeddings2label)[:, 1])
        # plt.title(f"new data {len(embeddings2label)}")
        # plt.show()
        # plt.close()
        if len(embeddings2label) > 0:
            clustering = KMeans(n_clusters=self.k, random_state=self.random_state).fit_predict(embeddings2label)
            #print(len(clustering))
            labels.extend(clustering)

        #print(len(embeddings), len(timestamps), len(labels))

        return {"embedding": deepcopy(embeddings), "timestamps": deepcopy(timestamps), "labels": deepcopy(labels)}


if __name__ == "__main__":
    make_fig = True
    stream, labels = load_data("complex9")  #[[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    emcstream = EMCStream(k=9, horizon=200, consistency_delay=4)#, consistency_drift=1.0, consistency_threshold=0)
    for i in range(len(stream)):
        cell_id, event_log = emcstream.learn_one(stream[i], ts[i])
        batch_start = max(0, i - 500)
        if ";" in event_log:
            print(i, event_log)
    summaries = emcstream.get_summary()
    print(summaries[-5:])
    clulabels = []

    for i in range(len(summaries)):
        clulabels.append(summaries[i][2])

    clulabels = np.array(clulabels)
    for i in range(len(labels)):
        if i % 200 == 0 and i != 0:
            plt.figure(figsize=(8, 8))
            batch_start = max(0, i - 200)
            # print(len(np.array(stream)[batch_start:i]))
            plt.scatter(x=np.array(stream)[batch_start:i][:, 0], y=np.array(stream)[batch_start:i][:, 1],
                        cmap=plt.get_cmap("jet"), c=clulabels[batch_start:i], s=10)
            plt.savefig(f"test_figures/emcstream/complex9_emcstream_{i}.png", bbox_inches='tight')
            plt.close()

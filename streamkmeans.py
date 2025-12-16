import numpy as np
from sklearn.cluster import KMeans

from datahandler import load_data
from online.stream_clustering import StreamClusterer


class STREAMKmeans(StreamClusterer):

    def __init__(self, k, chunk_size, prototype_num, random_state=42):
        super().__init__()
        self.chunk_size = chunk_size
        self.k = k
        self.prototype_num = prototype_num
        self.random_state = random_state

        self.cur_data = []
        self.cur_timestamps = []

        self.prototypes = []

    def learn_one(self, x, t: int = None):
        event_log = ""
        x = np.array(x)
        self.cur_data.append(x)
        self.cur_timestamps.append(t)

        if len(self.cur_data) == self.chunk_size:
            self.cur_data = np.array(self.cur_data)

            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
            clustering = kmeans.fit_predict(self.cur_data)

            clu_labels, index_cluster, inverse_cluster, cluster_sizes = np.unique(clustering, return_index = True, return_counts=True, return_inverse=True)
            #print(inverse_cluster)
            #centroids = kmeans.cluster_centers_
            # print(centroids)
            # unicentroids, inverse_centroid = np.unique(kmeans.cluster_centers_, axis=0, return_inverse=True)
            # print(inverse_centroid)

            #print(len(np.unique(centroids, axis=0)), len(centroids))
            #print(clu_labels, cluster_sizes)#, centroids)
            #print(inverse_cluster, inverse_centroid)
            #print(inverse_cluster)
            for c in range(len(clu_labels)):
                #center = centroids[c]
                clu_points = self.cur_data[np.where(inverse_cluster == c)[0]]
                center = np.mean(clu_points, axis=0)
                weight = cluster_sizes[c]
                # if center not in centroids:
                #     print(center)
                #     print(centroids)
                #assert center in centroids
                #print(c, clu_labels[c], weight, len(clu_points), flush=True)
                assert len(clu_points) == weight
                p = StreamPrototype(center, weight)
                self.prototypes.append(p)

            self.cur_data = []
            self.cur_timestamps = []
            event_log += "chunk;"

        if len(self.prototypes) > self.prototype_num:
            proto_centers = [p.center for p in self.prototypes]
            proto_weight = [p.weight for p in self.prototypes]
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
            clustering = kmeans.fit_predict(proto_centers, sample_weight=proto_weight)
            centroids = kmeans.cluster_centers_
            self.prototypes = []
            for c in range(len(centroids)):
                center = centroids[c]
                #print("proto weights", proto_weight)
                #print("clusters", np.where(clustering == c)[0])
                weight = np.sum(np.array(proto_weight)[np.where(clustering == c)[0]])
                #print("weight", weight)
                p = StreamPrototype(center, weight)
                self.prototypes.append(p)
            event_log += "compress;"

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
        return dict(zip(list(range(len(self.prototypes))), self.prototypes))

class StreamPrototype():
    def __init__(self, center, weight):
        self.center = center
        self.weight = weight



import matplotlib.pyplot as plt
if __name__ == "__main__":
    make_fig = False

    stream, labels = load_data("kddcup")  #[[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    cstream = STREAMKmeans(k=100, chunk_size=100, prototype_num=100)
    for i in range(len(stream)):
        cell_id, event_log = cstream.learn_one(stream[i], ts[i])
        batch_start = max(0, 0)
        if ";" in event_log:
            print(i, event_log.replace("\n", ""))
            batch_end = min(len(stream), i + 1)
            summaries = cstream.get_summary()
            if make_fig:
                plt.figure(figsize=(8, 8))
                plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
                            cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
                for p in summaries:
                    plt.scatter(p.center[0], p.center[1], s=p.weight*10, c="black")
                plt.ylim((0, 1))
                plt.xlim((0, 1))
                #plt.show()
                plt.savefig(f"test_figures/streamkmeans/complex9_streamkmeans_{i}.png", bbox_inches='tight')
                plt.close()

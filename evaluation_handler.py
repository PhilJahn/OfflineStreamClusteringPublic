from sklearn.utils import _safe_indexing
import numpy as np
from s_dbw import S_Dbw

from clustpy.metrics import purity, unsupervised_clustering_accuracy, fair_normalized_mutual_information, \
    variation_of_information, information_theoretic_external_cluster_validity_measure
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, \
    fowlkes_mallows_score, homogeneity_score, completeness_score, silhouette_score, davies_bouldin_score, \
    calinski_harabasz_score

from cmm import get_cmm
from temporalsilhouette.TSindex import tempsil
from utils import get_noise_labels

def get_full_evaluation(online_method, offline_method, batch_size, dataset, data_points, labels):
    batch_num = np.ceil(len(data_points) / batch_size)

def evaluate_clustering(data_batches, label_batches, clusterings, eval_mode):
    metrics = {"purity": 0.0, "accuracy": 0.0, "ari": 0.0, "nmi": 0.0, "ami": 0.0, "cmm": 0.0, "cmm_alt": 0.0,
               "fowlkes_mallows": 0.0, "homogeneity": 0.0, "completeness": 0.0, "fair_nmi": 0.0,
               "variation_of_information": 0.0, "information_theoretic_external_cluster_validity_measure": 0.0,
               "silhouette_score": 0.0, "davies_bouldin_score": 0.0, "calinski_harabasz_score": 0.0,
               "s_dbw": 0.0, "temporal_silhouette_index": 0.0,
               }

    labels = []
    data_points = []

    for i in range(len(clusterings)):
        cur_labels = label_batches[i]
        labels.extend(cur_labels)
        # print(cur_labels)
        cur_preds = clusterings[i].reshape(len(cur_labels))
        cur_preds = get_noise_labels(cur_preds)

        cur_data = data_batches[i]
        data_points.extend(cur_data)

        _, cur_labels = np.unique(cur_labels, return_inverse=True)
        _, cur_preds = np.unique(cur_preds, return_inverse=True)

        if eval_mode == "purity":
            cur_purity = purity(cur_labels, cur_preds)
            metrics["purity"] += cur_purity * len(cur_labels)
        elif eval_mode == "ariami":
            cur_ami = adjusted_mutual_info_score(cur_labels, cur_preds)
            metrics["ami"] += cur_ami * len(cur_labels)
            cur_ari = adjusted_rand_score(cur_labels, cur_preds)
            metrics["ari"] += cur_ari * len(cur_labels)
        else:
            cur_purity = purity(cur_labels, cur_preds)
            metrics["purity"] += cur_purity * len(cur_labels)
            cur_acc = unsupervised_clustering_accuracy(cur_labels, cur_preds)
            metrics["accuracy"] += cur_acc * len(cur_labels)
            cur_ami = adjusted_mutual_info_score(cur_labels, cur_preds)
            metrics["ami"] += cur_ami * len(cur_labels)
            cur_nmi = normalized_mutual_info_score(cur_labels, cur_preds)
            metrics["nmi"] += cur_nmi * len(cur_labels)
            cur_ari = adjusted_rand_score(cur_labels, cur_preds)
            metrics["ari"] += cur_ari * len(cur_labels)
            cur_cmm, cur_cmm_alt = get_cmm(cur_data, cur_labels, cur_preds, np.ones(len(cur_labels)), k=5,
                                           alternate=True)
            metrics["cmm"] += cur_cmm * len(cur_labels)
            metrics["cmm_alt"] += cur_cmm_alt * len(cur_labels)
            cur_fm = fowlkes_mallows_score(cur_labels, cur_preds)
            metrics["fowlkes_mallows"] += cur_fm * len(cur_labels)
            cur_homo = homogeneity_score(cur_labels, cur_preds)
            metrics["homogeneity"] += cur_homo * len(cur_labels)
            cur_comp = completeness_score(cur_labels, cur_preds)
            metrics["completeness"] += cur_comp * len(cur_labels)
            cur_fnmi = fair_normalized_mutual_information(cur_labels, cur_preds)
            metrics["fair_nmi"] += cur_fnmi * len(cur_labels)
            cur_voi = variation_of_information(cur_labels, cur_preds)
            metrics["variation_of_information"] += cur_voi * len(cur_labels)
            cur_itevcm = information_theoretic_external_cluster_validity_measure(cur_labels, cur_preds)
            metrics["information_theoretic_external_cluster_validity_measure"] += cur_itevcm * len(cur_labels)
            if len(np.unique(cur_labels)) > 1:
                cur_sil = silhouette_score(cur_data, cur_preds)
            else:
                cur_sil = -1  # worst case
            metrics["silhouette_score"] += cur_sil * len(cur_labels)
            if len(np.unique(cur_labels)) > 1:
                cur_db = davies_bouldin_score(cur_data, cur_preds)
            else:
                max_db = 0  # undefined -> worst case, but uncapped -> use max 1 vs all score
                for i in range(len(cur_data)):
                    fake_labels = [0] * len(cur_data)
                    fake_labels[i] = 1
                    dbs = davies_bouldin_score(cur_data, fake_labels)
                    if max_db < dbs:
                        max_db = dbs
                cur_db = max_db
            metrics["davies_bouldin_score"] = cur_db * len(cur_labels)
            if len(np.unique(cur_labels)) > 1:
                cur_ch = calinski_harabasz_score(cur_data, cur_preds)
            else:
                max_ch = 0  # undefined -> worst case, but uncapped -> use max 1 vs all score
                for i in range(len(cur_data)):
                    fake_labels = [0] * len(cur_data)
                    fake_labels[i] = 1
                    ch = calinski_harabasz_score(cur_data, fake_labels)
                    if max_ch < ch:
                        max_ch = ch
                cur_ch = max_ch
            metrics["calinski_harabasz_score"] = cur_ch * len(cur_labels)

            sdbw = S_Dbw(cur_data, cur_preds, centers_id=None, method='Tong', alg_noise='bind',
                         centr='mean', nearest_centr=True, metric='euclidean')
            metrics["s_dbw"] += sdbw * len(cur_labels)
    #print(len(data_points))
    #print(len(labels))

    if eval_mode == "all":
        min_clu = 0
        pred_all = []
        for i in range(len(clusterings)):
            cur_labels = labels[i]
            cur_preds = clusterings[i].reshape(len(cur_labels))
            cur_preds = get_noise_labels(cur_preds)
            mapping = {}
            mapping_overlap = {}
            rev_mapping = {}
            rev_mapping_overlap = {}
            for label in np.unique(cur_labels):
                rev_mapping_overlap[label] = 0
            for clu in np.unique(cur_preds):
                clu_ids = np.where(cur_preds == clu)[0]
                # print(clu, clu_ids)
                max_overlap = 0
                for label in np.unique(cur_labels):
                    label_ids = np.where(cur_labels == label)[0]
                    overlap = len(np.intersect1d(clu_ids, label_ids))
                    # print(label, clu, overlap)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        mapping[clu] = label
                        mapping_overlap[clu] = overlap
                label_max_overlap = rev_mapping_overlap[mapping[clu]]
                if label_max_overlap < mapping_overlap[clu]:
                    rev_mapping[mapping[clu]] = clu
                    rev_mapping_overlap[mapping[clu]] = mapping_overlap[clu]
            assignment = {}
            for clu in np.unique(cur_preds):
                if clu == -1:
                    assignment[clu] = -1
                else:
                    if clu == rev_mapping[mapping[clu]]:
                        assignment[clu] = mapping[clu]
                    else:
                        min_clu += 1
                        assignment[clu] = min_clu

            for pred in cur_preds:
                pred_all.append(assignment[pred])
        timesteps = np.arange(len(data_points))
        _, _, TS = tempsil(timesteps, data_points, pred_all, s=100, kn=1000, c=1)
        metrics["temporal_silhouette_index"] = TS

    for metric in metrics.keys():
        metrics[metric] = metrics[metric] / len(labels)
    return metrics

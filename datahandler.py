import math
import os

#from capymoa.datasets import Electricity, RBFm_100k, Hyper100k # commented out as no Java on server
import numpy as np
import sklearn
from clustpy.metrics import unsupervised_clustering_accuracy, purity
from numpy.random import PCG64
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ucimlrepo import fetch_ucirepo

from defines import complete_real_datasets
from temporalsilhouette.TSindex import tempsil

def read_file(dsname):
    dic = {}
    classmember = 0
    try:
        data = np.load(f"./data/{dsname}.npy")
        x = data[:, :-1]
        label = data[:, -1]
        return x, label
    except FileNotFoundError:
        if dsname == "kddcup99":
            data, label = sklearn.datasets.fetch_kddcup99(return_X_y =True, download_if_missing=True, data_home="./data/sklearn_download/")
            symbolic_columns = [1,2,3,6,11,20,21]
            continuous_columns = list(range(len(data[0])))
            for i in symbolic_columns:
                continuous_columns.remove(i)

            data = data[:, continuous_columns]
            _, label = np.unique(label, return_inverse=True)
            return data, label
        elif dsname == "covertype":
            data, label = sklearn.datasets.fetch_covtype(return_X_y =True, download_if_missing=True, data_home="./data/sklearn_download/")
            _, label = np.unique(label,return_inverse=True)
            return data, label
        elif dsname == "pokerhand": # UCI: 10.24432/C5KW38, Robert Cattral, Franz Oppacher, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/pokerhand_label.npy"):
                pokerhand = fetch_ucirepo(id=158)
                X = pokerhand.data.features.to_numpy()
                y = pokerhand.data.targets.to_numpy()
                np.save("./data/uci_download/pokerhand_data.npy", X)
                np.save("./data/uci_download/pokerhand_label.npy", y)
            X = np.load("./data/uci_download/pokerhand_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/pokerhand_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "pendigits": # UCI: DOI:10.24432/C5MG6K, E. Alpaydin, Fevzi. Alimoglu, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/pendigits_label.npy"):
                pendigits = fetch_ucirepo(id=81)
                X = pendigits.data.features.to_numpy()
                y = pendigits.data.targets.to_numpy()
                np.save("./data/uci_download/pendigits_data.npy", X)
                np.save("./data/uci_download/pendigits_label.npy", y)
            X = np.load("./data/uci_download/pendigits_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/pendigits_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "sepsissurvival": # UCI: Davide Chicco, Giuseppe Jurman, Survival prediction of patients with sepsis from age, sex, and septic episode number alone. Scientific Reports 10, 17156 (2020) https://doi.org/10.1038/s41598-020-73558-3, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/sepsissurvival_label.npy"):
                sepsissurvival = fetch_ucirepo(id=827)
                X = sepsissurvival.data.features.to_numpy()
                y = sepsissurvival.data.targets.to_numpy()
                np.save("./data/uci_download/sepsissurvival_data.npy", X)
                np.save("./data/uci_download/sepsissurvival_label.npy", y)
            X = np.load("./data/uci_download/sepsissurvival_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/sepsissurvival_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "htru2": # UCI: R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
                                # R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1., licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/htru2_label.npy"):
                htru2 = fetch_ucirepo(id=372)
                X = htru2.data.features.to_numpy()
                y = htru2.data.targets.to_numpy()
                np.save("./data/uci_download/htru2_data.npy", X)
                np.save("./data/uci_download/htru2_label.npy", y)
            X = np.load("./data/uci_download/htru2_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/htru2_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "phiusiil": # UCI: Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. doi: https://doi.org/10.1016/j.cose.2023.103545, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/uci_download/phiusiil_label.npy"):
                phiusiil = fetch_ucirepo(id=967)
                X = phiusiil.data.features.to_numpy()
                numeric_dims = []
                for d in range(len(X[0])):
                    values = np.unique(X[:,d])
                    if "float" in str(type(values[0])) or "int" in str(type(values[0])):
                        numeric_dims.append(d)
                X = X[:, numeric_dims]
                y = phiusiil.data.targets.to_numpy()
                np.save("./data/uci_download/phiusiil_data.npy", X)
                np.save("./data/uci_download/phiusiil_label.npy", y)
            X = np.load("./data/uci_download/phiusiil_data.npy", allow_pickle=True)
            y = np.load("./data/uci_download/phiusiil_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "har": # UCI: DOI:10.24432/C54S4K, Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/real_uci/HAR/HAR_data.npy"):
                X = []
                file = open("./data/real_uci/HAR/X_test.txt")
                for line in file:
                    linesplit = line.split(" ")
                    x = []
                    for split in linesplit:
                        if len(split) > 0:
                            x.append(float(split))
                    X.append(x)
                file = open("./data/real_uci/HAR/X_train.txt")
                for line in file:
                    linesplit = line.split(" ")
                    x = []
                    for split in linesplit:
                        if len(split) > 0:
                            x.append(float(split))
                    X.append(x)
                X = np.array(X)
                #X_train = np.loadtxt("./data/uci/HAR/X_train.txt", delimiter=" ", skiprows=0, dtype=str)
                y_test = np.loadtxt("./data/real_uci/HAR/y_test.txt", delimiter=" ", skiprows=0, dtype=np.int32)
                y_train = np.loadtxt("./data/real_uci/HAR/y_train.txt", delimiter=" ", skiprows=0, dtype=np.int32)
                y = y_test.tolist()
                y.extend(y_train.tolist())
                y = np.array(y).reshape(1, len(y))[0]
                np.save("./data/real_uci/HAR/HAR_data.npy", X)
                np.save("./data/real_uci/HAR/HAR_label.npy", y)
            X = np.load("./data/real_uci/HAR/HAR_data.npy", allow_pickle=True)
            y = np.load("./data/real_uci/HAR/HAR_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "adult":
            # UCI: DOI:10.24432/C5XW20, Barry Becker, Ronny Kohavi, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            # Preprocessed: Pustozerova, A., & Schuster, V. (2024). Adult dataset preprocessed [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12533514, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/adult/adult_full.txt"):
                file1 = open("./data/adult/adult_train.csv", "r")
                file1_lines = file1.readlines()
                #print(len(file1_lines))
                file2 = open("./data/adult/adult_test.csv", "r")
                file2_lines = file2.readlines()
                #print(len(file2_lines))
                file3 = open("./data/adult/adult_full.txt", "w")
                for line in file1_lines:
                    file3.write(line)
                for line in file2_lines:
                    file3.write(line)
                file3.close()
            file = open("./data/adult/adult_full.txt", "r")
        # elif dsname == "identity_test":
        #     X = np.ones((4000, 5))
        #     y = np.zeros(4000)
        #     return X, y
        elif dsname == "starlightcurves":
            file = open("./data/usp_download/StarLightCurves.csv", "r")
        elif dsname == "mirs":
            file = open("./data/usp_download/MIRS.csv", "r")
        elif dsname == "electricity":
            file = open("./data/usp_download/Electricity.csv", "r")
        elif dsname == "sensorstream":
            file = open("./data/usp_download/SensorStream.csv", "r")
        elif dsname == "heartbeats":
            file = open("./data/usp_download/HeartBeats.csv", "r")
        elif dsname == "cancer":
            file = open("./data/usp_download/Cancer.csv", "r")
        elif dsname == "posture":
            #fixed linebreaks at lying1.1695634126663 (swapped to lying\n1.1695634126663)
            file = open("./data/usp_download/Posture.csv", "r")
        elif dsname == "smartmeter":
            #fixed linebreak at 30.0918 (swapped to 3\n0.0918)
            file = open("./data/usp_download/SmartMeter_LADPU.csv", "r")
        elif dsname == "uwave":
            file = open("./data/usp_download/UWave.csv", "r")
        elif dsname == "yoga":
            file = open("./data/usp_download/Yoga.csv", "r")
        elif dsname == "asfault":
            #fixed linebreaks at good0.5288053160 (swapped to good\n0.5288053160)
            file = open("./data/usp_download/Asfault.csv", "r")
        elif dsname == "iabrupt":
            file = open("./data/usp_download/INSECTS abrupt_balanced.csv", "r")
        elif dsname == "iincremental":
            file = open("./data/usp_download/INSECTS incremental_balanced.csv", "r")
        elif dsname == "ireocurring":
            file = open("./data/usp_download/INSECTS incremental-reoccurring_balanced.csv", "r")
        elif dsname == "igradual":
            file = open("./data/usp_download/INSECTS gradual_balanced.csv", "r")
        elif dsname == "iabruptreocurring":
            file = open("./data/usp_download/INSECTS incremental-abrupt_balanced.csv", "r")
        elif dsname == "powersupply":
            #removed extra "," in 128.3,93.3,2,
            file = open("./data/usp_download/Powersupply.csv", "r")
        elif dsname == "outdoor":
            file = open("./data/usp_download/Outdoor.csv", "r")
        elif dsname == "noaa":
            file = open("./data/usp_download/NOAA.csv", "r")
        elif dsname == "gassensor":
            file = open("./data/usp_download/GasSensorArray.csv", "r")
        elif dsname == "rialto":
            file = open("./data/usp_download/Rialto.csv", "r")
        elif dsname == "stream_densired":
            X = np.load("./data/stream_densired_data.npy", allow_pickle=True)
            y = np.load("./data/stream_densired_label.npy", allow_pickle=True)
            y = y.reshape(1, len(y))[0]
            return X, y
        elif dsname == "rbf3":
            file = open("./data/RBF3_40000.csv", "r")
        else:
            raise FileNotFoundError
    x = []
    label = []
    for line in file:
        if (line.startswith("@") or line.startswith("%") or "class" in line or "duration" in line or len(
                line.strip()) == 0 or "fnlwgt" in line):
            pass
        else:
            j = line.split(",")
            if ("?" in j):
                continue
            k = []


            if dsname == "adult":
                temp = j.pop(6) # prediction target is at position 6, move to end
                j.append(temp)
            for i in range(len(j) - 1):
                k.append(float(j[i]))
            if (not j[len(j) - 1].startswith("noise") and not j[len(j) - 1].startswith("-1")):
                clsname = j[len(j) - 1].rstrip()
                if (clsname in dic.keys()):
                    label.append(dic[clsname])
                else:
                    dic[clsname] = classmember
                    label.append(dic[clsname])
                    classmember += 1
            else:
                label.append(-1)
            x.append(k)
    #print(x[0])
    file.close()
    return np.array(x), np.array(label).reshape(1, len(label))[0]

def load_data(dsname, seed=0, batch=None, batch_max=None, scale=True, standard=False):
    X, y = read_file(dsname)
    X = X.astype(np.double)
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    if standard:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


    #y = get_noise_labels(y)

    if batch is not None and batch_max is not None:
        X = X[:batch*batch_max]
        y = y[:batch*batch_max]
    return X, y

if __name__ == "__main__":
    dss = ["mirs", "starlightcurves", "har", "pendigits", "electricity", "adult", "kddcup99", "covertype",
     "pokerhand", "sensorstream",
     "iabrupt", "iincremental", "ireocurring", "igradual", "iabruptreocurring",
    "heartbeats", "cancer", "posture", "smartmeter", "uwave", "yoga", "asfault",
        "powersupply", "outdoor", "noaa", "gassensor", "rialto",
           "phiusiil", "sepsissurvival", "htru2"]

    #dss = complete_real_datasets

    dss.sort()

    dss.append("stream_densired")
    dss.append("rbf3")




    ds_table_str = ""
    timesum = 0
    timesum2 = 0
    for ds in dss:
        data, label = load_data(ds)
        batch_base_time = 900
        batch_size = 1000
        batch_num = int(np.ceil(len(data) / batch_size))
        batch_scale = math.log10(len(data[0]))
        budget = math.ceil(batch_base_time * batch_scale) * batch_num

        match_prior = 1
        for i in range(len(label)-1):
            if label[i] == label[i+1]:
                match_prior += 1

        kmeans_accs = []
        kmeans_aris = []
        kmeans_amis = []
        kmeans_purs = []
        batch_ids = np.arange(0, len(data), step=batch_size).tolist()
        batch_ids.append(len(data))
        for batch in range(len(batch_ids) - 1):
            stream_batch = data[batch_ids[batch]:batch_ids[batch + 1]]
            stream_eval_batch = data[batch_ids[max(0, batch)]:batch_ids[batch + 1]]
            label_eval_batch = label[batch_ids[max(0, batch)]:batch_ids[batch + 1]]
            clunum = len(np.unique(label_eval_batch))
            # if clunum > 1:
            clustering = KMeans(n_clusters=clunum, random_state=0).fit_predict(stream_eval_batch)
            #print(clustering)
            #print(label_eval_batch)
            acc = unsupervised_clustering_accuracy(labels_true=label_eval_batch, labels_pred=clustering)
            ari = adjusted_rand_score(labels_true=label_eval_batch, labels_pred=clustering)
            ami = adjusted_mutual_info_score(labels_true=label_eval_batch, labels_pred=clustering)
            pur = purity(labels_true=label_eval_batch, labels_pred=clustering)
            # else:
            #     acc = 1.0
            #     ari = 1.0
            #     ami = 1.0
            #     pur = 1.0
            kmeans_accs.append(acc)
            kmeans_aris.append(ari)
            kmeans_amis.append(ami)
            kmeans_purs.append(pur)

        timesteps = np.arange(len(data))
        #_, _, TS_100_1000 = tempsil(timesteps, data, label, s=100, kn=1000, c=1)

        nochange_acc = match_prior / len(label)

        #print(f"{ds}: N: {len(label)}, d: {data.shape[1]} , k: {len(np.unique(label))}, nca: {nochange_acc:.3f}, "
        #      f"kacc: {np.mean(kmeans_accs):.3f}, "
        #      f"kari: {np.mean(kmeans_aris):.3f}, "
        #      f"kami: {np.mean(kmeans_amis):.3f}, "
        #      f"kpur: {np.mean(kmeans_purs):.3f}, "
        #      f"b: {budget}s, {budget / 60:.2f}m, {budget / 3600:.2f}h, {budget / (3600 * 24):.2f}d")

        timesum += budget
        #print(f"Time: {timesum}s, {timesum / 60:.2f}m, {timesum / 3600:.2f}h, {timesum / (3600 * 24):.2f}d")

        #if np.mean(kmeans_aris) > 0.1:
        #    timesum2 += budget
        #    print(f"Alt. Time: {timesum2}s, {timesum2 / 60:.2f}m, {timesum2 / 3600:.2f}h, {timesum2 / (3600 * 24):.2f}d")

        category = "Synth." if ds in ["stream_densired", "rbf3"] else "Real"

        table_line = f"{ds} & & {category} & {len(label)} & {data.shape[1]} & {len(np.unique(label))} & {nochange_acc:.3f} & {budget / 3600:.2f}h\\\\"
        print(table_line, flush=True)

        ds_table_str += f'["{ds}"]={len(batch_ids)-2} '

    print(ds_table_str)
    print(f"{timesum / 3600:.2f}h")
""

    # for ds in ["stream_densired", "rbf3", "har", "electricity", "insects_abrupt"]:
    #     data, label = load_data(ds)
    #
    #     match_prior = 1
    #     for i in range(len(label)-1):
    #         if label[i] == label[i+1]:
    #             match_prior += 1
    #
    #     nochange_acc = match_prior / len(label)
    #
    #     tss = []
    #     for kn in [200, 400, 600, 800, 1000]:
    #         for s in [20, 50, 100, 200, 500]:
    #             tss.append(tempsil(t=np.arange(len(data)), x=data, l=label, c=1, kn=kn, s=s)[2])
    #
    #     ts = np.mean(tss)
    #
    #     print(f"{ds} & {len(label)} & {data.shape[1]} & {len(np.unique(label))} & {nochange_acc:.3f} & {ts:.3f}")


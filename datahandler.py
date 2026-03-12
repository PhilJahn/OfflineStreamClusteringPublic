import os

#from capymoa.datasets import Electricity, RBFm_100k, Hyper100k # commented out as no Java on server
import numpy as np
import sklearn
from clustpy.metrics import unsupervised_clustering_accuracy
from numpy.random import PCG64
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ucimlrepo import fetch_ucirepo

from temporalsilhouette.TSindex import tempsil

def read_file(dsname):
    dic = {}
    classmember = 0
    if dsname == "kddcup":
        data, label = sklearn.datasets.fetch_kddcup99(return_X_y =True, download_if_missing=True, data_home="./data/sklearn_download/")
        _, label = np.unique(label,return_inverse=True)
        third_uniques = dict(enumerate(np.sort(np.unique(np.array(data)[:, 2]))))
        # from https://stackoverflow.com/questions/1031851/how-do-i-exchange-keys-with-values-in-a-dictionary
        third_dict = dict((v, k) for k, v in third_uniques.items())
        fourth_uniques = dict(enumerate(np.sort(np.unique(np.array(data)[:, 3]))))
        fourth_dict = dict((v, k) for k, v in fourth_uniques.items())

        for k in data:
            if k[1] == b'tcp':
                k[1] = 1
            elif k[1] == b'udp':
                k[1] = 2
            elif k[1] == b'icmp':
                k[1] = 3
            else:
                raise ValueError
            k[2] = third_dict[k[2]]
            k[3] = fourth_dict[k[3]]
        return data, label
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
    elif dsname == "stream_densired":
        X = np.load("./data/stream_densired_data.npy", allow_pickle=True)
        y = np.load("./data/stream_densired_label.npy", allow_pickle=True)
        y = y.reshape(1, len(y))[0]
        return X, y
    elif dsname == "rbf3":
        file = open("./data/RBF3_40000.csv", "r")
    elif dsname == "insects_abrupt":
        file = open("./data/usp_ds_data/INSECTS abrupt_balanced.csv", "r")
    elif dsname == "insects_incremental":
        file = open("./data/usp_ds_data/INSECTS incremental_balanced.csv", "r")
    elif dsname == "electricity":
        if not os.path.exists("./data/capymoa_download/"):
            os.mkdir("./data/capymoa_download/")
        #if not os.path.exists("./data/capymoa_download/electricity.arff"):
            #Electricity(directory="./data/capymoa_download/")
        file = open("./data/capymoa_download/electricity.arff", "r")
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

            if dsname == "kddcup":
                k.append(float(j[0]))
                k.append(j[1])
                k.append(j[2])
                k.append(j[3])
                for i in range(4, len(j) - 1):
                    k.append(float(j[i]))
            else:
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
    for ds in ["stream_densired", "rbf3", "har", "electricity", "insects_abrupt"]:
        data, label = load_data(ds)

        match_prior = 1
        for i in range(len(label)-1):
            if label[i] == label[i+1]:
                match_prior += 1

        nochange_acc = match_prior / len(label)

        tss = []
        for kn in [200, 400, 600, 800, 1000]:
            for s in [20, 50, 100, 200, 500]:
                tss.append(tempsil(t=np.arange(len(data)), x=data, l=label, c=1, kn=kn, s=s)[2])

        ts = np.mean(tss)

        print(f"{ds} & {len(label)} & {data.shape[1]} & {len(np.unique(label))} & {nochange_acc:.3f} & {ts:.3f}")


import os

#from capymoa.datasets import Electricity, RBFm_100k, Hyper100k # commented out as no Java on server
import numpy as np
import sklearn
from numpy.random import PCG64
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ucimlrepo import fetch_ucirepo

from utils import get_noise_labels


def read_file(dsname):
    dic = {}
    classmember = 0
    try:
        data = np.load(f"./data/{dsname}.npy")
        x = data[:, :-1]
        label = data[:, -1]
        return x, label
    except FileNotFoundError:
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
        elif dsname == "covertype":
            data, label = sklearn.datasets.fetch_covtype(return_X_y =True, download_if_missing=True, data_home="./data/sklearn_download/")
            return data, label
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
        elif dsname == "har": # UCI: DOI:10.24432/C54S4K, Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
            if not os.path.exists("./data/adult/HAR_data.npy"):
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

        elif dsname == "electricity":
            if not os.path.exists("./data/capymoa_download/"):
                os.mkdir("./data/capymoa_download/")
            #if not os.path.exists("./data/capymoa_download/electricity.arff"):
                #Electricity(directory="./data/capymoa_download/")
            file = open("./data/capymoa_download/electricity.arff", "r")
        elif dsname == "hyper":
            if not os.path.exists("./data/capymoa_download/"):
                os.mkdir("./data/capymoa_download/")
            #if not os.path.exists("./data/capymoa_download/hyper100k.arff"):
                #Hyper100k(directory="./data/capymoa_download/")
            file = open("./data/capymoa_download/hyper100k.arff", "r")
        elif dsname == "rbfm":
            if not os.path.exists("./data/capymoa_download/"):
                os.mkdir("./data/capymoa_download/")
            #if not os.path.exists("./data/capymoa_download/rbfm_100k.arff"):
                #RBFm_100k(directory="./data/capymoa_download/")
            file = open("./data/capymoa_download/rbfm_100k.arff", "r")
        else:
            try:
                file = open("./data/synthetic_milaan9" + "/" + dsname + ".arff", "r")
            except FileNotFoundError:
                try:
                    file = open("./data/real-world" + "/" + dsname + ".arff", "r")
                except FileNotFoundError:
                    try:
                        file = open("./data/ucipp-master/uci/" + dsname + ".arff", "r")
                    except:
                        try:
                            file = open("./data/artificial" + "/" + dsname + ".arff", "r")
                        except:
                            raise FileNotFoundError
    x = []
    label = []
    for line in file:
        if (line.startswith("@") or line.startswith("%") or ("class" in line and dsname not in ["hyper", "rbfm"]) or "duration" in line or len(
                line.strip()) == 0 or "fnlwgt" in line):
            pass
        else:
            j = line.split(",")
            if ("?" in j):
                continue
            k = []

            if dsname == "hyper":
                j = j[:-1]

            if dsname == "adult":
                temp = j.pop(6) # prediction target is at position 6, move to end
                j.append(temp)

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

shuffle_list = ["densired2","densired5","densired10","densired50","densired100",
                "complex9","complex8","diamond9","aggregation","D31","R15"]
def load_data(dsname, seed=0, batch=None, batch_max=None, scale=True, standard=False):
    generator = np.random.Generator(PCG64(seed))
    X, y = read_file(dsname)
    if dsname in shuffle_list:
        idx = generator.permutation(len(X))
        X, y = X[idx], y[idx]
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    if standard:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = get_noise_labels(y)

    if batch is not None and batch_max is not None:
        X = X[:batch*batch_max]
        y = y[:batch*batch_max]
    return X, y

if __name__ == "__main__":
    data, label = load_data("complex9")
    print(data[0:40])
    print(data.shape, label.shape)
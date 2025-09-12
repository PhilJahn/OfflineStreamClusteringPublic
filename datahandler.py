import numpy as np
from numpy.random import PCG64
from sklearn.preprocessing import MinMaxScaler

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
        if dsname == "covertype":
            file = open("./data/stream_data/ForestCoverType.csv", "r")
        elif dsname == "kddcup":
            file = open("./data/stream_data/KDDCUP99.csv", "r")
        elif dsname == "gassensor":
            file = open("./data/stream_data/GasSensorArray.csv", "r")
        elif dsname == "electricity":
            file = open("./data/stream_data/Electricity.csv", "r")
        elif dsname == "powersupply":
            file = open("./data/stream_data/Powersupply.csv", "r")
        elif dsname == "starlight":
            file = open("./data/stream_data/StarLightCurves.csv", "r")
        elif dsname == "airlines":
            file = open("./data/stream_data/Airlines.csv", "r")
        elif dsname == "chess":
            file = open("./data/stream_data/Chess.csv", "r")
        elif dsname == "keystroke":
            file = open("./data/stream_data/Keystroke.csv", "r")
        elif dsname == "luxembourg":
            file = open("./data/stream_data/Luxembourg.csv", "r")
        elif dsname == "noaa":
            file = open("./data/stream_data/NOAA.csv", "r")
        elif dsname == "outdoor":
            file = open("./data/stream_data/Outdoor.csv", "r")
        elif dsname == "ozone":
            file = open("./data/stream_data/Ozone.csv", "r")
        elif dsname == "poker":
            file = open("./data/stream_data/PokerHand.csv", "r")
        elif dsname == "rialto":
            file = open("./data/stream_data/Rialto.csv", "r")
        elif dsname == "sensor":
            file = open("./data/stream_data/SensorStream.csv", "r")
        elif dsname == "yoga":
            file = open("./data/stream_data/Yoga.csv", "r")
        elif dsname == "asfault":
            file = open("./data/stream_data/Asfault.csv", "r")
        elif dsname == "rbf3":
            file = open("./data/RBF3_40000.csv", "r")
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
        # print(line)
        if (line.startswith("@") or line.startswith("%") or "class" in line or "duration" in line or len(
                line.strip()) == 0):
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

    if dsname == "kddcup":
        #print(np.sort(np.char.lower(np.unique(np.array(x)[:,2]))))
        #print(np.sort(np.char.lower(np.unique(np.array(x)[:,3]))))

        third_uniques = dict(enumerate(np.char.lower(np.sort(np.unique(np.array(x)[:, 2])))))
        #from https://stackoverflow.com/questions/1031851/how-do-i-exchange-keys-with-values-in-a-dictionary
        third_dict = dict((v, k) for k, v in third_uniques.items())
        fourth_uniques = dict(enumerate(np.char.lower(np.sort(np.unique(np.array(x)[:, 3])))))
        fourth_dict = dict((v, k) for k, v in fourth_uniques.items())

        for k in x:
            if k[1] == 'tcp':
                k[1] = 1
            elif k[1] == 'udp':
                k[1] = 2
            elif k[1] == 'icmp':
                k[1] = 3
            else:
                raise ValueError
            k[2] = third_dict[k[2].lower()]
            k[3] = fourth_dict[k[3].lower()]
    file.close()
    return np.array(x), np.array(label).reshape(1, len(label))[0]

shuffle_list = ["densired2","densired5","densired10","densired50","densired100",
                "complex9","complex8","diamond9","aggregation","D31","R15"]
def load_data(dsname, seed=0, batch=None, batch_max=None):
    generator = np.random.Generator(PCG64(seed))
    scaler = MinMaxScaler()
    X, y = read_file(dsname)
    if dsname in shuffle_list:
        idx = generator.permutation(len(X))
        X, y = X[idx], y[idx]
    X = scaler.fit_transform(X)

    y = get_noise_labels(y)

    if batch is not None and batch_max is not None:
        X = X[:batch*batch_max]
        y = y[:batch*batch_max]
    return X, y

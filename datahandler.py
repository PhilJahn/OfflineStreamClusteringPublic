import numpy as np
from numpy.random import PCG64
from sklearn.preprocessing import MinMaxScaler


def read_file(dsname):
    dic = {}
    classmember = 0
    try:
        data = np.load(f"./data/{dsname}.npy")
        x = data[:, :-1]
        label = data[:, -1]
        return x, label
    except FileNotFoundError:
        try:
            file = open("./data/artificial" + "/" + dsname + ".arff", "r")
        except FileNotFoundError:
            try:
                file = open("./data/real-world" + "/" + dsname + ".arff", "r")
            except FileNotFoundError:
                try:
                    file = open("./data/ucipp-master/uci/" + dsname + ".arff", "r")
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
            if (not j[len(j) - 1].startswith("noise")):
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
    return np.array(x), np.array(label).reshape(1, len(label))[0]


def load_data(dsname, seed=0):
    generator = np.random.Generator(PCG64(seed))
    scaler = MinMaxScaler()
    X, y = read_file(dsname)
    X = scaler.fit_transform(X)
    return X, y

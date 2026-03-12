from densired.datagen import densityDataGen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from metrics.disco import disco_score


def get_densired():

    np.random.seed(0)
    skeleton = densityDataGen(dim=2, clunum=10, core_num=500, seed=42, dens_factors=True, verbose=False)

    data = skeleton.generate_data(10000)
    X_full = np.array(data)[:,:-1]
    y_full = np.array(data)[:,-1]
    plt.figure(figsize=(30, 30))
    plt.scatter(X_full[:,0], X_full[:,1],c=y_full)
    plt.show()


    core_dict = {}
    for cluid in range(10):
        core_dict[cluid] = {}
        has_core = True
        i = 0
        while has_core:
            try:
                skeleton.init_stream(command=f"{cluid}[{i}]", default_duration=100)
                core_dict[cluid][i] = []
                for _ in range(100):
                    core_dict[cluid][i].append(skeleton.__next__().tolist())
                #core_dict[cluid][i] = np.array(core_dict[cluid][i])
                i += 1
            except Exception as e:
                #print(e)
                del core_dict[cluid][i]

                has_core = False

    print([core_dict[cluid].keys() for cluid in core_dict.keys()])
    content_all = []

    ordering = {}
    ordering[0] = [33,34,35,36,37,38,39,40,41,43,44,45,46,45,44,42,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
    ordering[1] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    ordering[2] = [127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,111,
                   110,51,50,49,48,47,46,45,44,43,42,58,57,63,62,47,91,92,93,94,95,96,97,98,99,100,101,102,103,
                   102,101,100,99,98,97,96,95,94,93,92,91,48,62,63,143,142,141,140,139,138,137,136,135,134,133,132,
                   131,130,147,148,149,150,151,152,153,154,137,136,76,75,74,73,72,71,70,79,80,81,82,83,31,32,
                   65,66,67,68,67,66,10,11,12,13,14,15,16,17,18,19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,70,26,27,28,29,30,
                   31,32,33,34,35,36,37]
    ordering[2].reverse()
    ordering[3] = list(range(129))
    ordering[4] = list(range(32))
    ordering[4].extend([30,29,28,27,26,25,24,23,22,21,20,19,18,12,11,16,15,14,13,17])
    ordering[5] = list(range(6))
    ordering[6] = [3,2,1,0,4,5,4,1,2]
    ordering[7] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,7,15,14,15,16,17,18,19,20,19,26,17,6,12,15,11,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    ordering[7].extend([33,32,35,30,29,28,27,18,17,6,5,4])
    ordering[8] = [0,1,2,3,4,5,13,6,7,9,10,11,12,13,14,15,16,17,18,19]
    ordering[9] = [11,10,3,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0]
    clu_content = {}
    for cluid in range(10):
        clu_content[cluid] = []
        j_max = len(ordering[cluid])-1
        for j in range(len(ordering[cluid])):
            index = ordering[cluid][j]
            index_minus = ordering[cluid][max(j-1,0)]
            index_plus = ordering[cluid][min(j_max, j+1)]

            for _ in range(20):
                x = np.random.random()
                if x < 0.2:
                    clu_content[cluid].append(core_dict[cluid][index_minus].pop())
                elif x > 0.8:
                    clu_content[cluid].append(core_dict[cluid][index_plus].pop())
                else:
                    clu_content[cluid].append(core_dict[cluid][index].pop())
    length = {}
    for cluid in range(10):
        length[cluid] = len(clu_content[cluid])

    offset = {0: [3,0], 1: [-10, -10], 2: [15,7], 3:[-15,5], 5: [0,-10], 6: [5, 0], 7: [15,-5], 8: [15,15],  9:[5,0]}
    for cluid in range(10):
        if cluid in offset:
            for i in range(len(clu_content[cluid])):
                clu_content[cluid][i][0] += offset[cluid][0]
                clu_content[cluid][i][1] += offset[cluid][1]

    cur_rel = [2,5,7]

    for i in range(10000):
        if i == 5000:
            cur_rel.append(0)
        elif i == 2000:
            cur_rel.append(8)

        elif i == 500:
            cur_rel.append(3)
        elif i == 4000:
            cur_rel.append(9)
        elif i == 7000:
            cur_rel.append(1)
        elif i == 3000:
            cur_rel.append(4)
        elif i == 8000:
            cur_rel.append(6)
        cluid = np.random.choice(cur_rel,1,replace=True)[0]
        #print(cluid)
        content_all.append(clu_content[cluid].pop())
        length[cluid] -=1
        if length[cluid] == 0:
            assert len(clu_content[cluid]) == 0
            cur_rel.remove(cluid)

    X_all = np.array(content_all)[:,:-1]
    y_all = np.array(content_all)[:,-1].astype(int)
    print(X_all.shape, y_all.shape)

    colors = np.array(["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
              "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"])

    # plt.figure(figsize=(50, 50))
    # #plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all)
    # means = []
    # for cluid in range(10):
    #     for key in core_dict[cluid].keys():
    #         means.append(np.mean(core_dict[cluid][key], axis=0))
    #         plt.text(np.mean(core_dict[cluid][key], axis=0)[0], np.mean(core_dict[cluid][key], axis=0)[1], s=f"{cluid}_{key}")
    # means = np.array(means)
    # plt.plot(means[:,0], means[:,1])
    # plt.show()

    for batch_start in range(0, 10000, 50):
        batch_end = batch_start + 100
        plt.figure(figsize=(10, 10))
        plt.scatter(X_all[batch_start:batch_end, 0], X_all[batch_start:batch_end, 1], c=colors[y_all[batch_start:batch_end]])
        plt.title(f"{y_all[batch_start]} {batch_start} - {batch_end}")
        plt.savefig(f"densired_data/{batch_start}_{batch_end}.png")
        #plt.show()
        plt.close()

    for batch_start in range(0, 10000, 500):
        batch_end = batch_start + 1000
        plt.figure(figsize=(10, 10))
        plt.scatter(X_all[batch_start:batch_end, 0], X_all[batch_start:batch_end, 1], c=colors[y_all[batch_start:batch_end]])
        plt.title(f"{y_all[batch_start]} {batch_start} - {batch_end}")
        plt.savefig(f"densired_data/batch_{batch_start}_{batch_end}.png")
        #plt.show()
        plt.close()

    for batch_start in range(0, 10000, 1000):
        batch_end = batch_start + 2000
        plt.figure(figsize=(10, 10))
        plt.scatter(X_all[batch_start:batch_end, 0], X_all[batch_start:batch_end, 1], c=colors[y_all[batch_start:batch_end]])
        plt.title(f"{y_all[batch_start]} {batch_start} - {batch_end}")
        plt.savefig(f"densired_data/two_batch_{batch_start}_{batch_end}.png")
        #plt.show()
        plt.close()

    plt.figure(figsize=(10, 10))
    plt.scatter(X_all[:,0], X_all[:,1], c=colors[y_all])
    plt.savefig(f"densired_data/all.png")
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_all[:,0], X_all[:,1], range(len(y_all)), c=colors[y_all])
    plt.show()

    print(X_all.shape, y_all.shape)

    for stride in [100, 500, 1000, 5000, 10000]:
        for start in range(0, 10000, stride):
            end = start + stride
            X_stride = X_all[start:end]
            y_stride = y_all[start:end]
            print(stride, start, disco_score(X_stride, y_stride))

    np.save("data/stream_densired_data.npy", X_all)
    np.save("data/stream_densired_label.npy", y_all)


if __name__ == "__main__":
    get_densired()
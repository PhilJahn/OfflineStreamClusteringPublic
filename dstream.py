import copy
import math

from datahandler import load_data
from online.stream_clustering import StreamClusterer

import numpy as np


class DStream(StreamClusterer):

    def __init__(self, dim_grid_num, dim, value_ranges=1, cm=3.0, cl=0.8, window=1, l=0.998, beta=0.3):
        super().__init__()
        self.dim_grid_num = dim_grid_num
        self.grid_num = min(10**12, self.dim_grid_num ** dim) # no point in going too large here
        self.window = window
        self.d_ctr = 0
        self.cm = cm
        self.cl = cl
        if not 0 < cl < 1:
            raise ValueError("cl needs to be between 0 and 1 (exclusive)")
        if not cm > 1:
            raise ValueError("cm needs to be larger than 1")
        if not cm < self.grid_num:
            raise ValueError("cm needs to be smaller than grid_num")

        print(self.grid_num)

        if type(value_ranges) is not np.ndarray:
            value_ranges = np.ones(dim) * value_ranges

        self.bounds = np.ones(dim)
        for d in range(dim):
            self.bounds[d] = value_ranges[d] / dim_grid_num
        #print(self.bounds)
        #print(1 // self.bounds)

        self.l = l
        self.beta = beta
        if self.grid_num > 10**10:
            self.gap = np.inf # effectively everything with an assigned point is dense by this point, no need to check
        else:
            self.gap = math.floor(
                math.log(max(self.cl / self.cm, (self.grid_num - self.cm) / (self.grid_num - self.cl)), self.l))
        #print(self.gap)
        if self.gap == 0:
            self.gap = 1
        self.t = 0

        self.cells = {}
        self.removed_cells = {}  # last_removal for grid cell

    def learn_one(self, x, t: int = None):
        if self.window is not None:
            self.t = self.d_ctr // self.window
        elif t is not None:
            self.t = t
        self.d_ctr += 1
        event_log = ""
        x = np.array(x)

        grid_coordinates = x // self.bounds
        cell_id_list = [coord for coord in grid_coordinates]
        for i in range(len(cell_id_list)):
            if cell_id_list[i] == self.dim_grid_num: # rounding for 1
                cell_id_list[i] = self.dim_grid_num - 1
        #print(cell_id_list)
        cell_id = str(cell_id_list).replace("[", "").replace("]", "").replace(".0", "").replace(" ", "").replace("np.float64(","").replace(")", "")
        #print(cell_id)
        assert "." not in cell_id
        new_cell_id = cell_id
        if cell_id in self.cells.keys():
            self.cells[cell_id].insert(1, self.t)
            event_log += f"insert {cell_id};"
        else:
            if cell_id in self.removed_cells.keys():
                removal = self.removed_cells[cell_id]
            else:
                removal = None
            lower_bound = [self.bounds[d] * grid_coordinates[d] for d in range(len(grid_coordinates))]
            upper_bound = [self.bounds[d] * (grid_coordinates[d] + 1) for d in range(len(grid_coordinates))]
            cell = DStreamGridCell(cell_id=cell_id, density=1, timestamp=self.t, last_removal=removal,
                                   grid_num=self.grid_num, cm=self.cm, cl=self.cl, beta=self.beta, l=self.l,
                                   lower_bound=lower_bound, upper_bound=upper_bound)
            self.cells[cell_id] = cell
            event_log += f"add {cell_id};"

        if self.t % self.gap == 0:
            del_ids = []
            for cell_id, cell in self.cells.items():
                cell.check_status(self.t)
                if cell.status == "delete":
                    del_ids.append(cell_id)
            for cell_id in del_ids:
                del (self.cells[cell_id])
                self.removed_cells[cell_id] = self.t
                event_log += f"delete {cell_id};"
        return new_cell_id, event_log

    def learn(self, xs: list, timestamps: list = None):
        mc_ids = []
        event_logs = []
        for i in range(len(xs)):
            mc_id, event_log = self.learn_one(xs[i], timestamps[i])
            mc_ids.append(mc_id)
            event_logs.append(event_log)
        return mc_ids, event_logs

    def get_summary(self):
        cells = copy.deepcopy(self.cells)
        for cell_id, cell in cells.items():
            cell.decay(self.t)
            cell.check_status(self.t)
        return cells


class DStreamGridCell():

    def __init__(self, cell_id, density, timestamp, last_removal, grid_num, cm, cl, beta, l, lower_bound, upper_bound):
        self.cell_id = cell_id
        self.density = density
        self.last_update = timestamp  # only updated during insert
        self.last_check = timestamp
        self.last_removal = last_removal

        #hyperparameter
        self.grid_num = grid_num
        self.cm = cm
        self.cl = cl
        self.l = l
        self.beta = beta
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.status = None
        self.check_status(timestamp)

    def decay(self, timestamp_new):
        if timestamp_new > self.last_update:
            weight = self.l ** (timestamp_new - self.last_update)
            self.density *= weight
            self.last_update = timestamp_new

    def insert(self, density2, timestamp2):
        if timestamp2 > self.last_update:
            weight = self.l ** (timestamp2 - self.last_update)
            self.density *= weight
            self.last_update = timestamp2
        self.density += density2

    def check_status(self, timestamp):
        if self.last_removal is not None:
            s2 = timestamp >= (1 + self.beta) * self.last_removal
        else:
            s2 = True

        if self.grid_num > 10**10: # otherwise risk OverflowError
            pi = (self.cl * (1 - self.l ** (timestamp - self.last_update + 1))) / (self.grid_num * (1 - self.l))
        else:
            pi = 0
        weight = self.l ** (timestamp - self.last_update)
        dgt = self.density * weight
        s1 = dgt < pi
        #print(self.density, dgt, pi, s1, s2)
        if s1 and s2:
            if self.status == "sporadic":
                if self.last_update <= self.last_check:
                    self.status = "delete"
            elif self.status != "delete":
                self.status = "sporadic"
        else:
            self.status = "normal"
        self.last_check = timestamp

    def get_dict(self):
        dic = {}
        dic["cell_id"] = self.cell_id
        dic["density"] = self.density
        dic["last_update"] = self.last_update
        dic["last_check"] = self.last_check
        dic["last_removal"] = self.last_removal
        dic["grid_num"] = self.grid_num
        dic["cm"] = self.cm
        dic["cl"] = self.cl
        dic["beta"] = self.beta
        dic["l"] = self.l
        dic["status"] = self.status
        return dic

    def __str__(self):
        return str(self.get_dict())


import matplotlib.pyplot as plt
import matplotlib.patches as ptc

if __name__ == "__main__":
    make_fig = True
    stream, labels = load_data("insects_incremental")  #[[0],[1],[3],[7],[8],[9]]
    ts = list(range(len(stream)))
    dstream = DStream(dim_grid_num=30, dim=len(stream[0]))
    for i in range(len(stream)):
        cell_id, event_log = dstream.learn_one(stream[i], ts[i])
        batch_start = max(0, i + 1 - 500)
        print(i, event_log)

        #grid_cells = dstream.get_summary()
        cell_side_length = dstream.bounds
        #print(cell_side_length)

        # insert_id = None
        # if "insert" in event_log:
        #     event_log_split = event_log.split(";")
        #     for event in event_log_split:
        #         if "insert" in event:
        #             insert_id = event.split(" ")[1]
        #print(insert_id)
        #plt.figure(figsize=(8, 8))

        # for cell_id, cell in grid_cells.items():
        #     cell_pos = cell_id.split(",")
        #     cell_xy = [int(cell_pos[0]) * cell_side_length[0], int(cell_pos[1]) * cell_side_length[1]]
        #     cell_mid = [(int(cell_pos[0]) + 0.5) * cell_side_length[0], (int(cell_pos[1]) + 0.5) * cell_side_length[1]]
        #     #print(int(cell_pos[0]), cell_side_length, cell_xy)
        #     if cell.status == "sporadic":
        #         cell_patch = ptc.Rectangle(xy=cell_xy, height=cell_side_length[0], width=cell_side_length[1], alpha=0.4,
        #                                    color="red")
        #     else:
        #         weight = cell.l ** (i - cell.last_update)
        #         cell_density = cell.density * weight
        #         if cell_density >= dstream.cm / (dstream.grid_num * (1 - dstream.l)):
        #             cell_patch = ptc.Rectangle(xy=cell_xy, height=cell_side_length[0], width=cell_side_length[1],
        #                                        alpha=0.4, color="blue")
        #         elif cell_density >= dstream.cl / (dstream.grid_num * (1 - dstream.l)):  # transitional
        #             cell_patch = ptc.Rectangle(xy=cell_xy, height=cell_side_length[0], width=cell_side_length[1],
        #                                        alpha=0.4, color="lightblue")
        #         else:
        #             cell_patch = ptc.Rectangle(xy=cell_xy, height=cell_side_length[0], width=cell_side_length[1],
        #                                        alpha=0.4, color="orange")
        #     #print(cell_patch)
        #     plt.gca().add_patch(cell_patch)
        #     plt.scatter(x=cell_mid[0], y=cell_mid[1], c="blue", alpha=1, s=100)
        #     if insert_id is not None:
        #         if insert_id == cell_id:
        #             plt.scatter(x=cell_mid[0], y=cell_mid[1], c="pink", alpha=1, marker="x", s=150)
        #
        # plt.scatter(x=np.array(stream)[batch_start:i + 1][:, 0], y=np.array(stream)[batch_start:i + 1][:, 1],
        #             cmap=plt.get_cmap("jet"), c=labels[batch_start:i + 1], s=10)
        # plt.scatter(x=np.array(stream)[i:i + 1][:, 0], y=np.array(stream)[i:i + 1][:, 1], c="pink", s=25)
        # plt.ylim((0, 1))
        # plt.xlim((0, 1))
        # #plt.show()
        # plt.savefig(f"test_figures/dstream/complex9_dstream_{i}.png", bbox_inches='tight')
        # plt.close()

    grid_cells = dstream.get_summary()
    print(len(grid_cells))
    for cell_id, cell in grid_cells.items():
        print(cell)

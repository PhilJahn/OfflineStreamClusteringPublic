import os

import matplotlib.pyplot as plt
import numpy as np

from datahandler import load_data
from defines import online_colors
from subset_methods import WindowSummarizer, BatchSummarizer

if __name__ == "__main__":
    X, y = load_data("stream_densired")

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 30})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


    for batch_sampler in ["batch_random", "batch_lwc_square", "batch_kcentroid", "batch_protras"]:
        batch_sampling = BatchSummarizer(batch_sampler, 30)
        batch_sampling.learn(X[9000:10000],list(range(2000)))
        batch_summary = batch_sampling.get_summary()
        batch_buffer = X[9000:10000]
        positions = batch_summary["centers"]

        plt.figure(figsize=(6,6))
        plt.gca().set_aspect('equal')
        plt.scatter(batch_buffer[:, 0], batch_buffer[:, 1], s=4, c="black", alpha=0.25)
        plt.scatter(positions[:, 0], positions[:, 1], s=75, c=online_colors[batch_sampler], marker="x", lw=4)
        plt.savefig(f"sampling_figure/{batch_sampler}_sampling.pdf", bbox_inches="tight")
        plt.show()





    batch_1, labels_1 = X[6000:7000], y[6000:7000]
    batch_2, labels_2 = X[7000:8000], y[7000:8000]

    if not os.path.exists("sampling_figure/"):
        os.makedirs("sampling_figure/", exist_ok=True)

    plt.figure(figsize=(5,5))
    plt.scatter(batch_1[:,0], batch_1[:,1], c=labels_1, s=1.5)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_batch_1.svg", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(batch_2[:, 0], batch_2[:, 1], c=labels_2, s=1.5)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_batch_2.svg", bbox_inches="tight")
    plt.show()

    window_sampling = WindowSummarizer("window_kcentroid", 10, 1000, 1000, 0.75)
    window_sampling.learn(X[6000:8000],list(range(2000)))

    window_buffer = np.array(window_sampling.data)
    print(len(window_buffer))

    plt.figure(figsize=(5, 5))
    plt.scatter(window_buffer[:, 0], window_buffer[:, 1], s=1.5, c="black")
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_window_buffer.svg", bbox_inches="tight")
    plt.show()

    window_summary = window_sampling.get_summary()
    positions = window_summary["centers"]

    window_color = online_colors["window_kcentroid"]
    plt.figure(figsize=(5, 5))
    plt.scatter(window_buffer[:, 0], window_buffer[:, 1], s=1.5, c="black", alpha=0.5)
    plt.scatter(positions[:, 0], positions[:, 1], s=100, c=window_color, marker="x", lw=5)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_window_centers.svg", bbox_inches="tight")
    plt.show()

    weights = window_summary["weights"]
    plt.figure(figsize=(5, 5))
    plt.scatter(window_buffer[:, 0], window_buffer[:, 1], s=1.5, c="black", alpha=0.5)
    plt.scatter(positions[:, 0], positions[:, 1], c=window_color, marker="x", s=5*weights, lw=7)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_window_weights.svg", bbox_inches="tight")
    plt.show()



    batch_sampling = BatchSummarizer("batch_kcentroid", 10)
    batch_sampling.learn(X[7000:8000],list(range(2000)))

    batch_buffer = X[7000:8000]
    print(len(batch_buffer))

    plt.figure(figsize=(5, 5))
    plt.scatter(batch_buffer[:, 0], batch_buffer[:, 1], s=1.5, c="black")
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_batch_buffer.svg", bbox_inches="tight")
    plt.show()

    batch_summary = batch_sampling.get_summary()
    positions = batch_summary["centers"]

    batch_color = online_colors["batch_kcentroid"]
    plt.figure(figsize=(5, 5))
    plt.scatter(batch_buffer[:, 0], batch_buffer[:, 1], s=1.5, c="black", alpha=0.5)
    plt.scatter(positions[:, 0], positions[:, 1], s=100, c=batch_color, marker="x", lw=5)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_batch_centers.svg", bbox_inches="tight")
    plt.show()

    weights = batch_summary["weights"]
    plt.figure(figsize=(5, 5))
    plt.scatter(batch_buffer[:, 0], batch_buffer[:, 1], s=1.5, c="black", alpha=0.5)
    plt.scatter(positions[:, 0], positions[:, 1], c=batch_color, marker="x", s=5*weights, lw=7)
    plt.xlim((0.05,0.85))
    plt.ylim((0.1,0.9))
    plt.axis("off")
    plt.savefig("sampling_figure/stream_densired_batch_weights.svg", bbox_inches="tight")
    plt.show()

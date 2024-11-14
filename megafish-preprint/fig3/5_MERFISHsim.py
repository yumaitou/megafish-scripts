import os
import time
import megafish as mf

import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt


root_dir = "/spo82/ana/240525_starfish/240926/"
pitch = [0.1939, 0.1002, 0.1002]
n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
    16, 4, 4, 1, 1600, 1600


def main_1():
    # Define the combination of cycle and round for gene count series

    c = np.arange(2, 10)
    r = np.arange(2, 10)

    # Combinations of cycle and round
    cr = np.array(np.meshgrid(c, r)).T.reshape(-1, 2)

    df_cr = pd.DataFrame(cr, columns=["c", "r"])

    df_cr["code"] = df_cr["c"] * df_cr["r"]
    df_cr["gene"] = df_cr["c"] ** df_cr["r"]

    df_cr["gpc"] = df_cr["gene"] / df_cr["c"]

    print(df_cr)

    target = 100
    df_cr = df_cr[df_cr["gene"] > target]
    df_cr = df_cr.sort_values("gene")
    print(df_cr)


"""
100, 300, 1000, 3000, 10000, 30000
c  r  code       gene         gpc
5  3    15        125        25.0
7  3    21        343        49.0
4  5    20       1024       256.0
5  5    25       3125       625.0
5  6    30      15625      3125.0
8  5    40      32768      4096.0

"""


def main_2():
    # Make codebook

    crs = ((5, 3), (7, 3), (4, 5), (5, 5), (5, 6), (8, 5))

    for cr in crs:
        c, r = cr
        n_gene = c ** r
        n_code = c * r

        code_base = np.eye(c)
        pos = range(c)
        pos_comb = list(itertools.product(pos, repeat=r))

        codebook = np.zeros((n_gene, n_code))
        for j, pos in enumerate(pos_comb):
            code = np.zeros((r, c))
            for i, p in enumerate(pos):
                code[i] = code_base[p]
            codebook[j] = code.flatten()

        l2 = np.linalg.norm(codebook, axis=1)
        codebook = codebook / l2[:, None]

        codebook_name = "codebook_" + str(n_gene) + ".npy"
        codebook_path = os.path.join(root_dir, codebook_name)
        np.save(codebook_path, codebook)


def main_3():
    # Speed test

    zarr_path = os.path.join(root_dir, "merfish.zarr")

    df_speed_path = os.path.join(root_dir, "df_speed.csv")

    def add_to_df_speed(df_speed_path, n_gene, use_gpu, scheduler, time):
        df_speed = pd.read_csv(df_speed_path)
        df_add = pd.DataFrame(
            {"n_gene": [n_gene], "use_gpu": [use_gpu],
                "scheduler": [scheduler], "time": [time]})
        df_speed = pd.concat([df_speed, df_add])
        df_speed.to_csv(df_speed_path, index=False)

    use_gpus = [True, False]
    schedulers = ["processes", "threads", "synchronous"]

    crs = ((5, 3), (7, 3), (4, 5), (5, 5), (5, 6), (8, 5))

    for use_gpu in use_gpus:
        for scheduler in schedulers:
            for cr in crs:
                n_gene = cr[0] ** cr[1]
                n_code = cr[0] * cr[1]
                n_cycle, n_y, n_x = n_code, 4000, 4000

                dims = ("cycle", "y", "x")
                coords = {"cycle": np.arange(n_cycle),
                          "y": np.arange(n_y), "x": np.arange(n_x), }
                chunks = (1, 1000, 1000)

                rand_data = da.random.random(
                    size=(n_cycle, n_y, n_x), chunks=chunks)
                xar = xr.DataArray(rand_data, dims=dims, coords=coords)
                ds = xar.to_dataset(name="data")
                ds.to_zarr(zarr_path, group="raw/0", mode="w")

                code_path = os.path.join(root_dir, "codebook",
                                         "codebook_" + str(n_gene) + ".npy")

                mf.config.use_gpu(use_gpu)
                mf.config.set_scheduler(scheduler)
                st = time.time()
                mf.decode.nearest_neighbor(zarr_path, "raw", code_path)
                add_to_df_speed(df_speed_path, n_gene, use_gpu,
                                scheduler, time.time() - st)


def main_4():
    # Plot speed

    df_speed_path = os.path.join(root_dir, "df_speed.csv")
    df = pd.read_csv(df_speed_path)

    df_gpu = df[df["use_gpu"].eq(True) & df["scheduler"].eq("threads")]

    print(df_gpu)

    df_sync = df[df["use_gpu"].eq(False) & df["scheduler"].eq("synchronous")]
    df_threads = df[df["use_gpu"].eq(False) & df["scheduler"].eq("threads")]
    df_processes = df[df["use_gpu"].eq(
        False) & df["scheduler"].eq("processes")]

    print(df_sync)
    print(df_threads)
    print(df_processes)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))

    x = df_gpu["n_gene"].values

    ax.plot(x, df_sync["time"].values, label="CPU (synchronous)", marker="o",
            markersize=2)
    ax.plot(x, df_threads["time"].values, label="CPU (threads)", marker="o",
            markersize=2)
    ax.plot(x, df_processes["time"].values, label="CPU (processes)", marker="o",
            markersize=2)
    ax.plot(x, df_gpu["time"].values, label="GPU", marker="o",
            markersize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e0, 1e5)
    ax.set_xlabel("Codebook size (genes)")
    ax.set_ylabel("Calculation time (s)")

    ax.legend(frameon=False, loc="upper left")

    # # save as png
    save_path = root_dir + "megafish_CPGA_codebook.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    # Uncomment only the process you want to execute

    main_1()
    # main_2()
    # main_3()
    # main_4()

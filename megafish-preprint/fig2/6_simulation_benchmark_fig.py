
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import xarray as xr

# From https://github.com/EliasNehme/DeepSTORM3D/blob/030deac79d348fc36d5549ca1fa1f89ffe70bc15/DeepSTORM3D/loss_utils.py#L95
#
# calculate jaccard and RMSE given two arrays of xyz's and the radius for matching
# matching is done based on the hungarian algorithm, where all coords. are given in microns


def calc_jaccard_rmse(xyz_gt, xyz_rec, radius):

    # if the net didn't detect anything return None's
    if xyz_rec is None:
        print("Empty Prediction!")
        return 0.0, None, None, None

    else:

        # calculate the distance matrix for each GT to each prediction
        C = pairwise_distances(xyz_rec, xyz_gt, 'euclidean')

        # number of recovered points and GT sources
        num_rec = xyz_rec.shape[0]
        num_gt = xyz_gt.shape[0]

        # find the matching using the Hungarian algorithm
        rec_ind, gt_ind = linear_sum_assignment(C)

        # number of matched points
        num_matches = len(rec_ind)

        # run over matched points and filter points radius away from GT
        indicatorTP = [False] * num_matches
        for i in range(num_matches):

            # if the point is closer than radius then TP else it's FP
            if C[rec_ind[i], gt_ind[i]] < radius:
                indicatorTP[i] = True

        # resulting TP count
        TP = sum(indicatorTP)

        # resulting jaccard index
        jaccard_index = TP / (num_rec + num_gt - TP)

        # if there's TP
        if TP:

            # pairs of TP
            rec_ind_TP = (rec_ind[indicatorTP]).tolist()
            gt_ind_TP = (gt_ind[indicatorTP]).tolist()
            xyz_rec_TP = xyz_rec[rec_ind_TP, :]
            xyz_gt_TP = xyz_gt[gt_ind_TP, :]

            # calculate mean RMSE in xy, z, and xyz
            RMSE_xy = np.sqrt(
                np.mean(np.sum((xyz_rec_TP[:, :2] - xyz_gt_TP[:, :2])**2, 1)))
            RMSE_z = np.sqrt(
                np.mean(np.sum((xyz_rec_TP[:, 2:] - xyz_gt_TP[:, 2:])**2, 1)))
            RMSE_xyz = np.sqrt(np.mean(np.sum((xyz_rec_TP - xyz_gt_TP)**2, 1)))

            return jaccard_index, RMSE_xy, RMSE_z, RMSE_xyz
        else:
            return jaccard_index, None, None, None


def natural_sort(list_to_sort):
    def _natural_keys(text):
        def _atoi(text):
            return int(text) if text.isdigit() else text
        return [_atoi(c) for c in re.split(r"(\d+)", text)]
    return sorted(list_to_sort, key=_natural_keys)


def main_1():
    # Extract coordinates from local max values

    root_dir = "/spo82/ana/240521_simu/240807/Simulated_spot_csv/spot_mip_reg_dog_lmx_ith_spt/0/"

    files = os.listdir(root_dir)
    files = natural_sort(files)
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(root_dir + file)
        df["id"] = i // 3
        df["rep"] = i % 3
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df[["id", "rep", "x", "y", "z"]]
    df = df.sort_values(["id", "rep", "x", "y", "z"])
    df = df.reset_index(drop=True)

    csv_path = "/spo82/ana/240521_simu/240807/Simulated_spot_csv/spot_zyx_megafish.csv"
    df.to_csv(csv_path, index=False)


def main_2():
    # Extract coordinates from ground truth

    root_dir = "/spo82/ana/240521_simu/240807/coordinate/"

    files = os.listdir(root_dir)
    files = natural_sort(files)
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(root_dir + file)
        df["id"] = i // 3
        df["rep"] = i % 3
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df["z"] = 0
    df = df[["id", "rep", "x", "y", "z"]]
    df = df.sort_values(["id", "rep", "x", "y", "z"])
    df = df.reset_index(drop=True)

    csv_path = "/spo82/ana/240521_simu/240807/coordinate/spot_zyx_gt.csv"
    df.to_csv(csv_path, index=False)


def main_3():
    # Extract coordinates from bigfish

    root_dir = "/spo82/ana/240521_simu/240807/bigfish/pixel/"
    root_dir = "/spo82/ana/240521_simu/240807/bigfish/subpix/"

    files = os.listdir(root_dir)
    files = natural_sort(files)
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(root_dir + file)
        df["id"] = i // 3
        df["rep"] = i % 3
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df["z"] = 0
    df = df[["id", "rep", "x", "y", "z"]]
    df = df.sort_values(["id", "rep", "x", "y", "z"])
    df = df.reset_index(drop=True)

    csv_path = "/spo82/ana/240521_simu/240807/bigfish/spot_zyx_bigfish_pix.csv"
    csv_path = "/spo82/ana/240521_simu/240807/bigfish/spot_zyx_bigfish_subpix.csv"
    df.to_csv(csv_path, index=False)


def main_4():
    # Extract coordinates from rsfish

    root_dir = "/spo82/ana/240521_simu/240807/rsfish/result/"

    files = os.listdir(root_dir)
    files = natural_sort(files)
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(root_dir + file)
        df["id"] = i // 3
        df["rep"] = i % 3
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df["z"] = 0
    df = df[["id", "rep", "x", "y", "z"]]
    df = df.sort_values(["id", "rep", "x", "y", "z"])
    df = df.reset_index(drop=True)

    csv_path = "/spo82/ana/240521_simu/240807/rsfish/result/spot_zyx_rsfish.csv"
    df.to_csv(csv_path, index=False)


def main_5():
    # Calculate jaccard and RMSE

    root_dir = "/spo82/ana/240521_simu/240807/spots/"

    gt_path = root_dir + "spot_zyx_gt.csv"
    gt = pd.read_csv(gt_path)
    xyz_gt = gt[["x", "y", "z"]].values

    # megafish_path = root_dir + "spot_zyx_megafish.csv"
    # megafish = pd.read_csv(megafish_path)
    # xyz_rec = megafish[["x", "y", "z"]].values

    # "spot_zyx_bigfish_pix.csv", "spot_zyx_bigfish_subpix.csv", "spot_zyx_rsfish.csv"]:
    for file_name in ["spot_zyx_megafish.csv"]:
        fish_path = root_dir + file_name
        fish = pd.read_csv(fish_path)

        density = [0.01, 0.03, 0.1, 0.3, 1, 3]
        ja_rmse = np.zeros((6, 3, 2))
        for cnt_id in tqdm(range(6)):
            for fov in tqdm(range(3), leave=False):

                gt_sel = gt[(gt["id"] == cnt_id) & (gt["rep"] == fov)]
                xyz_gt = gt_sel[["x", "y", "z"]].values
                xyz_gt = xyz_gt  # - 0.5

                fish_sel = fish[(fish["id"] == cnt_id) & (fish["rep"] == fov)]
                xyz_rec = fish_sel[["x", "y", "z"]].values

                # calculate jaccard and RMSE
                jaccard, RMSE_xy, RMSE_z, RMSE_xyz = calc_jaccard_rmse(
                    xyz_gt, xyz_rec, 1)

                ja_rmse[cnt_id, fov, 0] = jaccard
                ja_rmse[cnt_id, fov, 1] = RMSE_xy

        # average for each count
        ja_rmse_avg = ja_rmse.mean(axis=1)
        ja_rmse_sd = ja_rmse.std(axis=1)
        # print(ja_rmse_avg)
        # print(ja_rmse_sd)

        ja_ave = ja_rmse_avg[:, 0]
        ja_sd = ja_rmse_sd[:, 0]

        rmse_ave = ja_rmse_avg[:, 1]
        rmse_sd = ja_rmse_sd[:, 1]

        # concat to one array
        ja_rmse = np.stack([density, ja_ave, ja_sd, rmse_ave, rmse_sd], axis=1)

        df = pd.DataFrame(ja_rmse, columns=[
            "density", "jaccard", "jaccard_sd", "rmse", "rmse_sd"])

        # save as csv
        save_path = root_dir + "jaccard_rmse_" + file_name
        df.to_csv(save_path, index=False)


def main_6():
    # make graph of jaccard and RMSE
    root_dir = "/spo82/ana/240521_simu/240807/spots/"

    groups = ["MEGA-FISH", "Big-FISH pixel", "Big-FISH subpixel", "RS-FISH"]
    colors = ["m", "g", "b", "k"]
    markers = [".-", ".-", ".-", ".-"]

    file_names = ["spot_zyx_megafish.csv", "spot_zyx_bigfish_pix.csv",
                  "spot_zyx_bigfish_subpix.csv", "spot_zyx_rsfish.csv"]

    dfs = []
    for group, file_name in zip(groups, file_names):
        save_path = root_dir + "/_jaccard_rmse_" + file_name
        df = pd.read_csv(save_path)
        df["group"] = group
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    print(df)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))
    for group, color, marker in zip(groups, colors, markers):

        df_sel = df[df["group"] == group]
        # make jaccard graph
        ax.errorbar(df_sel["density"], df_sel["jaccard"], yerr=df_sel["jaccard_sd"],
                    fmt=marker, color=color, label=group,
                    markerfacecolor="none", capsize=2)

    ax.set_xscale("log")
    ax.set_xlabel("Lateral spot density (spots " + r"$\rm \mu m^{-2}$)")
    ax.set_ylabel("Jaccard index")
    ax.legend(frameon=False)

    # save as png
    save_path = root_dir + "/jaccard.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))
    for group, color, marker in zip(groups, colors, markers):

        df_sel = df[df["group"] == group]

        ax.errorbar(df_sel["density"], df_sel["rmse"] * 100, yerr=df_sel["rmse_sd"] * 100,
                    fmt=marker, color=color, label=group,
                    markerfacecolor="none", capsize=2)

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Lateral spot density (spots " + r"$\rm \mu m^{-2}$)")
    ax.set_ylabel("RMSE (" + r"$\rm nm$)")
    ax.legend(frameon=False)

    # save as png
    save_path = root_dir + "/rmse.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_7():
    import matplotlib.pyplot as plt

    x = np.array([1, 2, 3, 4])
    labels = ["MEGA-FISH", "Big-FISH pixel", "Big-FISH subpixel", "RS-FISH"]

    pitch = [0, 0.1, 0.1]
    n_gene = 18
    n_channel = 1
    stitched_shape = [1000, 1000]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    cpus = np.array([3.8845205307006836,
                     13.64416241645813,
                     1440.3865644931793,
                     83.354])

    cpus = cpus / (area * genes)

    data = [cpus]

    margin = 0.3  # 0 <margin< 1
    totoal_width = 1 - margin

    root_dir = "/spo82/ana/240521_simu/240807/spots/"
    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data), color="gray")

    # Set the x ticks with names
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=-50)
    ax.set_ylim(10, 10000)
    # ax.legend(["CPU single", "CPU multi", "GPU"])
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")
    ax.set_yscale("log")

    # save as png
    save_path = root_dir + "Simu_CPGA.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == '__main__':
    # Uncomment only the process you want to execute

    # main_1()
    # main_2()
    # main_3()
    # main_4()
    # main_5()
    # main_6()
    main_7()

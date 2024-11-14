import os
import numpy as np
import pandas as pd
import tifffile

import matplotlib.pyplot as plt


def main_1():
    # Count the number of bigfish
    for rna in ["rna1", "rna2", "rna3"]:
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/bigfish/" + rna + "/"
        counts = []
        for i in range(74):
            path = "FishQuant_results_" + \
                str(i) + "_7_7.tif_noSubLocalization.csv"
            path = os.path.join(data_dir, path)
            df = pd.read_csv(path)
            count = len(df)
            counts.append(count)

        counts = np.array(counts)

        df_count = pd.DataFrame(counts, columns=["count"])
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/bigfish/"
        df_count.to_csv(os.path.join(
            data_dir, "bigfish_" + rna + "_counts.csv"), index=False)


def main_2():
    # convert to one csv
    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/bigfish/"

    for rna in ["rna1", "rna2", "rna3"]:
        file_name = "bigfish_" + rna + "_counts.csv"
        path = os.path.join(data_dir, file_name)
        df = pd.read_csv(path)
        df = df.rename(columns={"count": rna})
        if rna == "rna1":
            df_all = df
        else:
            df_all = pd.concat([df_all, df[rna]], axis=1)

    df_all.to_csv(os.path.join(data_dir, "bigfish_counts.csv"), index=False)


def main_3():
    # Count the number of rna in rsfish

    for rna in ["rna1", "rna2", "rna3"]:
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/rsfish/" + rna + "/"

        counts = []
        for i in range(74):
            path = str(i) + "_7_7.tif.csv"
            path = os.path.join(data_dir, path)
            df = pd.read_csv(path)
            count = len(df)
            counts.append(count)

        counts = np.array(counts)

        df_count = pd.DataFrame(counts, columns=["count"])
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/rsfish/"
        df_count.to_csv(os.path.join(
            data_dir, "rsfish_" + rna + "_counts.csv"), index=False)


def main_4():
    # convert to one csv
    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/rsfish/"

    for rna in ["rna1", "rna2", "rna3"]:
        file_name = "rsfish_" + rna + "_counts.csv"
        path = os.path.join(data_dir, file_name)
        df = pd.read_csv(path)
        df = df.rename(columns={"count": rna})
        if rna == "rna1":
            df_all = df
        else:
            df_all = pd.concat([df_all, df[rna]], axis=1)

    df_all.to_csv(os.path.join(data_dir, "rsfish_counts.csv"), index=False)


def main_5():
    # count dog_lmx_ith image
    for rna in ["rna1", "rna2", "rna3"]:
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/dog_lmx_ith/" + rna + "/"
        counts = []
        for i in range(74):
            path = str(i) + "_7_7.tif"
            path = os.path.join(data_dir, path)
            img = tifffile.imread(path)
            count = np.sum(img > 2)
            counts.append(count)

        counts = np.array(counts)

        df_count = pd.DataFrame(counts, columns=["count"])
        data_dir = "/spo82/ana/012_SeqFISH_IF/240822/dog_lmx_ith/"
        df_count.to_csv(os.path.join(
            data_dir, "dog_lmx_ith_" + rna + "_counts.csv"), index=False)

    # convert to one csv
    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/dog_lmx_ith/"

    for rna in ["rna1", "rna2", "rna3"]:
        file_name = "dog_lmx_ith_" + rna + "_counts.csv"
        path = os.path.join(data_dir, file_name)
        df = pd.read_csv(path)
        df = df.rename(columns={"count": rna})
        if rna == "rna1":
            df_all = df
        else:
            df_all = pd.concat([df_all, df[rna]], axis=1)

    df_all.to_csv(os.path.join(
        data_dir, "megafish_counts.csv"), index=False)


def main_6():
    # merge all
    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/"
    df_bigfish = pd.read_csv(os.path.join(data_dir, "bigfish_counts.csv"))
    df_bigfish["cycle"] = np.arange(74)

    dfs = []
    for rna in ["rna1", "rna2", "rna3"]:
        df_col = df_bigfish[["cycle", rna]]
        df_col["rna"] = rna
        df_col = df_col.rename(columns={rna: "count"})
        dfs.append(df_col)
    df_bigfish = pd.concat(dfs)

    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/"
    df_rsfish = pd.read_csv(os.path.join(data_dir, "rsfish_counts.csv"))
    df_rsfish["cycle"] = np.arange(74)

    dfs = []
    for rna in ["rna1", "rna2", "rna3"]:
        df_col = df_rsfish[["cycle", rna]]
        df_col["rna"] = rna
        df_col = df_col.rename(columns={rna: "count"})
        dfs.append(df_col)
    df_rsfish = pd.concat(dfs)

    data_dir = "/spo82/ana/012_SeqFISH_IF/240822/"
    df_megafish = pd.read_csv(os.path.join(data_dir, "megafish_counts.csv"))
    df_megafish["cycle"] = np.arange(74)
    dfs = []
    for rna in ["rna1", "rna2", "rna3"]:
        df_col = df_megafish[["cycle", rna]]
        df_col["rna"] = rna
        df_col = df_col.rename(columns={rna: "count"})
        dfs.append(df_col)
    df_megafish = pd.concat(dfs)

    df = pd.merge(df_bigfish, df_megafish, on=["cycle", "rna"])
    df = df.rename(columns={"count_x": "bigfish", "count_y": "megafish"})

    df = pd.merge(df, df_rsfish, on=["cycle", "rna"])
    df = df.rename(columns={"count": "rsfish"})
    df.to_csv(os.path.join(data_dir, "fish_counts.csv"), index=False)


def main_7():
    # compare bigfish and megafish

    root_dir = "/spo82/ana/012_SeqFISH_IF/240822/"

    df = pd.read_csv(root_dir + "fish_counts.csv")

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
    ax.scatter(df["megafish"], df["bigfish"], c="black", s=3)

    r = np.corrcoef(df["megafish"], df["bigfish"])[0, 1]
    ax.text(0.05, 0.9, f"R = {r:.3f}", transform=ax.transAxes)

    ax.set_xlabel("MEGA-FISH (counts/image)")
    ax.set_ylabel("Big-FISH (counts/image)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xticks(np.arange(0, 1600, 300))
    # ax.set_yticks(np.arange(0, 1600, 300))
    # ax.set_xlim(0, 1500)
    # ax.set_ylim(0, 1500)

    save_path = root_dir + "mega_vs_big.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_8():
    # compare megafish and rsfish
    root_dir = "/spo82/ana/012_SeqFISH_IF/240822/"

    df = pd.read_csv(root_dir + "fish_counts.csv")

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
    ax.scatter(df["megafish"], df["rsfish"], c="black", s=3)

    r = np.corrcoef(df["megafish"], df["rsfish"])[0, 1]
    ax.text(0.05, 0.9, f"R = {r:.3f}", transform=ax.transAxes)

    ax.set_xlabel("MEGA-FISH (counts/image)")
    ax.set_ylabel("RS-FISH (counts/image)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xticks(np.arange(0, 1600, 300))
    # ax.set_yticks(np.arange(0, 1600, 300))
    # ax.set_xlim(0, 1500)
    # ax.set_ylim(0, 1500)

    save_path = root_dir + "mega_vs_rs.pdf"
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
    # main_5()
    # main_6()
    # main_7()
    # main_8()

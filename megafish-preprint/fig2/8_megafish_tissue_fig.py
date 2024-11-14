import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = "/spo82/ana/012_SeqFISH_IF/240808/"

pitch = [0.0994, 0.0994]
chunk_size_y = 1000
chunk_size_x = 1000
n_cycle = 74


def main_1():
    # Merge RNA count csv files

    groups = ["rna1", "rna2", "rna3"]
    csv_header = "012_SeqFISH_IF_1_"
    csv_footer = "_mip_reg_dog_lmx_ith_cnt.csv"

    seg_info_name = "012_SeqFISH_IF_1_nuc_cyt_rgb_lbl_olr_fil_seg.csv"

    df_seg = pd.read_csv(root_dir + seg_info_name)
    df_seg["centroid_y"] = (
        df_seg["chunk_y"] * chunk_size_y + df_seg["centroid_y"]) * pitch[0]
    df_seg["centroid_x"] = (
        df_seg["chunk_x"] * chunk_size_x + df_seg["centroid_x"]) * pitch[1]
    df_seg = df_seg[["seg_id", "area", "centroid_y", "centroid_x"]]
    df_seg = df_seg.rename(columns={"area": "area_pix2",
                                    "centroid_y": "centroid_y_um",
                                    "centroid_x": "centroid_x_um"})
    cols = []
    for group in groups:
        csv_path = root_dir + csv_header + group + csv_footer
        df = pd.read_csv(csv_path)

        df = df.groupby(["cycle", "seg_id"]).sum().reset_index()

        df_cycles = []
        for cycle in range(n_cycle):
            df_cycle = df[df["cycle"] == cycle]
            df_cycle = df_cycle[["seg_id", "count"]]
            col = group + "_cycle" + str(cycle + 1)
            df_cycle = df_cycle.rename(columns={"count": col})
            cols.append(col)
            df_cycles.append(df_cycle)

        df = df_cycles[0]
        for i in range(1, len(df_cycles)):
            df = pd.merge(df, df_cycles[i], on="seg_id", how="outer")

        if group == "rna1":
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="seg_id", how="outer")

    df_all = df_all[df_all["seg_id"] != 0]

    df_all = pd.merge(df_all, df_seg, on="seg_id", how="outer")

    df_all = df_all[["seg_id", "area_pix2",
                     "centroid_y_um", "centroid_x_um"] + cols]

    save_name = csv_header + "rna" + csv_footer
    df_all.to_csv(root_dir + save_name, index=False)


def main_2():
    # Rename columns in the RNA count csv file
    csv_header = "012_SeqFISH_IF_1_"
    csv_footer = "_mip_reg_dog_lmx_ith_cnt.csv"

    csv_name = csv_header + "rna" + csv_footer
    df_rna = pd.read_csv(root_dir + csv_name)
    df_genename = pd.read_csv(root_dir + "012_SeqFISH_IF_genename_rna.csv")

    df_genename = df_genename[["ch2", "ch3", "ch4"]]

    names = []
    for _, row in df_genename.iterrows():
        names += row.values.tolist()

    cols = [col for col in df_rna.columns if "cycle" in col]
    names_cols = {}
    empty_cols = []
    for name, col in zip(names, cols):
        if "empty" in name:
            empty_cols.append(col)
        else:
            names_cols[col] = name

    df_rna = df_rna.drop(columns=empty_cols)
    df_rna = df_rna.rename(columns=names_cols)
    df_rna = df_rna[["seg_id", "area_pix2",
                     "centroid_y_um", "centroid_x_um"] +
                    list(names_cols.values())]

    save_name = csv_header + "rna" + "_rename.csv"
    df_rna.to_csv(root_dir + save_name, index=False)


def main_3():
    # Calculate RNA count/cell correlation between replicates

    csv_1_header = "012_SeqFISH_IF_1_"
    csv_name_1 = csv_1_header + "rna_rename.csv"
    df_rna_1 = pd.read_csv(root_dir + csv_name_1)
    df_rna_1 = df_rna_1[df_rna_1["area_pix2"] > 700]

    df_rna_1_mean = df_rna_1.mean()
    df_rna_1_mean = df_rna_1_mean.drop(["seg_id", "area_pix2",
                                        "centroid_y_um", "centroid_x_um"])

    csv_2_header = "012_SeqFISH_IF_2_"
    csv_name_2 = csv_2_header + "rna_rename.csv"
    df_rna_2 = pd.read_csv(root_dir + csv_name_2)
    df_rna_2 = df_rna_2[df_rna_2["area_pix2"] > 700]
    df_rna_2_mean = df_rna_2.mean()
    df_rna_2_mean = df_rna_2_mean.drop(["seg_id", "area_pix2",
                                        "centroid_y_um", "centroid_x_um"])

    df_rna_mean = pd.concat([df_rna_1_mean, df_rna_2_mean], axis=1)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
    ax.scatter(df_rna_mean.iloc[:, 0], df_rna_mean.iloc[:, 1],
               color="black", s=3)

    r = np.corrcoef(df_rna_mean.iloc[:, 0], df_rna_mean.iloc[:, 1])[0, 1]
    ax.text(0.05, 0.9, f"R = {r:.3f}", transform=ax.transAxes)

    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 31, 10))
    ax.set_yticks(np.arange(0, 31, 10))
    ax.set_xlabel("Replicate 1 (RNA count/cell)")
    ax.set_ylabel("Replicate 2 (RNA count/cell)")

    save_path = root_dir + "012_SeqFISH_IF_rna_replicate.pdf"
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

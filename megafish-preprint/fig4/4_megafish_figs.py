import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


root_dir = "/media/ito/HDD464/Analysis/012_SeqFISH_IF/240930/"
# root_dir = "/media/ito/M24G/Analysis/012_SeqFISH_IF/240924/"
pitch = [0.1650, 0.0994, 0.0994]
n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
    74, 10, 10, 6, 2048, 2048


def main():

    csv_names = ["012_SeqFISH_IF_1_is_cpn.csv",
                 "012_SeqFISH_IF_2_is_cpn.csv",
                 "012_SeqFISH_IF_3_is_cpn.csv",
                 "012_SeqFISH_IF_4_is_cpn.csv",
                 "012_SeqFISH_IF_5_is_cpn.csv",
                 "012_SeqFISH_IF_6_is_cpn.csv",
                 "012_SeqFISH_IF_1_is_mdi.csv",
                 "012_SeqFISH_IF_2_is_mdi.csv",
                 "012_SeqFISH_IF_3_is_mdi.csv",
                 "012_SeqFISH_IF_4_is_mdi.csv",
                 "012_SeqFISH_IF_5_is_mdi.csv",
                 "012_SeqFISH_IF_6_is_mdi.csv",]

    dfs = []
    for csv_name in csv_names:
        csv_path = os.path.join(root_dir, csv_name)
        df = pd.read_csv(csv_path)
        dfs.append(df)

    lens = [len(df) for df in dfs]
    lens = np.array(lens)
    lens = lens.reshape((2, 6)).T
    print(lens)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(6.6 / 2.54, 7.6 / 2.54))

    for pl in range(len(lens)):
        ax.plot([0, 1], lens[pl], marker="o", c="k", markersize=4,
                alpha=0.7, linewidth=1)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 35000)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cellpose", "MEDIAR"])
    ax.set_ylabel("Number of detected cells")
    save_name = "240930_mf_IF_012_fig_66x76"
    save_path = os.path.join(root_dir, save_name + ".pdf")
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # main_mini()

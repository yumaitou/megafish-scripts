import os
import time
import megafish as mf  # v0.0.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


root_dir = "/spo82/ana/240525_starfish/240926/"


def main():

    df_speed_path = os.path.join(root_dir, "df_speed.csv")

    df = pd.read_csv(df_speed_path)

    print(df)
    df = df.drop_duplicates(
        subset=["chunk_size", "use_gpu", "scheduler", "process"],
        keep="first")

    df_proc = df[df["process"].isin(["1_prefilter", "4_nn"])]

    df_proc = df_proc.drop(columns=["process"])
    df_proc = df_proc.groupby(["chunk_size", "use_gpu", "scheduler"]).mean()
    df_proc = df_proc.reset_index()

    df_gpu = df_proc[df_proc["use_gpu"].eq(True)]

    df_gpu = df_gpu.drop(columns=["scheduler", "use_gpu"])

    df_gpu = df_gpu.groupby(["chunk_size"]).min()
    df_gpu = df_gpu.reset_index()

    print(df_gpu)

    df_cpu = df_proc[df_proc["use_gpu"].eq(False)]
    print(df_cpu)

    df_sync = df_cpu[df_cpu["scheduler"].eq("synchronous")]
    df_proc = df_cpu[df_cpu["scheduler"].eq("processes")]
    df_thre = df_cpu[df_cpu["scheduler"].eq("threads")]

    print(df_sync)
    print(df_proc)
    print(df_thre)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))

    x = df_gpu["chunk_size"].values

    ax.plot(x, df_sync["time"].values, label="CPU (synchronous)", marker="o",
            markersize=2)
    ax.plot(x, df_thre["time"].values, label="CPU (threads)", marker="o",
            markersize=2)
    ax.plot(x, df_proc["time"].values, label="CPU (processes)", marker="o",
            markersize=2)
    ax.plot(x, df_gpu["time"].values, label="GPU", marker="o",
            markersize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Chunk size (pixels)")
    ax.set_ylabel("Calculation time (s)")

    ax.legend(frameon=False)

    # save as png
    save_path = root_dir + "megafish_CPGA_chunk_size.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # main_mini()

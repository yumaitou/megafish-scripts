import os
import time
import megafish as mf  # v0.0.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


root_dir = "/spo83/ana/012_SeqFISH_IF/240925/"
pitch = [0.1650, 0.0994, 0.0994]
n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = 10, 4, 4, 6, 2048, 2048


def main_1():
    # Loading speed

    # M.2 Threads: 829.8255572319031 s
    # M.2 Processes: 351.74993991851807 s
    # M.2 Synchronous: 698.4801301956177
    # SSD Threads: 824.4896929264069
    # SSD Processes: 354.55245208740234
    # SSD Synchronous: 698.6569573879242
    # HDD Threads: 906.8037292957306
    # HDD Processes: 345.7970416545868
    # HDD Synchronous: 697.4877910614014

    x = np.array([0, 1, 2])
    labels = ["HDD", "SSD", "M.2"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    sync = np.array([
        697.4877910614014,
        698.6569573879242,
        698.4801301956177
    ])
    thre = np.array([
        906.8037292957306,
        824.4896929264069,
        829.8255572319031
    ])
    proc = np.array([
        345.7970416545868,
        354.55245208740234,
        351.74993991851807
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)

    data = [sync, thre, proc]

    margin = 0.2  # 0 <margin< 1
    totoal_width = 1 - margin

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 60)
    ax.legend(["Synchronous", "Multi-thread", "Multi-process"], frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_Loading.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


# CPU Synchronous: 93.61437320709229 s
# CPU Threads: 28.955502033233643 s
# CPU Processes: 40.7228422164917 s
# GPU Synchronous: 5.8643810749053955 s

# CPU Synchronous: 1143.2455370426178 s
# CPU Threads: 893.4107768535614 s
# CPU Processes: 238.64635562896729 s
# GPU Synchronous: 972.1602139472961
# GPU Processes: 237.3976378440857

# CPU Synchronous: 1377.138215303421
# CPU Threads: 453.79035782814026 s
# CPU Processes: 185.13135194778442 s
# GPU Synchronous:353.0775876045227
# GPU Threads: 732.6443862915039
# GPU Processes: 254.6411488056183


def main_2():
    # Registration speed

    x = np.array([0, 1, 2])
    labels = ["PCC registration", "SIFT registration",
              "Stitching"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    sync = np.array([
        93.61437320709229,
        1143.2455370426178,
        1377.138215303421
    ])
    thre = np.array([
        28.955502033233643,
        893.4107768535614,
        453.79035782814026
    ])
    proc = np.array([
        40.7228422164917,
        238.64635562896729,
        185.13135194778442
    ])
    gpu = np.array([
        5.8643810749053955,
        237.3976378440857,
        254.6411488056183
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)
    gpu = gpu / (area * genes)

    data = [sync, thre, proc, gpu]

    margin = 0.2  # 0 <margin< 1
    totoal_width = 1 - margin

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(["Synchronous CPU", "Multi-thread CPU",
               "Multi-process CPU", "GPU"], frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_registration.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_3():
    # PCC registration speed

    xaxis = ["PCC registration"]
    colors = ["Synchronous CPU", "Multi-thread CPU",
              "Multi-process CPU", "GPU"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    # colors group and xaxis list
    sync = np.array([
        93.61437320709229,
    ])
    thre = np.array([
        28.955502033233643,
    ])
    proc = np.array([
        40.7228422164917,
    ])
    gpu = np.array([
        5.8643810749053955,
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)
    gpu = gpu / (area * genes)

    data = [sync, thre, proc, gpu]

    margin = 0.4  # 0 <margin< 1
    totoal_width = 1 - margin
    x = np.array(range(len(xaxis)))

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(4 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, xaxis)
    ax.set_ylim(0, 5)
    ax.legend(colors, frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_registration_PCC.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_3():
    # SIFT registration speed

    xaxis = ["SIFT registration"]
    colors = ["Synchronous CPU", "Multi-thread CPU",
              "Multi-process CPU", "GPU"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    # colors group and xaxis list
    sync = np.array([
        1143.2455370426178,
    ])
    thre = np.array([
        893.4107768535614,
    ])
    proc = np.array([
        238.64635562896729,
    ])
    gpu = np.array([
        237.3976378440857,
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)
    gpu = gpu / (area * genes)

    data = [sync, thre, proc, gpu]

    margin = 0.4  # 0 <margin< 1
    totoal_width = 1 - margin
    x = np.array(range(len(xaxis)))

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(4 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, xaxis)
    ax.set_ylim(0, 60)
    # ax.legend(colors, frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_registration_SIFT.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_4():
    # Stitching speed

    xaxis = ["Stitching"]
    colors = ["Synchronous CPU", "Multi-thread CPU",
              "Multi-process CPU", "GPU"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    # colors group and xaxis list
    sync = np.array([
        377.138215303421,
    ])
    thre = np.array([
        453.79035782814026,
    ])
    proc = np.array([
        185.13135194778442,
    ])
    gpu = np.array([
        254.6411488056183,
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)
    gpu = gpu / (area * genes)

    data = [sync, thre, proc, gpu]

    margin = 0.4  # 0 <margin< 1
    totoal_width = 1 - margin
    x = np.array(range(len(xaxis)))

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(4 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, xaxis)
    ax.set_ylim(0, 25)
    # ax.legend(colors, frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_registration_stitching.pdf"
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
    plt.savefig(save_path, bbox_inches="tight")


def main_5():
    # Detection speed

    # detection
    # CPU Synchronous: 365.49443078041077
    # CPU Threads: 105.71617913246155
    # CPU Processes: 302.2726504802704
    # GPU Synchronous: 111.28626275062561
    # GPU Threads: 186.29381155967712
    # GPU Processes: 307.13768434524536

    # counting
    # CPU Synchronous: 65.33826899528503
    # CPU Threads: 49.25640106201172
    # CPU Processes:　163.69229578971863
    # GPU Synchronous: 230.3195662498474
    # GPU Threads: 642.3680515289307
    # GPU Processes: 163.48801255226135

    xaxis = ["Spot detection", "Counting"]
    colors = ["Synchronous CPU", "Multi-thread CPU",
              "Multi-process CPU", "GPU"]

    pitch = [0.1650, 0.0994, 0.0994]
    n_gene = 10
    n_channel = 3
    stitched_shape = [n_tile_y * n_y, n_tile_x * n_x]
    area = (stitched_shape[0] * pitch[1] / 1000) * \
        (stitched_shape[1] * pitch[2] / 1000)
    genes = n_channel * n_gene

    # colors group and xaxis list
    sync = np.array([
        365.49443078041077,
        65.33826899528503,
    ])
    thre = np.array([
        105.71617913246155,
        49.25640106201172
    ])
    proc = np.array([
        302.2726504802704,
        163.69229578971863
    ])
    gpu = np.array([
        111.28626275062561,
        163.48801255226135
    ])

    sync = sync / (area * genes)
    thre = thre / (area * genes)
    proc = proc / (area * genes)
    gpu = gpu / (area * genes)

    data = [sync, thre, proc, gpu]

    margin = 0.2  # 0 <margin< 1
    totoal_width = 1 - margin
    x = np.array(range(len(xaxis)))

    # make bar graph
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    for i, h in enumerate(data):
        pos = x - totoal_width * (1 - (2 * i + 1) / len(data)) / 2
        ax.bar(pos, h, width=totoal_width / len(data))

    # Set labels
    ax.set_xticks(x, xaxis)
    ax.set_ylim(0, 20)
    ax.legend(colors, frameon=False)
    ax.set_ylabel(
        "Calculation time per gene area \n (s gene$^{-1}$ mm$^{-2}$)")

    # save as png
    save_path = root_dir + "megafish_CPGA_seqfish.pdf"
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

import os
import scanpy as sc
import rapids_singlecell as rsc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

root_dir = "/spo82/ana/012_SeqFISH_IF/241007/"
pitch = [0.1650, 0.0994, 0.0994]
n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
    74, 10, 10, 6, 2048, 2048


def main_1():
    # create anndata object

    type_name = "cpi"  # mdi

    samples = {}
    offsets = {}
    for sample_no in [1, 2, 3, 4, 5, 6]:
        sample_id = type_name + str(sample_no)
        filename = f"012_SeqFISH_IF_{sample_no}_is_{type_name}.csv"
        samples[sample_id] = filename

        offsets[sample_id] = [0, 0]

    adatas = {}
    for sample_id, filename in samples.items():
        df = pd.read_csv(root_dir + filename)
        df = df.dropna()

        df["centroid_y_um"] = df["centroid_y_um"].max() - df["centroid_y_um"]
        df["centroid_y_pix"] = df["centroid_y_pix"].max() - df["centroid_y_pix"]

        offset = offsets[sample_id]
        df["centroid_y_um"] += offset[0]
        df["centroid_x_um"] += offset[1]

        counts = df.iloc[:, 7:].values
        adata = ad.AnnData(counts)
        cell_names = [f"cell_{i}" for i in df["segment_id"].values]
        adata.obs_names = cell_names
        adata.var_names = df.columns[7:].values
        adata.obs["area_um2"] = df["area_um2"].values
        adata.obs["diameter_um"] = np.sqrt(adata.obs["area_um2"] / np.pi) * 2
        adata.obs["label"] = df["segment_id"].values
        adata.var["names"] = df.columns[7:].values
        adata.obs["centroid_y_um"] = df["centroid_y_um"].values
        adata.obs["centroid_x_um"] = df["centroid_x_um"].values

        adata.obsm["X_spatial"] = df[[
            "centroid_x_um", "centroid_y_um"]].values.astype(np.float32)

        adatas[sample_id] = adata

    adata = ad.concat(adatas, label="sample")
    print(adata.obs["sample"].value_counts())
    adata.obs_names_make_unique()
    print(adata)
    adata.write(root_dir + "012_SeqFISH_IF_" + type_name + ".h5ad", "gzip")


def main_2():
    # calculate PCA
    load_name = "012_SeqFISH_IF_cpn.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)

    adata = adata[adata.obs["diameter_um"] >= 2]

    adata.layers["sel"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    sc.pp.scale(adata)
    adata.layers["scale"] = adata.X.copy()
    print("Calculating PCA")
    sc.pp.pca(adata)
    adata.layers["pca"] = adata.X.copy()

    print("Calculating neighbors")
    rsc.pp.neighbors(adata)
    adata.layers["nn"] = adata.X.copy()

    print("Calculating UMAP")
    rsc.tl.umap(adata)
    adata.layers["umap"] = adata.X.copy()

    print("Calculating Kmean")
    X_pca = adata.obsm["X_pca"]
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_pca)
    adata.obs['kmeans5'] = kmeans.labels_.astype(str)

    save_name = load_name.replace(".h5ad", "_km.h5ad")
    adata.write(root_dir + save_name, "gzip")


def main_3():
    # calculate spot density for PCA plot
    load_name = "012_SeqFISH_IF_cpn_km.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)

    X = adata.obsm["X_pca"]

    fig, ax = plt.subplots(figsize=(6, 6))

    y = X[:, 0]
    x = X[:, 1]

    from scipy.stats import gaussian_kde

    def scatter_density(x, y):
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)

    c = scatter_density(x, y)

    adata.obs["scat_dens"] = c

    save_name = load_name.replace(".h5ad", "_dens.h5ad")
    adata.write(root_dir + save_name, "gzip")


def scatter_density(x, y):
    xy = np.vstack([x, y])
    return gaussian_kde(xy)(xy)


def main_4():
    # pca plot

    load_name = "012_SeqFISH_IF_cpn_km_dens.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)

    X = adata.obsm["X_pca"]
    c = adata.obs["scat_dens"].values

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(9 / 2.54, 5 / 2.54))

    y = X[:, 0]
    x = X[:, 1]

    ax.scatter(y, x, c=c, s=1, edgecolors="none", vmin=0, vmax=0.02)

    ax.set_xlim(-16, 16)
    ax.set_ylim(-8, 8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    save_name = load_name.replace(".h5ad", "_pca")
    save_path = os.path.join(root_dir, save_name + ".png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)

    # export color bar
    fig, ax = plt.subplots(figsize=(0.2, 3 / 2.54))
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=0.02))
    sm.set_array([])
    plt.colorbar(sm, cax=ax)
    ax.set_yticks([0, 0.02])
    ax.set_yticklabels(["0", "0.02"])
    ax.set_ylabel("Density")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position("left")

    save_name = load_name.replace(".h5ad", "_pca_colorbar.pdf")
    save_path = os.path.join(root_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight", dpi=600)


def main_5():
    # representative genes

    load_name = "012_SeqFISH_IF_cpn_km.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)
    adata = adata[adata.obs["diameter_um"] >= 2]
    adata = adata[adata.obs["sample"] == "cpn2"]

    X = adata.X
    X = X.toarray()

    Pos = adata.obsm["X_spatial"]
    Pos = Pos.astype(np.float32)
    y = Pos[:, 1]
    x = Pos[:, 0]

    names = adata.var_names.to_list()

    for i in range(len(names)):
        colors = X[:, i].copy()

        min_c = np.percentile(colors, 10)
        max_c = np.percentile(colors, 90)
        colors = (colors - min_c) / (max_c - min_c)

        plt.rcParams['font.family'] = "Arial"
        plt.rcParams["font.size"] = 12
        fig, ax = plt.subplots(figsize=(2.2 / 2.54, 2.2 / 2.54))

        ax.scatter(x, y, c=colors, cmap="coolwarm", s=0.1, vmin=0, vmax=1,
                   edgecolors="none")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-00, 2000)
        ax.set_ylim(-00, 2000)
        ax.axis("off")

        file_name = names[i].replace("/", "_")
        save_name = load_name.replace(
            ".h5ad", "_spatial_" + file_name + "_cpn.png")
        save_path = os.path.join(root_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=600,
                    pad_inches=0.01)

        plt.show()

    # Save color bar
    fig, ax = plt.subplots(figsize=(0.2, 2.2))
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, cax=ax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["10", "90"])
    ax.set_ylabel("Expression (percentile)")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position("left")

    save_name = load_name.replace(".h5ad", "_colorbar.pdf")
    save_path = os.path.join(root_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight", dpi=600,
                pad_inches=0.01)


def main_6():
    # plot standard deviation of PCA
    load_name = "012_SeqFISH_IF_cpn_km.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)
    X = adata.obsm["X_pca"]

    val1s = []
    for i in range(1, 7):
        obs = "cpn" + str(i)
        adata_sel = adata[adata.obs["sample"] == obs]
        X = adata_sel.obsm["X_pca"]
        val1 = np.std(X, axis=0)
        val1s.append(val1)

    load_name = "012_SeqFISH_IF_mdi_km.h5ad"
    adata = ad.read_h5ad(root_dir + load_name)
    X = adata.obsm["X_pca"]

    val2s = []
    for i in range(1, 7):
        obs = "mdi" + str(i)
        adata_sel = adata[adata.obs["sample"] == obs]
        X = adata_sel.obsm["X_pca"]
        val2 = np.std(X, axis=0)
        val2s.append(val2)

    val1s = np.array(val1s)
    val2s = np.array(val2s)

    val1_mean = np.mean(val1s, axis=0)
    val2_mean = np.mean(val2s, axis=0)

    val1_std = np.std(val1s, axis=0)
    val2_std = np.std(val2s, axis=0)

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(3.3 * 2 / 2.54, 4.8 * 2 / 2.54))
    x = np.arange(len(val1)) + 1
    ax.errorbar(x[:5], val1_mean[:5], yerr=val1_std[:5],
                marker="o", c="k", label="Cellpose", markersize=3,
                capsize=3)
    ax.errorbar(x[:5], val2_mean[:5], yerr=val2_std[:5],
                marker="o", c="r", label="MEDIAR", markersize=3,
                capsize=3)

    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 6)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Standard deviation")
    ax.legend(frameon=False)

    save_name = "240930_mf_IF_012_fig_std"
    save_path = os.path.join(root_dir, save_name + ".pdf")
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

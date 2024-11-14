import os
import tifffile
import numpy as np
import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt
import megafish as mf


def main_1():
    root_dir = "/spo82/ana/012_SeqFISH_IF/240807/"
    pitch = [0.1650, 0.0994, 0.0994]
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
        74, 10, 10, 6, 2048, 2048

    for i in [1, 2, 3, 4, 5, 6]:
        sample_name = "012_SeqFISH_IF_" + str(i)
        zarr_path = os.path.join(root_dir, sample_name + ".zarr")
        print("===== " + sample_name + " =====")

        mf.config.set_resource(gpu=False, scheduler="processes")

        # ----- IF -----
        n_cycle = 26
        n_z = 1
        image_dir = "/spo82/img/012_SeqIF/" + str(i) + "/"
        groups = ["hcsti", "is1", "is2", "is3"]
        channels = [1, 2, 3, 4]
        scan_type = "snake_up_right"
        mf.load.make_imagepath_cYX(
            zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x,
            scan_type, image_dir, "_is")
        mf.load.ims_cYXzyx(zarr_path, n_z, n_y, n_x, "_is_imagepath")

        # ----- Stitched -----
        stitched_dir = "/spo82/img/012_SeqIF/" + str(i) + "/"
        stitched_path = os.path.join(stitched_dir, "entireImg.ims")
        stitch_channel = 0
        mf.load.stitched_ims(zarr_path, "stitched",
                             stitched_path, stitch_channel, n_tile_y, n_tile_x)

        # ===== Max intensity projection ======
        groups = ["hcsti", "is1", "is2", "is3"]
        for group in groups:
            mf.process.projection(zarr_path, group)

        # ===== Regsitration ======
        # ----- shift - ----
        sift_kwargs = {
            "upsampling": 1, "n_octaves": 8, "n_scales": 3, "sigma_min": 2,
            "sigma_in": 1, "c_dog": 0.01, "c_edge": 40, "n_bins": 12,
            "lambda_ori": 1.5, "c_max": 0.8, "lambda_descr": 6,
            "n_hist": 4, "n_ori": 8}
        match_kwargs = {
            # "metric": "euclidean", "p": 2, "max_distance": 500,
            # "cross_check": True,
            "max_ratio": 0.5}
        ransac_kwargs = {
            "min_samples": 4, "residual_threshold": 10, "max_trials": 500}

        mf.register.shift_cycle_cYXyx(zarr_path, "hcsti_mip",
                                      sift_kwargs, match_kwargs, ransac_kwargs,
                                      subfooter="_is")
        mf.register.shift_tile_cYXyx(zarr_path, "hcsti_mip", "stitched", 100,
                                     sift_kwargs, match_kwargs, ransac_kwargs,
                                     subfooter="_is")

        mf.register.merge_shift_cYXyx(zarr_path, "hcsti_mip", subfooter="_is")

        # ----- Check and correct -----
        # Use register script

        # ----- registration  -----
        groups = ["hcsti_mip", "is1_mip", "is2_mip", "is3_mip"]
        for group in groups:
            mf.register.registration_cYXyx(zarr_path, group, "stitched", (1000, 1000),
                                           subfooter="_is")
            mf.view.make_pyramid(zarr_path, group + "_reg")

        # ----- Make whole image -----
        groups = ["hcsti_mip_reg", "is1_mip_reg",
                  "is2_mip_reg", "is3_mip_reg"]
        for group in groups:
            mf.tif.save_whole_image(zarr_path, group, 3)

        #  ===== SeqIF =====
        groups = ["is1_mip_reg", "is2_mip_reg", "is3_mip_reg"]
        for group in groups:
            mf.seqif.TCEP_subtraction(zarr_path, group)

        mf.seqif.skip_odd_cycle(zarr_path, "hcsti_mip_reg")

        groups = ["is1_mip_reg_sub",
                  "is2_mip_reg_sub", "is3_mip_reg_sub"]
        for group in groups:
            mf.view.make_pyramid(zarr_path, group)

        # ===== Surface detection =====
        group = "is1_mip_reg_sub"
        mf.segment.select_slice(zarr_path, group, "cycle", 0, None, "_sf1")
        group = "is2_mip_reg_sub"
        mf.segment.select_slice(zarr_path, group, "cycle", 0, None, "_sf2")
        group = "is2_mip_reg_sub"
        mf.segment.select_slice(zarr_path, group, "cycle", 6, None, "_sf3")
        group = "is2_mip_reg_sub"
        mf.segment.select_slice(zarr_path, group, "cycle", 11, None, "_sf4")
        group = "hcsti_mip_reg_skc"
        mf.segment.select_slice(zarr_path, group, "cycle", 0, None, "_slc")
        # merge to one group
        group_names = ["is1_mip_reg_sub_sf1", "is2_mip_reg_sub_sf2",
                       "is2_mip_reg_sub_sf3", "is2_mip_reg_sub_sf4"]
        mf.segment.merge_groups(zarr_path, group_names, "is_reg_sub_slc_mrg")
        mf.segment.normalize_groups(zarr_path, "is_reg_sub_slc_mrg")

        mf.tif.save_rgb(zarr_path, None, "is_reg_sub_slc_mrg_nrm",
                        "hcsti_mip_reg_skc_slc", "nuc_cyt_rgb")

        #  ===== Segmentation =====
        mf.tif.save(zarr_path, "hcsti_mip_reg_skc_slc")

        # Use segmentation script here


def main_2():
    root_dir = "/spo82/ana/012_SeqFISH_IF/240807/"
    pitch = [0.1650, 0.0994, 0.0994]
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
        74, 10, 10, 6, 2048, 2048

    for i in [1, 2, 3, 4, 5, 6]:
        sample_name = "012_SeqFISH_IF_" + str(i)
        zarr_path = os.path.join(root_dir, sample_name + ".zarr")
        print("===== " + sample_name + " =====")

        mf.config.set_resource(gpu=False, scheduler="processes")

        #  ===== Segmentation =====
        # ----- cellpose -----
        mf.tif.load(zarr_path, "hcsti_mip_reg_skc_slc_cpl",
                    "hcsti_mip_reg_skc_slc", "_label.tiff", "uint32")
        mf.config.set_resource(gpu=False, scheduler="processes")
        mf.segment.merge_split_label(zarr_path, "hcsti_mip_reg_skc_slc_cpl")

        groups = ["hcsti_mip_reg_skc_slc_cpl_msl"]
        for group in groups:
            mf.view.make_pyramid(zarr_path, group)

        groups = ["hcsti_mip_reg_skc_slc_cpl_msl"]
        for group in groups:
            mf.tif.save_whole_image(zarr_path, group, 3)

        mf.segment.info_csv(zarr_path, "hcsti_mip_reg_skc_slc_cpl_msl",
                            pitch[1:])

        # --- MEDIAR ---
        mf.tif.load(zarr_path, "nuc_cyt_rgb_lbl", "is_reg_sub_slc_mrg_nrm",
                    "_label.tiff", "uint32")
        mf.config.set_resource(gpu=False, scheduler="processes")
        mf.segment.merge_split_label(zarr_path, "nuc_cyt_rgb_lbl")
        mf.segment.fill_holes(zarr_path, "nuc_cyt_rgb_lbl_msl")

        groups = ["nuc_cyt_rgb_lbl_msl_fil", "hcsti_mip_reg_skc_slc",
                  "is_reg_sub_slc_mrg_nrm"]
        for group in groups:
            mf.view.make_pyramid(zarr_path, group)
        mf.segment.info_csv(zarr_path, "nuc_cyt_rgb_lbl_msl_fil", pitch[1:])

        # ===== Make whole image tif =====
        groups = ["nuc_cyt_rgb_lbl_olr_fil", "is1_mip_reg"]
        for group in groups:
            mf.tif.save_whole_image(
                zarr_path, group, zoom=4, clip=[1270, 1270])

        # ===== Compare with Cellpose =====
        mf.tif.save(zarr_path, "is_reg_sub_slc_mrg_nrm")

        groups = ["nuc_cyt_rgb_lbl_olr_fil"]
        for group in groups:
            dar = da.from_zarr(zarr_path, component=group + "/0/data")

            tif_dir = zarr_path.replace(".zarr", "_tif")
            tif_dir = os.path.join(tif_dir, group)
            if not os.path.exists(tif_dir):
                os.makedirs(tif_dir)

            tif_name = group + "_clip.tif"

            img = dar[0, :, :].compute()

            img = img[10000:14000, 10000:14000]

            tif_path = os.path.join(tif_dir, tif_name)
            tifffile.imwrite(tif_path, img)

        # ===== Save representative tif =====
        chunk = []
        for cycle in range(n_cycle):
            chunk.append([cycle, 7, 7])

        groups = ["is1_mip_reg", "is1_mip_reg_sub"]
        for group in groups:
            mf.tif.save_chunk(zarr_path, group, chunk, "0_chunk_7_7")

        # ----- make rgb label

        save_dir = "/spo82/ana/012_SeqFISH_IF/240824/"
        img_dec = tifffile.imread(os.path.join(
            save_dir, "nuc_cyt_rgb_lbl_olr_fil_zoom4.tif"))

        # get colormap of matplotlib as RGB
        cmap = plt.get_cmap("Set1")
        colors = cmap.colors

        # Convert img_dec to RGB
        img_rgb = np.zeros(
            (img_dec.shape[0], img_dec.shape[1], 3), dtype=np.uint8)

        labels = np.unique(img_dec)
        for i, label in enumerate(tqdm(labels)):
            if label == 0:
                mask = img_dec == label
                img_rgb[:, :, 0] = (mask * 255).astype(np.uint8)
                img_rgb[:, :, 1] = (mask * 255).astype(np.uint8)
                img_rgb[:, :, 2] = (mask * 255).astype(np.uint8)
            else:
                color = colors[i % len(colors)]
                mask = img_dec == label
                img_rgb[:, :, 0] += (mask * color[0] * 255).astype(np.uint8)
                img_rgb[:, :, 1] += (mask * color[1] * 255).astype(np.uint8)
                img_rgb[:, :, 2] += (mask * color[2] * 255).astype(np.uint8)

        tifffile.imwrite(os.path.join(
            save_dir, "nuc_cyt_rgb_lbl_olr_fil_zoom4_rgb.tif"), img_rgb)

        # ===== get IS intensity ======
        for group in ["is1_mip_reg_sub", "is2_mip_reg_sub", "is3_mip_reg_sub"]:
            mf.seqif.get_intensity(zarr_path, group,
                                   "nuc_cyt_rgb_lbl_olr_fil", "_mdi")

        # ----- make cell-gene table -----
        groups = ["is1_mip_reg_sub_mdi", "is2_mip_reg_sub_mdi",
                  "is3_mip_reg_sub_mdi"]
        group_seg = "nuc_cyt_rgb_lbl_olr_fil_seg"
        channels = [2, 3, 4]
        genename_path = os.path.join(
            root_dir, "012_SeqFISH_IF_genename_is.csv")
        group_out = "is_mdi"
        mf.seqif.intnensity_summary(
            zarr_path, groups, group_seg, group_out, channels, genename_path)

        # ===== NO SUB methods =====
        groups = ["is1_mip_reg", "is2_mip_reg", "is3_mip_reg"]
        for group in groups:
            mf.seqif.skip_odd_cycle(zarr_path, group)

        # ===== get IS intensity (cellpose )======
        for group in ["is1_mip_reg_skc", "is2_mip_reg_skc", "is3_mip_reg_skc"]:
            mf.seqif.get_intensity(zarr_path, group,
                                   "hcsti_mip_reg_skc_slc_cpl_msl", "_cpn")

        # ----- make cell-gene table -----
        groups = ["is1_mip_reg_skc_cpn", "is2_mip_reg_skc_cpn",
                  "is3_mip_reg_skc_cpn"]
        group_seg = "hcsti_mip_reg_skc_slc_cpl_msl_seg"
        channels = [2, 3, 4]
        genename_path = os.path.join(
            root_dir, "012_SeqFISH_IF_genename_is.csv")
        group_out = "is_cpn"
        mf.seqif.intnensity_summary(
            zarr_path, groups, group_seg, group_out, channels, genename_path)


if __name__ == "__main__":
    # Uncomment only the process you want to execute

    main_1()
    # main_2()

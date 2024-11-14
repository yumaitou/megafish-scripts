import os
import megafish as mf
import numpy as np

from dask.diagnostics import ProgressBar
import xarray as xr


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

        # ===== Loading =====
        # ----- RNA -----
        dirlist_path = os.path.join(
            root_dir, sample_name + "_rna_directorylist.csv")

        groups = ["hcstr", "rna1", "rna2", "rna3"]
        channels = [1, 2, 3, 4]
        scan_type = "snake_up_right"
        mf.load.make_imagepath_cYX_from_dirlist(
            zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x,
            scan_type, dirlist_path, "_rna")
        mf.load.ims_cYXzyx(zarr_path, n_z, n_y, n_x, "_rna_imagepath")

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
        groups = ["hcstr", "rna1", "rna2", "rna3"]
        for group in groups:
            mf.process.projection(zarr_path, group)

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

        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip",
                                      subfooter="_rna")
        mf.register.shift_tile_cYXyx(zarr_path, "hcstr_mip", "stitched", 100,
                                     subfooter="_rna")

        mf.register.shift_cycle_cYXyx(zarr_path, "hcsti_mip",
                                      sift_kwargs, match_kwargs, ransac_kwargs,
                                      subfooter="_is")
        mf.register.shift_tile_cYXyx(zarr_path, "hcsti_mip", "stitched", 100,
                                     sift_kwargs, match_kwargs, ransac_kwargs,
                                     subfooter="_is")

        mf.register.merge_shift_cYXyx(zarr_path, "hcstr_mip", "_rna")
        mf.register.merge_shift_cYXyx(zarr_path, "hcsti_mip", subfooter="_is")

        # ----- Check and correct -----
        # Use register script

        # ----- registration  -----
        groups = ["hcstr_mip", "rna1_mip", "rna2_mip", "rna3_mip"]
        for group in groups:
            mf.register.registration_cYXyx(zarr_path, group, "stitched", (1000, 1000),
                                           subfooter="_rna")
            mf.register.make_pyramid(zarr_path, group + "_reg")

        groups = ["hcsti_mip", "is1_mip", "is2_mip", "is3_mip"]
        for group in groups:
            mf.register.registration_cYXyx(zarr_path, group, "stitched", (1000, 1000),
                                           subfooter="_is")
            mf.register.make_pyramid(zarr_path, group + "_reg")

        # ----- Make whole image -----
        groups = ["hcsti_mip_reg", "is1_mip_reg",
                  "is2_mip_reg", "is3_mip_reg"]
        groups = ["hcstr_mip_reg", "rna1_mip_reg", "rna2_mip_reg",
                  "rna3_mip_reg"]
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


# run segmentation here
# using fig4/2_run_cellpose.py and fig4/3_run_MEDIAR.py

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

        # --- MEDIAR ---

        # --- post processing ----
        mf.tif.load(zarr_path, "nuc_cyt_rgb_lbl", "is_reg_sub_slc_mrg_nrm",
                    "_label.tiff", "uint32")
        mf.config.set_resource(gpu=False, scheduler="processes")
        mf.segment.merge_split_label(zarr_path, "nuc_cyt_rgb_lbl")
        mf.segment.fill_holes(zarr_path, "nuc_cyt_rgb_lbl_msl")

        groups = ["nuc_cyt_rgb_lbl_msl_fil", "hcsti_mip_reg_skc_slc",
                  "is_reg_sub_slc_mrg_nrm"]
        for group in groups:
            mf.view.make_pyramid(zarr_path, group)

        # --- Dilation (optional) ---
        mf.segment.grow_voronoi(zarr_path, "hcstr_mip_reg_mip_bpf_nuc_olr_fil", 100,
                                100)

        mf.segment.info_csv(zarr_path, "nuc_cyt_rgb_lbl_msl_fil", pitch[1:])

        mf.tif.save(zarr_path, "hcstr_mip_reg_mip_bpf_nuc_olr_fil")

        # ===== Spot detection =====
        NA = 1.4
        wavelengths_um = [0.519, 0.592, 0.671]
        mean_pitch_yx = np.mean(pitch[1:])

        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog", "rna2_mip_reg_dog",
                       "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))

        factor = 1.5
        # ------- selection ------
        group_names = ["rna1_mip_reg_dog_lmx", "rna2_mip_reg_dog_lmx",
                       "rna3_mip_reg_dog_lmx"]
        footer = "_ith"  # Intensity THresholding
        for group_name in group_names:
            print("Selecting spots: " + group_name)
            with ProgressBar():
                root = xr.open_zarr(zarr_path, group=group_name + "/0")
                xar = root["data"]
                total = xar.sum().compute()
                count = (xar != 0).sum().compute()
                ave = total / count
                sd = (xar != 0).std().compute()
                threshold = ave + factor * sd
                # convert to zero if below threshold
                res = xar.where(xar > threshold, 0)
                # save as zarr
                res.to_zarr(zarr_path, group=group_name +
                            footer + "/0", mode="w")

        # ===== count spots =====
        groups = ["rna1_mip_reg_dog_lmx_ith",
                  "rna2_mip_reg_dog_lmx_ith",
                  "rna3_mip_reg_dog_lmx_ith",]
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group,
                                   "nuc_cyt_rgb_lbl_olr_fil")

        group = "nuc_cyt_rgb_lbl_olr_fil"
        mf.segment.info_csv(zarr_path, group)

        # # ===== Intensity Calculation =====
        for group in ["is1_mip_reg_sub", "is2_mip_reg_sub", "is3_mip_reg_sub"]:
            mf.seqif.get_intensity(zarr_path, group,
                                   "nuc_cyt_rgb_lbl_olr_fil")

        # ===== Save representative tif =====
        chunk = []
        for cycle in range(n_cycle):
            chunk.append([cycle, 7, 7])

        groups = ["is1_mip_reg", "is1_mip_reg_sub"]
        for group in groups:
            mf.tif.save_chunk(zarr_path, group, chunk, "0_chunk_7_7")


if __name__ == "__main__":
    # Uncomment only the process you want to execute

    main_1()
    # main_2()

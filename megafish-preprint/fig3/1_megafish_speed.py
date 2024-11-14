import os
import time
import megafish as mf

import numpy as np

import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar


def main_1():
    root_dir = "/spo82/ana/012_SeqFISH_IF/240924/"

    pitch = [0.1650, 0.0994, 0.0994]
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = 10, 4, 4, 6, 2048, 2048

    for i in [2]:
        sample_name = "012_SeqFISH_IF_" + str(i)
        zarr_path = os.path.join(root_dir, sample_name + ".zarr")
        print("===== " + sample_name + " =====")

        # ===== Loading =====
        # ----- file preparation -----
        # remove outside of mini square

        img_dir = "/spo82/img/012_SeqFISH/2/org/"
        tile_names = [112, 113, 114, 115, 124, 125, 126, 127,
                      132, 133, 134, 135, 144, 145, 146, 147]
        dirs = os.listdir(img_dir)
        for d in dirs:
            img_names = os.listdir(os.path.join(img_dir, d))
            for img_name in img_names:

                tile_name = int(img_name[-7:-4])
                if tile_name not in tile_names:
                    os.remove(os.path.join(img_dir, d, img_name))

        # ----- RNA -----
        dirlist_path = os.path.join(
            root_dir, sample_name + "_rna_directorylist.csv")

        groups = ["hcstr", "rna1", "rna2", "rna3"]
        channels = [1, 2, 3, 4]
        scan_type = "snake_up_right"
        mf.load.make_imagepath_cYX_from_dirlist(
            zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x,
            scan_type, dirlist_path, "_rna")

        st = time.time()
        mf.config.set_scheduler("Threads")
        mf.load.ims_cYXzyx(zarr_path, n_z, n_y, n_x, "_rna_imagepath")
        print(time.time() - st)

        st = time.time()
        mf.config.set_scheduler("Processes")
        mf.load.ims_cYXzyx(zarr_path, n_z, n_y, n_x, "_rna_imagepath")
        print(time.time() - st)

        st = time.time()
        mf.config.set_scheduler("Synchronous")
        mf.load.ims_cYXzyx(zarr_path, n_z, n_y, n_x, "_rna_imagepath")
        print(time.time() - st)

        # M.2 Threads: 829.8255572319031 s
        # M.2 Processes: 351.74993991851807 s
        # M.2 Synchronous: 698.4801301956177
        # SSD Threads: 824.4896929264069
        # SSD Processes: 354.55245208740234
        # SSD Synchronous: 698.6569573879242
        # HDD Threads: 906.8037292957306
        # HDD Processes: 345.7970416545868
        # HDD Synchronous: 697.4877910614014

        # ===== Max intensity projection ======
        groups = ["hcstr", "rna1", "rna2", "rna3"]
        for group in groups:
            mf.process.projection(zarr_path, group)

        # ===== Save pseudo whole tif ======
        groups = ["hcstr_mip", "rna1_mip", "rna2_mip", "rna3_mip"]
        for group in groups:
            mf.tif.save_tile_montage(zarr_path, group, [200, 200])

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

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Threads")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", None,
                                      None, None, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", None,
                                      None, None, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Synchronous")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", None,
                                      None, None, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(True)
        mf.config.set_scheduler("Synchronous")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", None,
                                      None, None, "_rna")
        print(time.time() - st)

        # CPU Synchronous: 93.61437320709229 s
        # CPU Threads: 28.955502033233643 s
        # CPU Processes: 40.7228422164917 s
        # GPU Synchronous: 5.8643810749053955 s

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Threads")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", sift_kwargs,
                                      match_kwargs, ransac_kwargs, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", sift_kwargs,
                                      match_kwargs, ransac_kwargs, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Synchronous")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", sift_kwargs,
                                      match_kwargs, ransac_kwargs, "_rna")
        print(time.time() - st)

        st = time.time()
        mf.config.use_gpu(True)
        mf.config.set_scheduler("Processes")
        mf.register.shift_cycle_cYXyx(zarr_path, "hcstr_mip", sift_kwargs,
                                      match_kwargs, ransac_kwargs, "_rna")
        print(time.time() - st)

        # CPU Synchronous: 1143.2455370426178 s
        # CPU Threads: 893.4107768535614 s
        # CPU Processes: 238.64635562896729 s
        # GPU Synchronous: 972.1602139472961
        # GPU Processes: 237.3976378440857

        # ----- export first cycle of hcst/dapi -----

        chunks = []
        for tile_y in range(n_tile_y):
            for tile_x in range(n_tile_x):
                chunks.append([0, tile_y, tile_x, 0, 0])
        mf.tif.save_chunk(zarr_path, "hcst_mip", chunks)

        # ----- Stitched -----
        stitched_dir = "/spo16/img/20240308_018-SeqFISH_buckup/2024-03-08/"

        stitched_names = [
            "dapi_1_2024-03-08_09.33.09_FusionStitcher_F0.ims",
            "dapi_2_2024-03-08_09.40.36_FusionStitcher_F0.ims",
            "dapi_3_2024-03-08_09.48.32_FusionStitcher_F0.ims",
            "dapi_4_2024-03-08_10.10.35_FusionStitcher_F0.ims",
            "dapi_5_2024-03-08_10.18.25_FusionStitcher_F0.ims",
            "dapi_6_2024-03-08_10.25.33_FusionStitcher_F0.ims",
        ]
        stitched_path = os.path.join(stitched_dir, stitched_names[i - 1])
        mf.load.stitched_ims(zarr_path, "stitched",
                             stitched_path, 0, n_tile_y, n_tile_x)

        # ----- registration ----
        mf.register.shift_tile_cYXyx(zarr_path, "hcstr_mip", "stitched", 100,
                                     sift_kwargs, match_kwargs,
                                     ransac_kwargs, "_rna")
        mf.register.merge_shift_cYXyx(zarr_path, "hcstr_mip", "_rna")

        stitched_shape = (n_tile_y, n_tile_x, n_y, n_x)
        groups = ["hcstr_mip", "rna1_mip", "rna2_mip", "rna3_mip"]

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Threads")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("CPU Thread", time.time() - st)

        st = time.time()
        mf.config.use_gpu(True)
        mf.config.set_scheduler("Threads")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("GPU Thread", time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("CPU Processes", time.time() - st)

        st = time.time()
        mf.config.use_gpu(True)
        mf.config.set_scheduler("Processes")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("GPU Processes", time.time() - st)

        st = time.time()
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Synchronous")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("CPU Synchronous", time.time() - st)

        st = time.time()
        mf.config.use_gpu(True)
        mf.config.set_scheduler("Synchronous")
        for group in groups:
            mf.register.registration_cYXyx_noref(
                zarr_path, group, stitched_shape, (1000, 1000), "_rna")
        print("GPU Synchronous", time.time() - st)

        # CPU Synchronous: 1377.138215303421
        # CPU Threads: 453.79035782814026 s
        # CPU Processes: 185.13135194778442 s
        # GPU Synchronous:353.0775876045227
        # GPU Threads: 732.6443862915039
        # GPU Processes: 254.6411488056183

        # ===== Segmentation ======
        # ----- pre-processing -----
        mf.segment.select_slice(
            zarr_path, "hcstr_mip_reg", "cycle", 0, None, "_slc")
        mf.tif.save(zarr_path, "hcstr_mip_reg_slc")


def main_2():
    root_dir = "/spo82/ana/012_SeqFISH_IF/240924/"

    pitch = [0.1650, 0.0994, 0.0994]
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = 10, 4, 4, 6, 2048, 2048

    for i in [2]:
        sample_name = "012_SeqFISH_IF_" + str(i)
        zarr_path = os.path.join(root_dir, sample_name + ".zarr")
        print("===== " + sample_name + " =====")

        # ----- post processing -----
        mf.tif.load(zarr_path, "hcstr_mip_reg_slc_lbl", "hcstr_mip_reg_slc",
                    "_label.tiff")
        mf.segment.merge_split_label(zarr_path, "hcstr_mip_reg_slc_lbl")
        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        mf.segment.fill_holes(zarr_path, "hcstr_mip_reg_slc_lbl_msl")

        # ===== Spot detection =====
        NA = 1.4
        wavelengths_um = [0.519, 0.592, 0.671]
        mean_pitch_yx = np.mean(pitch[1:])

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Synchronous")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("CPU Synchronous", time.time() - st)

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Threads")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("CPU Threads", time.time() - st)

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("CPU Processes", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Synchronous")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("GPU Synchronous", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Threads")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("GPU Threads", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Processes")
        st = time.time()
        group_names = ["rna1_mip_reg", "rna2_mip_reg", "rna3_mip_reg"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            dog_sd1, dog_sd2 = mf.seqfish.dog_sds(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.DoG_filter(zarr_path, group_name, dog_sd1,
                                  dog_sd2, axes=(1, 2), mask_radius=9)
        group_names = ["rna1_mip_reg_dog",
                       "rna2_mip_reg_dog", "rna3_mip_reg_dog"]
        for group_name, wavelength_um in zip(group_names, wavelengths_um):
            footprint = mf.seqfish.local_maxima_footprint(
                NA, wavelength_um, mean_pitch_yx)
            mf.seqfish.local_maxima(
                zarr_path, group_name, footprint, axes=(1, 2))
        print("GPU Processes", time.time() - st)

        # CPU Synchronous: 365.49443078041077
        # CPU Threads: 105.71617913246155
        # CPU Processes: 302.2726504802704
        # GPU Synchronous: 111.28626275062561
        # GPU Threads: 186.29381155967712
        # GPU Processes: 307.13768434524536

        factor = 0
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
                print(threshold)
                # convert to zero if below threshold
                res = xar.where(xar > threshold, 0)
                # save as zarr
                res.to_zarr(zarr_path, group=group_name +
                            footer + "/0", mode="w")

        # ===== count spots =====

        groups = ["rna1_mip_reg_dog_lmx_ith",
                  "rna2_mip_reg_dog_lmx_ith",
                  "rna3_mip_reg_dog_lmx_ith"]
        group_lbl = "hcstr_mip_reg_slc_lbl_msl"

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Synchronous")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("CPU Synchronous", time.time() - st)

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Threads")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("CPU Threads", time.time() - st)

        mf.config.use_gpu(False)
        mf.config.set_scheduler("Processes")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("CPU Processes", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Synchronous")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("GPU Synchronous", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Threads")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("GPU Threads", time.time() - st)

        mf.config.use_gpu(True)
        mf.config.set_scheduler("Processes")
        st = time.time()
        for group in groups:
            mf.seqfish.count_spots(zarr_path, group, group_lbl)
        print("GPU Processes", time.time() - st)

        # CPU Synchronous: 65.33826899528503
        # CPU Threads: 49.25640106201172
        # CPU Processes:　163.69229578971863
        # GPU Synchronous: 230.3195662498474
        # GPU Threads: 642.3680515289307
        # GPU Processes: 163.48801255226135


if __name__ == "__main__":
    # Uncomment only the process you want to execute

    main_1()
    # main_2()

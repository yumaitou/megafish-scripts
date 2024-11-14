import os
import time
import megafish as mf

import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd

# this create speed benchmark csv file for Decoding


def main():
    root_dir = "/spo82/ana/240525_starfish/240926/"
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
        16, 4, 4, 1, 1600, 1600

    for i in [1]:
        sample_name = "240525_starfish_" + str(i)
        zarr_path = os.path.join(root_dir, sample_name + ".zarr")

        # First download the data from the starfish tutorial
        # Then crop to 1600x1600
        # Then make imagepath.csv with 4x4 repetitions of the cropped image

        # ===== Loading =====
        mf.load.tif_cYXzyx(zarr_path, n_z, n_y, n_x, dtype="float32")

        # ===== Skip mip and registration  =====
        group = "rna"
        ds = xr.open_zarr(zarr_path, group=group + "/0")
        res = ds["data"].isel({"z": 0, "tile_y": 0, "tile_x": 0})
        res = res.drop_vars(["z", "tile_y", "tile_x"])
        res = res.chunk(chunks={"cycle": 1, "y": 1600, "x": 1600})
        res.to_zarr(zarr_path, group=group + "_mip_reg" + "/0", mode="w")

        # ===== Make dummy mask  =====
        group = "rna_mip_reg"
        ds = xr.open_zarr(zarr_path, group=group + "/0")
        res = ds["data"].isel({"cycle": 0})
        res = res.drop_vars(["cycle"])
        res[:, :] = 1
        res = res.chunk(chunks={"y": 1600, "x": 1600})
        res.to_zarr(zarr_path, group="mask" + "/0", mode="w")

        # ===== Rechunk =====

        df_speed_path = os.path.join(root_dir, "df_speed.csv")

        chunk_sizes = [100, 200, 400, 800, 1600, 3200]
        for chunk_size in chunk_sizes:
            subfooter = "_c" + str(chunk_size)

            # ----- rna -----
            group = "rna_mip_reg"
            dar = da.from_zarr(zarr_path, component=group + "/0/data")
            res = dar.rechunk((1, chunk_size, chunk_size))

            dims = ["cycle", "y", "x"]
            coords = {"cycle": np.arange(dar.shape[0]),
                      "y": np.arange(dar.shape[1]),
                      "x": np.arange(dar.shape[2])}

            chunks = {dim: size for dim, size in zip(dims, res.chunks)}
            out = xr.DataArray(res, dims=dims, coords=coords)
            ds = out.to_dataset(name="data")
            ds = ds.chunk(chunks=chunks)
            ds.to_zarr(zarr_path, group=group + subfooter + "/0", mode="w")

            # ----- mask -----
            group = "mask"
            dar = da.from_zarr(zarr_path, component=group + "/0/data")
            res = dar.rechunk((chunk_size, chunk_size))

            dims = ["y", "x"]
            coords = {"y": np.arange(dar.shape[0]),
                      "x": np.arange(dar.shape[1])}

            chunks = {dim: size for dim, size in zip(dims, res.chunks)}
            out = xr.DataArray(res, dims=dims, coords=coords)
            ds = out.to_dataset(name="data")
            ds = ds.chunk(chunks=chunks)
            ds.to_zarr(zarr_path, group=group + subfooter + "/0", mode="w")

            def add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler, process, time):
                df_speed = pd.read_csv(df_speed_path)
                df_add = pd.DataFrame(
                    {"chunk_size": [chunk_size], "use_gpu": [use_gpu],
                     "scheduler": [scheduler], "process": [process],
                     "time": [time]})
                df_speed = pd.concat([df_speed, df_add])
                df_speed.to_csv(df_speed_path, index=False)

            # ===== Pixel decoding ======
            sigma_high = (3, 3)
            sigma = 2
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            psf = mf.decode.gaussian_kernel(
                shape=(kernel_size, kernel_size), sigma=sigma)
            iterations = 15
            sigma_low = (1, 1)
            mask_size = 5

            use_gpu = [True]
            schedulers = ["synchronous", "threads", "processes"]

            for use_gpu in use_gpu:
                for scheduler in schedulers:
                    mf.config.use_gpu(use_gpu)
                    mf.config.set_scheduler(scheduler)
                    groups = ["rna_mip_reg" + subfooter]
                    st = time.time()
                    for group in groups:
                        mf.decode.merfish_prefilter(
                            zarr_path, group, sigma_high, psf, iterations, sigma_low,
                            mask_size)
                    add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                                    "1_prefilter", time.time() - st)

            # ----- Apply each threshold -----
            use_gpu = False
            scheduler = "processes"
            mf.config.use_gpu(use_gpu)
            mf.config.set_scheduler(scheduler)
            group = "rna_mip_reg" + subfooter + "_mfp"
            thres = [104.83, 45.66, 87.186, 41.184, 89.545, 44.731, 107.29, 50.718,
                     83.262, 44.571999999999996, 79.469, 41.757, 78.45100000000001,
                     45.019, 67.742, 44.622]
            st = time.time()
            ds = xr.open_zarr(zarr_path, group=group + "/0")
            res = ds["data"]
            for i, th in enumerate(thres):
                res[i, :, :] = res[i, :, :] / th

            res = res.chunk(
                chunks={"cycle": 1, "y": chunk_size, "x": chunk_size})
            res.to_zarr(zarr_path, group=group +
                        "_scl" + "/0", mode="w")
            add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                            "2_threshold", time.time() - st)

            # ----- Normalize -----

            use_gpu = [False, True]
            schedulers = ["synchronous", "threads", "processes"]

            for use_gpu in use_gpu:
                for scheduler in schedulers:
                    mf.config.use_gpu(use_gpu)
                    mf.config.set_scheduler(scheduler)
                    st = time.time()
                    groups = ["rna_mip_reg" + subfooter + "_mfp_scl"]
                    for group in groups:
                        mf.decode.norm_value(zarr_path, group)

                    groups = [["rna_mip_reg" + subfooter + "_mfp_scl",
                               "rna_mip_reg" + subfooter + "_mfp_scl_nmv"]]
                    for group in groups:
                        mf.decode.divide_by_norm(zarr_path, group[0], group[1])
                    add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                                    "3_normalize", time.time() - st)

            use_gpu = [False, True]
            schedulers = ["synchronous", "threads", "processes"]

            for use_gpu in use_gpu:
                for scheduler in schedulers:
                    mf.config.use_gpu(use_gpu)
                    mf.config.set_scheduler(scheduler)

                    groups = ["rna_mip_reg" + subfooter + "_mfp_scl_nrm"]
                    code_paths = [
                        os.path.join(root_dir, "codebook", "codebook.npy")]

                    st = time.time()
                    for group, code_intensity_path in zip(groups, code_paths):
                        mf.decode.nearest_neighbor(
                            zarr_path, group, code_intensity_path)
                    add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                                    "4_nn", time.time() - st)

            groups = [["rna_mip_reg" + subfooter + "_mfp_scl_nmv",
                       "rna_mip_reg" + subfooter + "_mfp_scl_nrm_nnd"]]

            min_intensity = 1.77e-5
            max_distance = 0.5176
            area_limits = (2, 1000)

            st = time.time()
            for group in groups:
                mf.decode.select_decoded(zarr_path, group[0], group[1],
                                         min_intensity, max_distance,
                                         area_limits)
            add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                            "5_select", time.time() - st)

            st = time.time()
            groups = ["rna_mip_reg" +
                      subfooter + "_mfp_scl_nrm_nnd_dec"]
            for group in groups:
                mf.decode.coordinates_decoded(zarr_path, group,
                                              "mask" + subfooter)

            groups = ["rna_mip_reg" + subfooter +
                      "_mfp_scl_nrm_nnd_dec_crd"]
            for group in groups:
                mf.decode.merge_decoded_csv(zarr_path, group)
            add_to_df_speed(df_speed_path, chunk_size, use_gpu, scheduler,
                            "6_merge", time.time() - st)


if __name__ == "__main__":
    main()

import os
import re
import time
import shutil

import dask.config
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import dask.array as da

from imaris_ims_file_reader.ims import ims
import tifffile

from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.morphology import disk
from scipy.spatial import cKDTree


import cupy as cp
USE_GPU = cp.cuda.is_available()
# USE_GPU = False
if USE_GPU:
    from cucim.skimage.registration import phase_cross_correlation
    from cucim.skimage.filters import difference_of_gaussians
    from cucim.skimage.morphology import binary_dilation
    from cucim.skimage.transform import ProjectiveTransform, AffineTransform, warp
    from cucim.skimage.filters import gaussian
    from cucim.skimage.measure import centroid
    from cupyx.scipy.signal import fftconvolve
    from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, binary_fill_holes
    from cupyx.scipy.ndimage import shift as nd_shift
    from cucim.skimage.measure import regionprops_table, label
    from cuml.neighbors import NearestNeighbors
else:
    from skimage.registration import phase_cross_correlation
    from skimage.filters import difference_of_gaussians
    from skimage.morphology import binary_dilation
    from skimage.transform import ProjectiveTransform, AffineTransform, warp
    from skimage.filters import gaussian
    from skimage.measure import centroid
    from scipy.signal import fftconvolve
    from scipy.ndimage import gaussian_filter, maximum_filter, binary_fill_holes
    from scipy.ndimage import shift as nd_shift
    from skimage.measure import regionprops_table, label
    # from sklearn.neighbors import NearestNeighbors

dask.config.set(scheduler='processes')
# dask.config.set(scheduler='synchronous')

""" Additional settings
Comment out: cupyx/jit/_interface.py:173: cupy._util.experimental('cupyx.jit.rawkernel')
Comment out: all print in imaris_ims_file_reader.ims.ims
"""

# ================ util ================


def natural_sort(list_to_sort):
    def _natural_keys(text):
        def _atoi(text):
            return int(text) if text.isdigit() else text
        return [_atoi(c) for c in re.split(r"(\d+)", text)]
    return sorted(list_to_sort, key=_natural_keys)


def get_tile_yx(n_tile_y, n_tile_x, scan_type):
    """
    Generates a list of (y, x) coordinates for tiles scanned according to a
    specified pattern.

    Args:
        n_tile_y (int): The number of tiles in the y direction.
        n_tile_x (int): The number of tiles in the x direction.
        scan_type (str): The scanning pattern type, e.g., "snake_up_right" or
            "snake_right_down".

    Returns:
        list of tuple: A list of (y, x) coordinates for each tile.

    Raises:
        ValueError: If the scan_type is not supported.
    """
    if scan_type == "snake_up_right":
        tile_y = [np.arange(n_tile_y)[::-1] if i % 2 ==
                  0 else np.arange(n_tile_y) for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    elif scan_type == "snake_right_down":
        tile_x = [np.arange(n_tile_x) if i % 2 == 0 else np.arange(
            n_tile_x)[::-1] for i in range(n_tile_y)]
        tile_y = np.repeat(np.arange(n_tile_y), n_tile_x)
    elif scan_type == "snake_down_right":
        # TODO: check this
        tile_y = [np.arange(n_tile_y) if i % 2 == 0 else np.arange(
            n_tile_y)[::-1] for i in range(n_tile_x)]
        tile_x = np.repeat(np.arange(n_tile_x), n_tile_y)
    else:
        raise ValueError("Unsupported scan_type")

    # Flatten the lists if they were created with list comprehensions
    if isinstance(tile_y, list):
        tile_y = np.concatenate(tile_y)
    if isinstance(tile_x, list):
        tile_x = np.concatenate(tile_x)

    return list(zip(tile_y, tile_x))


def get_round_cycle(n_round, n_cycle):
    """
    Generates a list of (round, cycle) pairs for each round and cycle.

    Args:
        n_round (int): The number of rounds.
        n_cycle (int): The number of cycles.

    Returns:
        list of tuple: A list of (round, cycle) pairs for each round and cycle.
    """
    return [(round_, cycle) for round_
            in range(n_round) for cycle in range(n_cycle)]


# ===== load ==============================
def make_dirlist(dirlist_path, image_dir):
    """
    Create a directory list file for the given image directory.

    Args:
        dirlist_path (str): The path to save the directory list file.
        image_dir (str): The path to the image directory.
    """
    dirs = os.listdir(image_dir)
    dirs = [os.path.join(image_dir, dir_) for dir_ in dirs]
    dirs = [dir_ for dir_ in dirs if os.path.isdir(dir_)]

    df = pd.DataFrame({"folder": dirs})
    df.to_csv(dirlist_path, index=False)


def make_imagepath_cYX_from_dirlist(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        dirlist_path, subfooter="", footer="_imagepath"):

    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    group_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    df_dirlist = pd.read_csv(dirlist_path)
    dirs = df_dirlist["folder"].values

    for cycle, dir_ in enumerate(dirs):
        files = os.listdir(dir_)
        for group_name, channel in zip(groups, channels):
            files = [file for file in files if file.endswith(".ims")]
            files = natural_sort(files)
            for tile_yx, file in zip(tile_yxs, files):
                tile_y, tile_x = tile_yx
                path = os.path.join(dir_, file)
                group_rows.append(group_name)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    df = pd.DataFrame({
        "group": group_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})

    df.to_csv(imagepath_path, index=False)


def make_imagepath_cYX(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        image_dir, subfooter="", footer="_imagepath"):

    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    group_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    sub_dirs = os.listdir(image_dir)
    sub_dirs = natural_sort(sub_dirs)
    for cycle, sub_dir in enumerate(sub_dirs):
        sub_img_dir = os.path.join(image_dir, sub_dir)
        files = os.listdir(sub_img_dir)
        for group_name, channel in zip(groups, channels):
            files = [file for file in files if file.endswith(".ims")]
            files = natural_sort(files)
            for tile_yx, file in zip(tile_yxs, files):
                tile_y, tile_x = tile_yx
                path = os.path.join(sub_img_dir, file)
                group_rows.append(group_name)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    df = pd.DataFrame({
        "group": group_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})

    df.to_csv(imagepath_path, index=False)


def make_imagepath_is_cYX(
        zarr_path, groups, channels, n_cycle, n_tile_y, n_tile_x, scan_type,
        image_dir, subfooter="", footer="_imagepath"):
    """
    ims only
    imunostain only
    """

    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    group_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    for cycle in range(0, n_cycle):
        if cycle % 2 == 0:
            sub_dir = "org"
            cycle_no = str(cycle // 2 + 1).zfill(2)
        else:
            sub_dir = "TCEP"
            cycle_no = str(cycle // 2 + 1).zfill(2)

        sub_img_dir = os.path.join(image_dir, sub_dir, cycle_no)
        files = os.listdir(sub_img_dir)
        for group_name, channel in zip(groups, channels):
            files = [file for file in files if file.endswith(".ims")]
            files = natural_sort(files)
            for tile_yx, file in zip(tile_yxs, files):
                tile_y, tile_x = tile_yx
                path = os.path.join(sub_img_dir, file)
                group_rows.append(group_name)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    df = pd.DataFrame({
        "group": group_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})

    df.to_csv(imagepath_path, index=False)


def make_imagepath_rc(
        zarr_path, groups, channels, n_round, n_cycle, n_tile_y, n_tile_x, scan_type,
        image_dir, subfooter="", footer="_imagepath"):

    imagepath_path = zarr_path.replace(
        ".zarr", subfooter + footer + ".csv")

    field_dirs = os.listdir(image_dir)
    field_dirs = natural_sort(field_dirs)

    # selct folders that name contains "Field_1"
    field_dirs = [field_dir for field_dir in field_dirs
                  if "Field_2" in field_dir]

    field_dirs = natural_sort(field_dirs)
    # remove the first 2 folders (manual hyb and deleted)
    field_dirs = field_dirs[2:]

    rounds_cysles = get_round_cycle(n_round, n_cycle)
    tile_yxs = get_tile_yx(n_tile_y, n_tile_x, scan_type)

    group_rows = []
    round_rows = []
    cycle_rows = []
    tile_y_rows = []
    tile_x_rows = []
    path_rows = []
    channel_rows = []

    for group_name, channel in zip(groups, channels):
        for round_cycle, field_dir in zip(rounds_cysles, field_dirs):
            files = os.listdir(os.path.join(image_dir, field_dir))
            files = [file for file in files if file.endswith(".ims")]
            files = natural_sort(files)
            for tile_yx, file in zip(tile_yxs, files):
                round_, cycle = round_cycle
                tile_y, tile_x = tile_yx
                path = os.path.join(image_dir, field_dir, file)

                group_rows.append(group_name)
                round_rows.append(round_ + 1)
                cycle_rows.append(cycle + 1)
                tile_y_rows.append(tile_y + 1)
                tile_x_rows.append(tile_x + 1)
                path_rows.append(path)
                channel_rows.append(channel)

    df = pd.DataFrame({
        "group": group_rows, "round": round_rows, "cycle": cycle_rows,
        "tile_y": tile_y_rows, "tile_x": tile_x_rows,
        "path": path_rows, "channel": channel_rows})

    df.to_csv(imagepath_path, index=False)


def load_images_ims_cYXzyx(zarr_path, n_z, n_y, n_x, imagepath_footer="_imagepath"):
    dask.config.set(scheduler='threads')

    imagepath_path = zarr_path.replace(".zarr", imagepath_footer + ".csv")
    df_imagepath = pd.read_csv(imagepath_path)

    n_cycle = df_imagepath["cycle"].max()
    n_tile_y = df_imagepath["tile_y"].max()
    n_tile_x = df_imagepath["tile_x"].max()

    groups = df_imagepath["group"].unique()

    dims = ("cycle", "tile_y", "tile_x", "z", "y", "x")
    coords = {
        "cycle": np.arange(n_cycle),
        "tile_y": np.arange(n_tile_y), "tile_x": np.arange(n_tile_x),
        "z": np.arange(n_z), "y": np.arange(n_y), "x": np.arange(n_x), }
    chunks = (1, 1, 1, n_z, n_y, n_x)

    # save empty images
    empty_data = da.zeros(
        (n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x),
        chunks=chunks, dtype=np.uint16)

    print("Saving empty images: ")
    with ProgressBar():
        for group in groups:
            xar = xr.DataArray(empty_data, dims=dims, coords=coords)
            ds = xar.to_dataset(name="data")
            ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # load images
    zr_group = zarr.open(zarr_path, mode="a")
    for group in groups:
        print(f"Loading cYXzyx ims images: {group}")
        zr = zr_group[group + "/0"]["data"]
        group_df = df_imagepath[df_imagepath["group"] == group]
        for _, row in tqdm(group_df.iterrows(), total=len(group_df)):
            cycle = row["cycle"] - 1
            tile_y = row["tile_y"] - 1
            tile_x = row["tile_x"] - 1
            channel = row["channel"] - 1
            path = row["path"]

            img_ims = ims(path)
            print(img_ims.shape)
            if len(img_ims.shape) != 5:  # (zoom, channel, z, y, x)
                print("Unexpected shape " + str(img_ims.shape) + ": " + path)
                continue
            if img_ims.shape[1] < channel + 1:
                print("No channel found: " + path)
                continue
            img = img_ims[0, channel]
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
            img = img[:n_z, :n_y, :n_x]
            zr[cycle, tile_y, tile_x] = img


def load_images_ims_cYXzyx_partz(zarr_path, n_z, n_y, n_x, imagepath_footer="_imagepath"):
    # if data contains various z size

    dask.config.set(scheduler='threads')

    imagepath_path = zarr_path.replace(".zarr", imagepath_footer + ".csv")
    df_imagepath = pd.read_csv(imagepath_path)

    n_cycle = df_imagepath["cycle"].max()
    n_tile_y = df_imagepath["tile_y"].max()
    n_tile_x = df_imagepath["tile_x"].max()

    groups = df_imagepath["group"].unique()

    dims = ("cycle", "tile_y", "tile_x", "z", "y", "x")
    coords = {
        "cycle": np.arange(n_cycle),
        "tile_y": np.arange(n_tile_y), "tile_x": np.arange(n_tile_x),
        "z": np.arange(n_z), "y": np.arange(n_y), "x": np.arange(n_x), }
    chunks = (1, 1, 1, n_z, n_y, n_x)

    # save empty images
    empty_data = da.zeros(
        (n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x),
        chunks=chunks, dtype=np.uint16)

    print("Saving empty images: ")
    with ProgressBar():
        for group in groups:
            xar = xr.DataArray(empty_data, dims=dims, coords=coords)
            ds = xar.to_dataset(name="data")
            ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # load images
    zr_group = zarr.open(zarr_path, mode="a")
    for group in groups:
        print(f"Loading cYXzyx_partz ims images: {group}")
        zr = zr_group[group + "/0"]["data"]
        group_df = df_imagepath[df_imagepath["group"] == group]
        for _, row in tqdm(group_df.iterrows(), total=len(group_df)):
            cycle = row["cycle"] - 1
            tile_y = row["tile_y"] - 1
            tile_x = row["tile_x"] - 1
            channel = row["channel"] - 1
            path = row["path"]

            img_ims = ims(path)
            img_temp = img_ims[0, channel]
            img = np.zeros((n_z, n_y, n_x), dtype=np.uint16)
            if len(img_temp.shape) == 2:
                img[0, :img_temp.shape[0], :img_temp.shape[1]] = img_temp
            else:
                img = img_temp[:n_z, :n_y, :n_x]
            zr[cycle, tile_y, tile_x] = img


def load_images_tif_cYXzyx(zarr_path, n_z, n_y, n_x, imagepath_footer="_imagepath"):
    dask.config.set(scheduler='threads')

    imagepath_path = zarr_path.replace(".zarr", imagepath_footer + ".csv")
    df_imagepath = pd.read_csv(imagepath_path)

    n_cycle = df_imagepath["cycle"].max()
    n_tile_y = df_imagepath["tile_y"].max()
    n_tile_x = df_imagepath["tile_x"].max()

    groups = df_imagepath["group"].unique()

    dims = ("cycle", "tile_y", "tile_x", "z", "y", "x")
    coords = {
        "cycle": np.arange(n_cycle),
        "tile_y": np.arange(n_tile_y), "tile_x": np.arange(n_tile_x),
        "z": np.arange(n_z), "y": np.arange(n_y), "x": np.arange(n_x), }
    chunks = (1, 1, 1, n_z, n_y, n_x)

    # save empty images
    empty_data = da.zeros(
        (n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x),
        chunks=chunks, dtype=np.uint16)

    print("Saving empty images: ")
    with ProgressBar():
        for group in groups:
            xar = xr.DataArray(empty_data, dims=dims, coords=coords)
            ds = xar.to_dataset(name="data")
            ds.to_zarr(zarr_path, group=group + "/0", mode="w")

    # load images
    zr_group = zarr.open(zarr_path, mode="a")
    for group in groups:
        print(f"Loading cYXzyx tif images: {group}")
        zr = zr_group[group + "/0"]["data"]
        group_df = df_imagepath[df_imagepath["group"] == group]
        for _, row in tqdm(group_df.iterrows(), total=len(group_df)):
            cycle = row["cycle"] - 1
            tile_y = row["tile_y"] - 1
            tile_x = row["tile_x"] - 1
            channel = row["channel"] - 1
            path = row["path"]

            img_tif = tifffile.imread(path)
            img = img_tif.astype(np.uint16)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
            img = img[:n_z, :n_y, :n_x]
            zr[cycle, tile_y, tile_x] = img


def load_stitched(
        zarr_path, group, image_path, channel, n_tile_y, n_tile_x):

    print("Loading stitched image: " + image_path)
    stitched_img = ims(image_path)[0, channel]

    if len(stitched_img.shape) == 3:
        stitched_img = stitched_img.max(axis=0)

    n_stitched_y, n_stitched_x = stitched_img.shape
    tile_y_size = n_stitched_y // n_tile_y
    tile_x_size = n_stitched_x // n_tile_x

    tiled_stitched = np.zeros((n_tile_y, n_tile_x, tile_y_size, tile_x_size))

    for y in range(n_tile_y):
        for x in range(n_tile_x):
            tiled_stitched[y, x, :, :] = stitched_img[y * tile_y_size:(
                y + 1) * tile_y_size, x * tile_x_size:(x + 1) * tile_x_size]

    # convert to xarray
    dims = ("tile_y", "tile_x", "y", "x")
    coords = {
        "tile_y": np.arange(n_tile_y),
        "tile_x": np.arange(n_tile_x),
        "y": np.arange(tile_y_size),
        "x": np.arange(tile_x_size), }

    tiled_stitched = xr.DataArray(
        tiled_stitched, dims=dims, coords=coords)
    tiled_stitched = tiled_stitched.chunk(
        {"tile_y": 1, "tile_x": 1, "y": tile_y_size, "x": tile_x_size})
    tiled_stitched = tiled_stitched.to_dataset(name="data")

    tiled_stitched.to_zarr(zarr_path, mode="w", group=group + "/0")


def _save_tif(img, tif_dir, block_info=None):

    chunk_pos = block_info[0]["chunk-location"]
    cunnk_name = [str(pos) for pos in chunk_pos]
    cunnk_name = "_".join(cunnk_name)
    tif_path = os.path.join(tif_dir, cunnk_name + ".tif")
    tifffile.imwrite(tif_path, img)

    dummy_shape = tuple([1] * len(img.shape))
    return np.zeros(dummy_shape, dtype=np.uint8)


def save_tif(zarr_path, group):

    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group, "0")
    if os.path.exists(tif_dir):
        shutil.rmtree(tif_dir)
    os.makedirs(tif_dir)

    dar.map_blocks(_save_tif, tif_dir, dtype=np.uint8).compute()


def _save_tif_chunk(img, tif_dir, chunk_list, block_info=None):

    chunk_pos = block_info[0]["chunk-location"]

    for chunk_target in chunk_list:
        if np.all(np.array(chunk_pos) == np.array(chunk_target)):
            tif_name = [str(pos) for pos in chunk_pos]
            tif_name = "_".join(tif_name) + ".tif"
            tif_path = os.path.join(tif_dir, tif_name)
            tifffile.imwrite(tif_path, img)

    dummy_shape = tuple([1] * len(img.shape))

    return np.zeros(dummy_shape, dtype=img.dtype)


def save_tif_chunk(zarr_path, group, chunk):
    dar = da.from_zarr(zarr_path, component=group + "/0/data")
    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group, "0_chunk")
    if os.path.exists(tif_dir):
        shutil.rmtree(tif_dir)
    os.makedirs(tif_dir)

    dar.map_blocks(_save_tif_chunk, tif_dir, chunk, dtype=dar.dtype).compute()


def _load_tif(img, tif_dir, footer_ext, block_info=None):

    chunk_y = block_info[0]["chunk-location"][0]
    chunk_x = block_info[0]["chunk-location"][1]

    tif_name = str(chunk_y) + "_" + str(chunk_x) + footer_ext
    tif_path = os.path.join(tif_dir, tif_name)

    output = np.zeros(img.shape, dtype=img.dtype)
    if os.path.exists(tif_path):
        output = tifffile.imread(tif_path)

    return output


def load_tif(zarr_path, group_load, group_template, footer_ext):
    # import tiff to zarr
    dask.config.set(scheduler='threads')

    tif_dir = zarr_path.replace(".zarr", "_tif")
    tif_dir = os.path.join(tif_dir, group_load, "0")

    dar = da.from_zarr(zarr_path, component=group_template + "/0/data")

    n_y, n_x = dar.shape
    original_chunks = dar.chunks
    chunk_y, chunk_x = original_chunks

    res = da.map_blocks(_load_tif, dar, tif_dir, footer_ext,
                        dtype=dar.dtype)

    with ProgressBar():
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": chunk_y[0], "x": chunk_x[0]}

        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group_load + "/0")

# ================ register cYXyx ================


def get_yx_dims(arr):
    shape = arr.shape
    keep_dims = len(shape) - 2
    slices = [0] * keep_dims + [slice(None), slice(None)]
    return arr[tuple(slices)]


if USE_GPU:
    def _shift_cycle(
            mov_tiles, ref_tiles, sift_kwargs=None, match_kwargs=None,
            ransac_kwargs=None):

        ref_img = get_yx_dims(ref_tiles)
        mov_img = get_yx_dims(mov_tiles)

        ref_img = cp.asarray(ref_img)
        mov_img = cp.asarray(mov_img)

        # preprocess
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        # apply phase correlation
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)

        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        keep_dims = len(mov_tiles.shape) - 2
        shift_matrix_shape = (1,) * keep_dims + (2, 9)
        shift_matrix = np.zeros(shift_matrix_shape, dtype=np.float32)
        append_slices = tuple([0] * (keep_dims + 1) + [slice(None)])
        if sift_kwargs is None:
            shift_matrix[append_slices] = H_shift.params.flatten().get()
            return shift_matrix

        mov_img = nd_shift(mov_img, shift).get()
        ref_img = ref_img.get()

        # Detect keypoints
        detector_extractor_ref = SIFT(**sift_kwargs)
        try:
            detector_extractor_ref.detect_and_extract(ref_img)
        except RuntimeError:
            keypoints_ref = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

        detector_extractor_mov = SIFT(**sift_kwargs)
        try:
            detector_extractor_mov.detect_and_extract(mov_img)
        except RuntimeError:
            keypoints_mov = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

        if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
            H = AffineTransform(translation=(0, 0))
            match_keys_ref = np.zeros((0, 2), dtype=np.float32)
            match_keys_mov = np.zeros((0, 2), dtype=np.float32)
            inliers = np.zeros(0, dtype=bool)
        else:
            matches = match_descriptors(
                keypoints_ref, keypoints_mov, **match_kwargs)

            match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
            match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

            if len(match_keys_ref) < 4:
                H = AffineTransform(translation=(0, 0))
            else:
                H, inliers = ransac(
                    (np.flip(match_keys_mov, axis=-1),
                     np.flip(match_keys_ref, axis=-1)),
                    ProjectiveTransform, **ransac_kwargs)
        H_inv = cp.linalg.inv(cp.asarray(H.params))
        H = ProjectiveTransform(H_shift.params @ H_inv)
        shift_matrix[append_slices] = H.params.flatten().get()
        return shift_matrix

else:
    def _shift_cycle(
            mov_tiles, ref_tiles, sift_kwargs=None, match_kwargs=None,
            ransac_kwargs=None):

        ref_img = get_yx_dims(ref_tiles)
        mov_img = get_yx_dims(mov_tiles)

        # preprocess
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        # apply phase correlation
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)

        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        keep_dims = len(mov_tiles.shape) - 2
        shift_matrix_shape = (1,) * keep_dims + (2, 9)
        shift_matrix = np.zeros(shift_matrix_shape, dtype=np.float32)
        append_slices = tuple([0] * (keep_dims + 1) + [slice(None)])
        if sift_kwargs is None:
            shift_matrix[append_slices] = H_shift.params.flatten()
            return shift_matrix

        mov_img = nd_shift(mov_img, shift)

        # Detect keypoints
        detector_extractor_ref = SIFT(**sift_kwargs)
        try:
            detector_extractor_ref.detect_and_extract(ref_img)
        except RuntimeError:
            keypoints_ref = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

        detector_extractor_mov = SIFT(**sift_kwargs)
        try:
            detector_extractor_mov.detect_and_extract(mov_img)
        except RuntimeError:
            keypoints_mov = np.zeros((0, 2), dtype=np.float32)
        else:
            keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

        if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
            H = AffineTransform(translation=(0, 0))
            match_keys_ref = np.zeros((0, 2), dtype=np.float32)
            match_keys_mov = np.zeros((0, 2), dtype=np.float32)
            inliers = np.zeros(0, dtype=bool)
        else:
            matches = match_descriptors(
                keypoints_ref, keypoints_mov, **match_kwargs)

            match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
            match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

            if len(match_keys_ref) < 4:
                H = AffineTransform(translation=(0, 0))
            else:
                H, inliers = ransac(
                    (np.flip(match_keys_mov, axis=-1),
                     np.flip(match_keys_ref, axis=-1)),
                    ProjectiveTransform, **ransac_kwargs)
        H_inv = np.linalg.inv(H.params)
        H = ProjectiveTransform(H_shift.params @ H_inv)
        shift_matrix[append_slices] = H.params.flatten()
        return shift_matrix


def shift_cycle_cYXyx(
        zarr_path, group, sift_kwargs=None, match_kwargs=None,
        ransac_kwargs=None, subfooter="", footer="_shift_cycle"):

    root = zarr.open(zarr_path)
    zr = root[group + "/0"]["data"]
    da_zr = da.from_zarr(zr)
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = da_zr.shape

    ref_tiles = da_zr[0, :, :, :, :]
    shifts = da.map_blocks(
        _shift_cycle, da_zr, ref_tiles, sift_kwargs, match_kwargs,
        ransac_kwargs,
        dtype=np.float32, chunks=(1, 1, 1, 2, 9))

    print("Calculating cycle shifts: " + group)
    with ProgressBar():
        shift_matrix = shifts.compute()

    n_rows = n_cycle * n_tile_y * n_tile_x
    n_cols = 9 * 2
    n_indices = 3
    shift_matrix_reshape = shift_matrix.reshape(n_rows, n_cols)[:, :9]
    index_matrix = np.indices(
        (n_cycle, n_tile_y, n_tile_x)).reshape(n_indices, n_rows).T

    shift_cols = ["shift_" + str(i) for i in range(9)]

    index_shift_matrix = np.concatenate(
        (index_matrix, shift_matrix_reshape), axis=1)
    shifts_df = pd.DataFrame(
        index_shift_matrix,
        columns=["cycle", "tile_y", "tile_x"] + shift_cols)

    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


if USE_GPU:
    def _shift_tile(img, max_shift, sift_kwargs, match_kwargs, ransac_kwargs):

        ref_tiles = img.sel(refmov="ref").values
        ref_img = get_yx_dims(ref_tiles)
        ref_img = np.nan_to_num(ref_img)
        ref_img = cp.asarray(ref_img)

        mov_tiles = img.sel(refmov="mov").values
        mov_img = get_yx_dims(mov_tiles)
        mov_img = np.nan_to_num(mov_img)
        mov_img = cp.asarray(mov_img)

        # preprocess
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        ref_img = np.nan_to_num(ref_img)
        mov_img = np.nan_to_num(mov_img)

        # apply phase correlation
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)

        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        shift_matrix = np.zeros((1, 1, 9), dtype=np.float32)
        if sift_kwargs is None:
            shift_matrix[0, 0, :] = H_shift.params.flatten().get()
        else:
            mov_img = nd_shift(mov_img, shift).get()
            ref_img = ref_img.get()

            # Detect keypoints
            detector_extractor_ref = SIFT(**sift_kwargs)
            try:
                detector_extractor_ref.detect_and_extract(ref_img)
            except RuntimeError:
                keypoints_ref = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

            detector_extractor_mov = SIFT(**sift_kwargs)
            try:
                detector_extractor_mov.detect_and_extract(mov_img)
            except RuntimeError:
                keypoints_mov = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

            if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
                H = AffineTransform(translation=(0, 0))
                match_keys_ref = np.zeros((0, 2), dtype=np.float32)
                match_keys_mov = np.zeros((0, 2), dtype=np.float32)
                inliers = np.zeros(0, dtype=bool)
            else:
                matches = match_descriptors(
                    keypoints_ref, keypoints_mov, **match_kwargs)

                match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
                match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

                if len(match_keys_ref) < 4:
                    H = AffineTransform(translation=(0, 0))
                else:
                    H, inliers = ransac(
                        (np.flip(match_keys_mov, axis=-1),
                         np.flip(match_keys_ref, axis=-1)),
                        ProjectiveTransform, **ransac_kwargs)
            H_inv = cp.linalg.inv(cp.asarray(H.params))
            H = ProjectiveTransform(H_shift.params @ H_inv)

            if cp.linalg.norm(H.params[0:2, 2]) > max_shift:
                H = AffineTransform(translation=(0, 0))
            shift_matrix[0, 0, :] = H.params.flatten().get()

        res = xr.DataArray(
            shift_matrix,
            dims=["tile_y", "tile_x", "shift"],
            coords={
                "tile_y": img.coords["tile_y"],
                "tile_x": img.coords["tile_x"],
                "shift": np.arange(9)})
        return res

else:
    def _shift_tile(img, max_shift, sift_kwargs, match_kwargs, ransac_kwargs):

        ref_tiles = img.sel(refmov="ref").values
        ref_img = get_yx_dims(ref_tiles)
        ref_img = np.nan_to_num(ref_img)

        mov_tiles = img.sel(refmov="mov").values
        mov_img = get_yx_dims(mov_tiles)
        mov_img = np.nan_to_num(mov_img)

        # preprocess
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min())

        ref_img = np.nan_to_num(ref_img)
        mov_img = np.nan_to_num(mov_img)

        # apply phase correlation
        shift, error, diffphase = phase_cross_correlation(
            ref_img, mov_img, normalization=None)  # (y, x)

        H_shift = AffineTransform(translation=(-shift[1], -shift[0]))

        shift_matrix = np.zeros((1, 1, 9), dtype=np.float32)
        if sift_kwargs is None:
            shift_matrix[0, 0, :] = H_shift.params.flatten()
        else:
            mov_img = nd_shift(mov_img, shift)

            # Detect keypoints
            detector_extractor_ref = SIFT(**sift_kwargs)
            try:
                detector_extractor_ref.detect_and_extract(ref_img)
            except RuntimeError:
                keypoints_ref = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_ref = detector_extractor_ref.keypoints  # (x, y)

            detector_extractor_mov = SIFT(**sift_kwargs)
            try:
                detector_extractor_mov.detect_and_extract(mov_img)
            except RuntimeError:
                keypoints_mov = np.zeros((0, 2), dtype=np.float32)
            else:
                keypoints_mov = detector_extractor_mov.keypoints  # (x, y)

            if len(keypoints_ref) == 0 or len(keypoints_mov) == 0:
                H = AffineTransform(translation=(0, 0))
                match_keys_ref = np.zeros((0, 2), dtype=np.float32)
                match_keys_mov = np.zeros((0, 2), dtype=np.float32)
                inliers = np.zeros(0, dtype=bool)
            else:
                matches = match_descriptors(
                    keypoints_ref, keypoints_mov, **match_kwargs)

                match_keys_ref = keypoints_ref[matches[:, 0]]  # (x, y)
                match_keys_mov = keypoints_mov[matches[:, 1]]  # (x, y)

                if len(match_keys_ref) < 4:
                    H = AffineTransform(translation=(0, 0))
                else:
                    H, inliers = ransac(
                        (np.flip(match_keys_mov, axis=-1),
                         np.flip(match_keys_ref, axis=-1)),
                        ProjectiveTransform, **ransac_kwargs)
            H_inv = np.linalg.inv(H.params)
            H = ProjectiveTransform(H_shift.params @ H_inv)

            if np.linalg.norm(H.params[0:2, 2]) > max_shift:
                H = AffineTransform(translation=(0, 0))
            shift_matrix[0, 0, :] = H.params.flatten()

        res = xr.DataArray(
            shift_matrix,
            dims=["tile_y", "tile_x", "shift"],
            coords={
                "tile_y": img.coords["tile_y"],
                "tile_x": img.coords["tile_x"],
                "shift": np.arange(9)})
        return res


def shift_tile_cYXyx(
        zarr_path, group_mov, group_stitched, max_shift=100, sift_kwargs=None, match_kwargs=None,
        ransac_kwargs=None, subfooter="", footer="_shift_tile"):

    group_ref = group_stitched + "/0"
    root = xr.open_zarr(zarr_path, group=group_ref)
    xar_ref = root["data"]

    xar_ref = xar_ref.expand_dims(dim={"refmov": ["ref"]}, axis=[0])

    group_mov = group_mov + "/0"
    root = xr.open_zarr(zarr_path, group=group_mov)
    xar_mov = root["data"]
    xar_mov = xar_mov.isel(cycle=0)  # TODO
    n_tile_y, n_tile_x, n_y, n_x = xar_mov.shape
    xar_mov = xar_mov.expand_dims(dim={"refmov": ["mov"]}, axis=[0])

    xar_in = xr.concat([xar_ref, xar_mov], dim="refmov")
    xar_in = xar_in.chunk({
        "refmov": 2, "tile_y": 1, "tile_x": 1, "y": n_y, "x": n_x})

    new_dims = ["tile_y", "tile_x", "shift"]
    new_coords = {
        "tile_y": np.arange(n_tile_y),
        "tile_x": np.arange(n_tile_x),
        "shift": np.arange(9)}
    template = xr.DataArray(
        da.empty((n_tile_y, n_tile_x, 9), dtype=np.float32, chunks=(1, 1, 9)),
        dims=new_dims, coords=new_coords)

    res = xar_in.map_blocks(
        _shift_tile,
        kwargs={
            "max_shift": max_shift,
            "sift_kwargs": sift_kwargs,
            "match_kwargs": match_kwargs,
            "ransac_kwargs": ransac_kwargs},
        template=template)

    print("Calculating tile shifts: " + group_mov)
    with ProgressBar():
        res = res.compute()

    n_tile_y, n_tile_x, n_shift = res.shape

    n_rows = n_tile_y * n_tile_x
    n_cols = 9
    shift_matrix = res.values.reshape(n_rows, n_cols)

    index_matrix = np.indices((n_tile_y, n_tile_x)).reshape(
        2, n_rows).T

    sift_cols = ["shift_" + str(i) for i in range(9)]
    index_shift_matrix = np.concatenate(
        (index_matrix, shift_matrix), axis=1)
    shifts_df = pd.DataFrame(
        index_shift_matrix,
        columns=["tile_y", "tile_x"] + sift_cols)

    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


def merge_shift_cYXyx(
        zarr_path, group, subfooter="", cycle_footer="_shift_cycle",
        tile_footer="_shift_tile", footer="_shift_tile_cycle"):
    # merege cycle shift and tile shift

    shift_tile_path = zarr_path.replace(
        ".zarr", subfooter + tile_footer + ".csv")
    shift_cycle_path = zarr_path.replace(
        ".zarr", subfooter + cycle_footer + ".csv")

    shifts_tile_df = pd.read_csv(shift_tile_path)
    shifts_cycle_df = pd.read_csv(shift_cycle_path)

    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = xar.shape

    shift_cols = ["shift_" + str(i) for i in range(9)]

    cycles = []
    tiles_y = []
    tiles_x = []
    shifts = []

    for cycle in range(n_cycle):
        for tile_y in range(n_tile_y):
            for tile_x in range(n_tile_x):
                shift_tile = shifts_tile_df[
                    (shifts_tile_df["tile_y"] == tile_y) &
                    (shifts_tile_df["tile_x"] == tile_x)
                ]
                shift_cycle = shifts_cycle_df[
                    (shifts_cycle_df["cycle"] == cycle) &
                    (shifts_cycle_df["tile_y"] == tile_y) &
                    (shifts_cycle_df["tile_x"] == tile_x)
                ]

                shift_tile = shift_tile[shift_cols].values
                shift_cycle = shift_cycle[shift_cols].values

                H_tile = shift_tile.reshape(3, 3)
                H_cycle = shift_cycle.reshape(3, 3)

                H = H_cycle @ H_tile

                shifts.append(H.flatten())
                cycles.append(cycle)
                tiles_y.append(tile_y)
                tiles_x.append(tile_x)

    index_array = np.array([cycles, tiles_y, tiles_x]).T
    shifts = np.array(shifts)
    index_shifts = np.concatenate((index_array, shifts), axis=1)
    shifts_df = pd.DataFrame(
        index_shifts,
        columns=["cycle", "tile_y", "tile_x"] + shift_cols)

    # save to csv
    shifts_df.to_csv(zarr_path.replace(
        ".zarr", subfooter + footer + ".csv"), index=False)


if USE_GPU:
    def get_edges(n_cycle, n_tile_y, n_tile_x, df_shift, n_y_stitched,
                  n_x_stitched, n_y, n_x):

        margin = 500

        edges = []
        for cycle in range(n_cycle):
            for tile_y in range(n_tile_y):
                for tile_x in range(n_tile_x):

                    edge = ((0, 0), (n_x, 0), (0, n_y),
                            (n_x, n_y))

                    offset = (tile_x * n_x_stitched, tile_y * n_y_stitched)

                    edge_offset = [(edge[0][0] + offset[0],
                                    edge[0][1] + offset[1]),
                                   (edge[1][0] + offset[0],
                                    edge[1][1] + offset[1]),
                                   (edge[2][0] + offset[0],
                                    edge[2][1] + offset[1]),
                                   (edge[3][0] + offset[0],
                                    edge[3][1] + offset[1])]

                    # get H
                    shift = df_shift[
                        (df_shift["cycle"] == cycle) &
                        (df_shift["tile_y"] == tile_y) &
                        (df_shift["tile_x"] == tile_x)]
                    shift_cols = ["shift_" + str(i) for i in range(9)]
                    H_mat = shift[shift_cols].values[0].reshape(3, 3)
                    H_inv = np.linalg.inv(H_mat)
                    H = ProjectiveTransform(matrix=cp.asarray(H_inv))

                    edge_offset = cp.array(edge_offset).T
                    edge_offset = cp.vstack((edge_offset, cp.ones(
                        edge_offset.shape[1])))  # make (x, y, 1)
                    edge_offset = H.params @ edge_offset  # apply H
                    edge_offset = edge_offset[:2, :].T  # remove 1

                    max_x = edge_offset[:, 0].max() + margin
                    min_x = edge_offset[:, 0].min() - margin
                    max_y = edge_offset[:, 1].max() + margin
                    min_y = edge_offset[:, 1].min() - margin

                    edges.append(
                        [cycle, tile_y, tile_x, min_y, max_y, min_x, max_x])

        edges = pd.DataFrame(
            edges, columns=["cycle", "tile_y", "tile_x",
                            "min_y", "max_y", "min_x", "max_x"])
        return edges

else:
    def get_edges(n_cycle, n_tile_y, n_tile_x, df_shift, n_y_stitched,
                  n_x_stitched, n_y, n_x):

        margin = 500

        edges = []
        for cycle in range(n_cycle):
            for tile_y in range(n_tile_y):
                for tile_x in range(n_tile_x):

                    edge = ((0, 0), (n_x, 0), (0, n_y),
                            (n_x, n_y))

                    offset = (tile_x * n_x_stitched, tile_y * n_y_stitched)

                    edge_offset = [(edge[0][0] + offset[0],
                                    edge[0][1] + offset[1]),
                                   (edge[1][0] + offset[0],
                                    edge[1][1] + offset[1]),
                                   (edge[2][0] + offset[0],
                                    edge[2][1] + offset[1]),
                                   (edge[3][0] + offset[0],
                                    edge[3][1] + offset[1])]

                    # get H
                    shift = df_shift[
                        (df_shift["cycle"] == cycle) &
                        (df_shift["tile_y"] == tile_y) &
                        (df_shift["tile_x"] == tile_x)]
                    shift_cols = ["shift_" + str(i) for i in range(9)]
                    H_mat = shift[shift_cols].values[0].reshape(3, 3)
                    H_inv = np.linalg.inv(H_mat)
                    H = ProjectiveTransform(matrix=H_inv)

                    edge_offset = np.array(edge_offset).T
                    edge_offset = np.vstack((edge_offset, np.ones(
                        edge_offset.shape[1])))  # make (x, y, 1)
                    edge_offset = H.params @ edge_offset  # apply H
                    edge_offset = edge_offset[:2, :].T  # remove 1

                    max_x = edge_offset[:, 0].max() + margin
                    min_x = edge_offset[:, 0].min() - margin
                    max_y = edge_offset[:, 1].max() + margin
                    min_y = edge_offset[:, 1].min() - margin

                    edges.append(
                        [cycle, tile_y, tile_x, min_y, max_y, min_x, max_x])

        edges = pd.DataFrame(
            edges, columns=["cycle", "tile_y", "tile_x",
                            "min_y", "max_y", "min_x", "max_x"])
        return edges


def create_chunk_dataframe(shape, chunk_size):

    def normalize_chunks(chunks, shape):
        num_chunks = [(shape[i] + chunks[i] - 1) // chunks[i]
                      for i in range(len(shape))]
        chunk_sizes = [
            [chunks[i]] * (num_chunks[i] - 1) + [
                shape[i] - chunks[i] * (num_chunks[i] - 1)]
            for i in range(len(shape))]

        return chunk_sizes

    def get_chunk_coordinates(shape, chunk_size):
        chunk_dims = normalize_chunks(chunk_size, shape)
        for y, _ in enumerate(chunk_dims[0]):
            for x, _ in enumerate(chunk_dims[1]):
                yield y, x, sum(chunk_dims[0][:y]), \
                    sum(chunk_dims[0][:y + 1]), sum(chunk_dims[1][:x]), \
                    sum(chunk_dims[1][:x + 1])

    data = list(get_chunk_coordinates(shape, chunk_size))
    return pd.DataFrame(data, columns=['chunk_y', 'chunk_x', 'upper_y', 'lower_y', 'left_x', 'right_x'])


def get_overlap_c(chunk_sel, tiles_df, cycle):
    """
    Check if any tiles in tiles_df overlap with the given chunk_sel.

    Args:
        chunk_sel (pd.Series): A single row representing a chunk.
        tiles_df (pd.DataFrame): DataFrame containing multiple tiles.
        cycle (int): Cycle number.

    Returns:
        pd.DataFrame: DataFrame of tiles that overlap with the chunk.
    """
    tiles_df_cycle = tiles_df[
        (tiles_df["cycle"] == cycle)]

    right_in = (chunk_sel["left_x"] <= tiles_df_cycle["max_x"]) & (
        tiles_df_cycle["max_x"] <= chunk_sel["right_x"])
    left_in = (chunk_sel["left_x"] <= tiles_df_cycle["min_x"]) & (
        tiles_df_cycle["min_x"] <= chunk_sel["right_x"])
    width_in = (tiles_df_cycle["min_x"] <= chunk_sel["left_x"]) & (
        chunk_sel["right_x"] <= tiles_df_cycle["max_x"])

    or_width_in = right_in | left_in | width_in

    upper_in = (chunk_sel["upper_y"] <= tiles_df_cycle["max_y"]) & (
        tiles_df_cycle["max_y"] <= chunk_sel["lower_y"])
    lower_in = (chunk_sel["upper_y"] <= tiles_df_cycle["min_y"]) & (
        tiles_df_cycle["min_y"] <= chunk_sel["lower_y"])
    height_in = (tiles_df_cycle["min_y"] <= chunk_sel["upper_y"]) & (
        chunk_sel["lower_y"] <= tiles_df_cycle["max_y"])

    or_height_in = upper_in | lower_in | height_in

    return tiles_df_cycle[or_width_in & or_height_in]


if USE_GPU:
    def _register_chunk_c(input_img, zarr_path, group_name, df_chunk, df_tile, df_H,
                          n_y, n_x, chunk_size, block_info=None):

        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        dar_img = da.from_zarr(zarr_path, component=group_name + "/0/data")

        chunk_sel = df_chunk[
            (df_chunk["chunk_y"] == chunk_y) &
            (df_chunk["chunk_x"] == chunk_x)].iloc[0]

        overlap = get_overlap_c(
            chunk_sel, df_tile, cycle)
        overlap = overlap.merge(
            df_H, on=["cycle", "tile_y", "tile_x"])

        overlap["offset_y"] = overlap["tile_y"] * n_y - chunk_sel["upper_y"]
        overlap["offset_x"] = overlap["tile_x"] * n_x - chunk_sel["left_x"]

        tile_img = cp.zeros(chunk_size)
        for i, row in overlap.iterrows():
            shift_cols = ["shift_" + str(i) for i in range(9)]
            H_mat = row[shift_cols].values.reshape(3, 3).astype(np.float32)
            H_mat = cp.array(H_mat)

            offset = cp.array([-row["offset_x"], -row["offset_y"]])
            H_offset = cp.eye(3)
            H_offset[:2, 2] = offset

            H_mat = H_mat @ H_offset
            H = AffineTransform(matrix=H_mat)

            tile_img_add = dar_img[cycle,
                                   int(row["tile_y"]), int(row["tile_x"])]
            tile_img_add = cp.asarray(tile_img_add.compute())
            tile_img_add = warp(tile_img_add, H, output_shape=chunk_size,
                                preserve_range=True, order=0)
            tile_img = np.maximum(tile_img, tile_img_add)

        upper_y = chunk_sel["upper_y"]
        lower_y = chunk_sel["lower_y"]
        left_x = chunk_sel["left_x"]
        right_x = chunk_sel["right_x"]

        if lower_y - upper_y != chunk_size[0] or right_x - left_x != chunk_size[1]:
            tile_img = tile_img[:lower_y - upper_y, :right_x - left_x]

        out_img = np.zeros((1, chunk_size[0], chunk_size[1]))
        out_img[0, :tile_img.shape[0], :tile_img.shape[1]] = tile_img.get()
        return out_img
else:
    def _register_chunk_c(input_img, zarr_path, group_name, df_chunk, df_tile, df_H,
                          n_y, n_x, chunk_size, block_info=None):

        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        dar_img = da.from_zarr(zarr_path, component=group_name + "/0/data")

        chunk_sel = df_chunk[
            (df_chunk["chunk_y"] == chunk_y) &
            (df_chunk["chunk_x"] == chunk_x)].iloc[0]

        overlap = get_overlap_c(
            chunk_sel, df_tile, cycle)
        overlap = overlap.merge(
            df_H, on=["cycle", "tile_y", "tile_x"])

        overlap["offset_y"] = overlap["tile_y"] * n_y - chunk_sel["upper_y"]
        overlap["offset_x"] = overlap["tile_x"] * n_x - chunk_sel["left_x"]

        tile_img = np.zeros(chunk_size)
        for i, row in overlap.iterrows():
            shift_cols = ["shift_" + str(i) for i in range(9)]
            H_mat = row[shift_cols].values.reshape(3, 3).astype(np.float32)

            offset = np.array([-row["offset_x"], -row["offset_y"]])
            H_offset = np.eye(3)
            H_offset[:2, 2] = offset

            H_mat = H_mat @ H_offset
            H = AffineTransform(matrix=H_mat)

            tile_img_add = dar_img[cycle,
                                   int(row["tile_y"]), int(row["tile_x"])]
            tile_img_add = tile_img_add.compute()
            tile_img_add = warp(tile_img_add, H, output_shape=chunk_size,
                                preserve_range=True, order=0)
            tile_img = np.maximum(tile_img, tile_img_add)

        upper_y = chunk_sel["upper_y"]
        lower_y = chunk_sel["lower_y"]
        left_x = chunk_sel["left_x"]
        right_x = chunk_sel["right_x"]

        if lower_y - upper_y != chunk_size[0] or right_x - left_x != chunk_size[1]:
            tile_img = tile_img[:lower_y - upper_y, :right_x - left_x]

        out_img = np.zeros((1, chunk_size[0], chunk_size[1]))
        out_img[0, :tile_img.shape[0], :tile_img.shape[1]] = tile_img
        return out_img


def registration_cYXyx(zarr_path, group_tile, group_ref, chunk_size,
                       subfooter="", shift_footer="_shift_tile_cycle",
                       footer="_reg"):

    if USE_GPU:
        dask.config.set(scheduler='synchronous')
    else:
        dask.config.set(scheduler='threads')

    shift_path = zarr_path.replace(
        ".zarr", subfooter + shift_footer + ".csv")
    df_H = pd.read_csv(shift_path)

    dar_img = da.from_zarr(zarr_path, component=group_tile + "/0/data")
    n_cycle, n_tile_y, n_tile_x, n_y, n_x = dar_img.shape

    dar_stitched = da.from_zarr(zarr_path, component=group_ref + "/0/data")

    n_tile_stiched_y, n_tile_stiched_x, n_y_stitched, n_x_stitched = \
        dar_stitched.shape

    df_tile = get_edges(n_cycle, n_tile_y, n_tile_x, df_H,
                        n_y_stitched, n_x_stitched, n_y, n_x)

    shape = (n_y_stitched * n_tile_stiched_y,
             n_x_stitched * n_tile_stiched_x)
    df_chunk = create_chunk_dataframe(shape, chunk_size)

    n_chunk_y = df_chunk["chunk_y"].max() + 1
    n_chunk_x = df_chunk["chunk_x"].max() + 1
    chunk_w = n_chunk_x * chunk_size[1]
    chunk_h = n_chunk_y * chunk_size[0]

    dar_chunk = da.zeros((n_cycle, chunk_h, chunk_w),
                         dtype=dar_img.dtype,
                         chunks=(1, chunk_size[0], chunk_size[1]))

    dar_res = da.map_blocks(
        _register_chunk_c, dar_chunk, zarr_path, group_tile, df_chunk, df_tile,
        df_H, n_y, n_x, chunk_size, dtype=dar_img.dtype,
        chunks=(1, chunk_size[0], chunk_size[1]))

    print("Registering: " + group_tile)
    with ProgressBar():
        dims = ["cycle", "y", "x"]
        coords = {
            "cycle": range(n_cycle),
            "y": range(chunk_h),
            "x": range(chunk_w)}
        chunks = {"cycle": 1,
                  "y": chunk_size[0], "x": chunk_size[1]}

        out = xr.DataArray(dar_res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, group=group_tile + footer + "/0", mode="w")

# ================ view ================


def make_pyramid(zarr_path, group, level=None):

    dar_in = da.from_zarr(
        zarr_path, component=group + "/0/data")
    init_size = dar_in.shape
    index_dims = len(init_size) - 2

    if index_dims == 0:
        n_indices = []
    else:
        n_indices = list(init_size[:-2])
    chunk_size = dar_in.chunksize

    max_zoom = 0
    zoom_size = init_size
    while zoom_size[-2] // 2 >= chunk_size[-2] \
            and zoom_size[-1] // 2 >= chunk_size[-1]:
        zoom_size_yx = [(zoom_size[-2] + 1) // 2, (zoom_size[-1] + 1) // 2]
        zoom_size = tuple(n_indices + zoom_size_yx)
        max_zoom += 1
    max_zoom += 1

    def func(block):
        new_block = np.zeros(chunk_size, dtype=block.dtype)
        clip_slices = [slice(None)] * len(n_indices) + \
            [slice(None, block.shape[-2]), slice(None, block.shape[-1])]
        new_block[tuple(clip_slices)] = block

        skip_slices = [slice(None)] * len(n_indices) + \
            [slice(None, None, 2), slice(None, None, 2)]
        new_block = new_block[tuple(skip_slices)]
        return new_block

    if level is not None:
        zooms = [level]
    else:
        zooms = range(1, max_zoom + 1)

    print("Making pyramid: ")
    for i in zooms:
        print(group + ": level " + str(i))
        dar_in = da.from_zarr(
            zarr_path, component=group + "/" + str(i - 1) + "/data")

        chunk_slices = [1] * len(n_indices) + \
            [chunk_size[-2] // 2, chunk_size[-1] // 2]

        dar_dist = da.map_blocks(
            func, dar_in, dtype=dar_in.dtype,
            chunks=tuple(chunk_slices))

        if index_dims == 0:
            dims = ["y", "x"]
        if index_dims == 1:
            dims = ['cycle', "y", "x"]
        if index_dims == 2:
            dims = ['round', 'cycle', "y", "x"]

        coords = {}
        for k, n in enumerate(n_indices):
            coords[dims[k]] = np.arange(n)
        coords['y'] = np.arange(dar_dist.shape[-2])
        coords['x'] = np.arange(dar_dist.shape[-1])

        chunk_size_dict = {}
        for k, n in enumerate(n_indices):
            chunk_size_dict[dims[k]] = 1
        chunk_size_dict['y'] = chunk_size[-2]
        chunk_size_dict['x'] = chunk_size[-1]

        out = xr.DataArray(dar_dist, dims=dims, coords=coords)
        out = out.to_dataset(name='data')
        out = out.chunk(chunk_size_dict)

        out.to_zarr(zarr_path, mode='w', consolidated=True,
                    group=group + "/" + str(i))


def _dilation(img, footprint):
    img = img.astype(np.float32)
    if USE_GPU:
        frm = cp.asarray(img.values)
    else:
        frm = img.values

    result = binary_dilation(frm, footprint=footprint)

    if USE_GPU:
        res_array = xr.DataArray(
            result.get(), dims=img.dims, coords=img.coords)
    else:
        res_array = xr.DataArray(
            result, dims=img.dims, coords=img.coords)
    return res_array


def dilation(zarr_path, group, mask_radius, footer="_dil"):

    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    x_shape = np.ones(len(xar.dims), dtype=np.int8)
    x_shape[-2] = 3
    y_shape = np.ones(len(xar.dims), dtype=np.int8)
    y_shape[-1] = 3
    if USE_GPU:
        footprint = [(cp.ones(tuple(x_shape)), mask_radius),
                     (cp.ones(tuple(y_shape)), mask_radius)]
    else:
        footprint = [(np.ones(tuple(x_shape)), mask_radius),
                     (np.ones(tuple(y_shape)), mask_radius)]

    with ProgressBar():
        dil = xar.map_blocks(
            _dilation, kwargs=dict(footprint=footprint),
            template=template)

        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")

# ================ process ================


def projection(zarr_path, group, dim="z", method="max",
               footer="_mip"):

    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}
    chunk_dict.pop(dim)

    print("Making " + method + " " + dim + " projection: " + group)
    with ProgressBar():
        if method == "max":
            res = xar.max(dim=dim)
        elif method == "min":
            res = xar.min(dim=dim)
        else:
            raise ValueError("Unsupported method")

        res = res.chunk(chunks=chunk_dict)
        res.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def _DoG_filter(
        img, low_sigmas, high_sigmas, footprint):

    img = img.astype(np.float32)
    frm = img.values

    if USE_GPU:
        frm = cp.asarray(frm)
    result = difference_of_gaussians(frm, low_sigmas, high_sigmas)

    if footprint is not None:
        mask = frm == 0
        mask = binary_dilation(mask, footprint=footprint)
        result[mask] = 0

    if USE_GPU:
        result = result.get()

    res_array = xr.DataArray(result, dims=img.dims, coords=img.coords)

    return res_array


def DoG_filter(
        zarr_path, group_name, dog_sd1, dog_sd2, axes, mask_radius=None,
        footer="_dog"):
    if USE_GPU:
        dask.config.set(scheduler='synchronous')
    root = xr.open_zarr(zarr_path, group=group_name + "/0")
    xar = root["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    low_sigmas = np.zeros(len(xar.dims))
    high_sigmas = np.zeros(len(xar.dims))
    for ax in axes:
        low_sigmas[ax] = dog_sd1
        high_sigmas[ax] = dog_sd2

    if mask_radius is not None:
        x_shape = np.ones(len(xar.dims), dtype=np.int8)
        x_shape[-2] = 3
        y_shape = np.ones(len(xar.dims), dtype=np.int8)
        y_shape[-1] = 3

        if USE_GPU:
            footprint = [(cp.ones(tuple(x_shape)), mask_radius),
                         (cp.ones(tuple(y_shape)), mask_radius)]
        else:
            footprint = [(np.ones(tuple(x_shape)), mask_radius),
                         (np.ones(tuple(y_shape)), mask_radius)]

    else:
        footprint = None

    with ProgressBar():
        dog = xar.map_blocks(
            _DoG_filter, kwargs=dict(
                low_sigmas=low_sigmas, high_sigmas=high_sigmas,
                footprint=footprint),
            template=template)
        original_chunks = xar.chunks
        dog = dog.chunk(original_chunks)
        ds = dog.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")


def dog_sds(NA, wavelength, pitch, psf_size_factor=1, dog_sd_factor=1):
    d_psf_pix = (1.22 * wavelength) / (NA * pitch)
    particle_size = psf_size_factor * d_psf_pix

    dog_sd1 = particle_size / (1 + np.sqrt(2))
    dog_sd2 = np.sqrt(2) * dog_sd1 * dog_sd_factor

    return dog_sd1, dog_sd2


if USE_GPU:
    def _local_maxima(img, footprint):
        img_cp = cp.asarray(img.copy())
        lmx = maximum_filter(
            img_cp, footprint=cp.asarray(footprint))

        lmx[cp.logical_not(lmx == img_cp)] = 0

        res_array = xr.DataArray(
            lmx.get(), dims=img.dims, coords=img.coords)

        return res_array

else:
    def _local_maxima(img, footprint):
        lmx = maximum_filter(
            img, footprint=footprint)

        lmx[np.logical_not(lmx == img)] = 0

        res_array = xr.DataArray(
            lmx, dims=img.dims, coords=img.coords)

        return res_array


def local_maxima(zarr_path, group_name, footprint, axes, footer="_lmx"):
    if USE_GPU:
        dask.config.set(scheduler='synchronous')

    root = xr.open_zarr(zarr_path, group=group_name + "/0")
    xar = root["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    for _ in range(axes[0]):
        footprint = np.expand_dims(footprint, axis=0)

    with ProgressBar():
        lmx = xar.map_blocks(_local_maxima, kwargs=dict(
            footprint=footprint), template=template)

        original_chunks = xar.chunks
        lmx = lmx.chunk(original_chunks)
        ds = lmx.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")


def local_maxima_footprint(NA, wavelength_um, pitch_um):
    d_psf_pix = (1.22 * wavelength_um) / (NA * pitch_um)
    mindist = np.round(d_psf_pix * 0.5)
    footprint = disk(mindist)
    return footprint


def _apply_offset(img, offset, block_info=None):

    chunk_y = block_info[0]["chunk-location"][0]
    chunk_x = block_info[0]["chunk-location"][1]

    img_offset = img + offset[chunk_y, chunk_x]
    img_offset = img_offset * (img > 0)

    return img_offset


def _change_value(img, src_list, dst_list, block_info=None):

    img_mod = img.copy()
    # get unique value from img_mod
    img_mod_unique = np.unique(img_mod)

    src_list = list(src_list)
    dst_list = list(dst_list)

    present_src = [src for src in src_list if src in img_mod_unique]
    corresponding_dst = [dst_list[src_list.index(src)] for src in present_src]
    for src, dst in zip(present_src, corresponding_dst):
        img_mod[img_mod == src] = dst

    return img_mod


def remove_overlap(zarr_path, group, footer="_olr"):

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    orignal_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, orignal_chunks)}

    n_chunks = [len(chunk) for chunk in xar.chunks]

    # get max values of every chunks
    ref = xar.coarsen({'y': chunk_dict['y'], 'x': chunk_dict['x']}).max()
    ref = ref.stack(z=('y', 'x'))
    ref = ref.shift(z=1).fillna(0).astype(int)
    ref = ref.cumsum('z').unstack('z')
    ref = ref.chunk({'y': n_chunks[0], 'x': n_chunks[1]})
    ref = ref.values

    dar_img = da.from_zarr(zarr_path, component=group + "/0/data")
    n_y, n_x = dar_img.shape

    with ProgressBar():
        dar_res = da.map_blocks(
            _apply_offset, dar_img, ref, dtype=np.float32,
            chunks=(chunk_dict["y"], chunk_dict["x"]))

        dims = ["y", "x"]
        coords = {
            "y": range(n_y),
            "x": range(n_x)}
        chunks = {"y": chunk_dict["y"], "x": chunk_dict["x"]}

        out = xr.DataArray(dar_res, dims=dims, coords=coords)
        out = out.chunk(chunks=chunks)

    # make removal values
    dfs = []
    next_pos_x = np.array(xar.chunks[1]).cumsum()[:-1]
    curr_pos_x = next_pos_x - 1
    for c, n in tqdm(zip(curr_pos_x, next_pos_x), total=len(next_pos_x)):
        xar_chunk = out.sel(x=slice(c, n)).values
        # remove rows where at least one of the two columns of xar_boder is 0
        xar_chunk = xar_chunk[xar_chunk[:, 0] * xar_chunk[:, 1] != 0]
        df = pd.DataFrame(xar_chunk)
        df["count"] = 1
        df = df.groupby([0, 1]).count().reset_index()
        # If the values in the second column are the same,
        # keep only the one with the larger count.
        df = df.sort_values(by=[1, "count"], ascending=[True, False])
        df = df.drop_duplicates(subset=[1], keep='first')
        dfs.append(df)

    next_pos_y = np.array(xar.chunks[0]).cumsum()[:-1]
    curr_pos_y = next_pos_y - 1
    for c, n in tqdm(zip(curr_pos_y, next_pos_y), total=len(next_pos_y)):
        xar_chunk = out.sel(y=slice(c, n)).values
        xar_chunk = xar_chunk.T
        xar_chunk = xar_chunk[xar_chunk[:, 0] * xar_chunk[:, 1] != 0]
        df = pd.DataFrame(xar_chunk)
        df["count"] = 1
        df = df.groupby([0, 1]).count().reset_index()
        df = df.sort_values(by=[1, "count"], ascending=[True, False])
        df = df.drop_duplicates(subset=[1], keep='first')
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sort_values(by=[1, "count"], ascending=[True, False])
    df = df.drop_duplicates(subset=[1], keep='first')

    df.columns = ["dst", "src", "count"]
    df = df[["src", "dst"]].reset_index(drop=True)

    src = df["src"].values
    dst = df["dst"].values

    # apply the removal values
    with ProgressBar():
        dar_res = da.map_blocks(
            _change_value, out.data, src, dst, dtype=np.uint32,
            chunks=(chunk_dict["y"], chunk_dict["x"]))

        dimps = ["y", "x"]
        coords = {
            "y": out.y.values,
            "x": out.x.values}
        chunks = {"y": chunk_dict["y"], "x": chunk_dict["x"]}

        out2 = xr.DataArray(dar_res, dims=dimps, coords=coords)
        out2 = out2.to_dataset(name="data")
        out2 = out2.chunk(chunks=chunks)
        out2.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def select_slice(zarr_path, group_name, dim, position, chunk_dict,
                 footer="sel"):

    ds = xr.open_zarr(zarr_path, group=group_name + "/0")

    if chunk_dict is None:

        original_chunks = ds["data"].chunks
        chunk_dict = {dim_name: chunk[0]
                      for dim_name, chunk in zip(ds.dims, original_chunks)}
        chunk_dict.pop(dim)
    res = ds["data"].isel({dim: position})

    res = res.drop_vars(dim)

    res = res.chunk(chunks=chunk_dict)
    res.to_zarr(zarr_path, group=group_name + footer + "/0", mode="w")

# ================ Segmentation ================


def _grow_voronoi_tree(img, max_distance):
    labels = np.unique(img[img > 0])

    if labels.size == 0:
        return img

    # Get the coordinates of the labeled pixels
    points = []
    point_labels = []
    for label in labels:
        coords = np.array(np.where(img == label)).T
        points.extend(coords)
        point_labels.extend([label] * len(coords))

    points = np.array(points)
    point_labels = np.array(point_labels)

    # Create KDTree for quick nearest-neighbor lookup
    tree = cKDTree(points)

    # Generate grid of coordinates
    xx, yy = np.meshgrid(
        np.arange(img.shape[1]), np.arange(img.shape[0]))
    coords_grid = np.column_stack([yy.ravel(), xx.ravel()])

    # Query nearest Voronoi site for each coordinate
    dists, idx = tree.query(coords_grid)

    # Mask the distances greater than max_distance
    within_distance = dists <= max_distance
    voronoi_labels = np.zeros(img.shape, dtype=img.dtype)
    voronoi_labels.ravel()[
        within_distance] = point_labels[idx[within_distance]]

    return voronoi_labels


def grow_voronoi(zarr_path, group, depth, max_distance, footer="_vor"):

    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    res = dar.map_overlap(_grow_voronoi_tree, depth=depth,
                          max_distance=max_distance, dtype=dar.dtype)

    with ProgressBar():
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": dar.chunks[0][0], "x": dar.chunks[1][0]}

        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def _gaussian_filter(img, sigma):
    img = img.astype(np.float32)
    return gaussian_filter(cp.asarray(img), sigma).get()


def gaussian_blur(zarr_path, group, sigma, footer="_gbr"):
    dar = da.from_zarr(zarr_path, component=group + "/0/data")

    res = dar.map_blocks(_gaussian_filter, sigma, dtype=dar.dtype)

    with ProgressBar():
        dims = ["y", "x"]
        coords = {"y": range(dar.shape[0]),
                  "x": range(dar.shape[1])}
        chunks = {"y": dar.chunks[0][0], "x": dar.chunks[1][0]}

        out = xr.DataArray(res, dims=dims, coords=coords)
        out = out.to_dataset(name="data")
        out = out.chunk(chunks=chunks)
        out.to_zarr(zarr_path, mode="w", group=group + footer + "/0")


def binalize(zarr_path, group, threshold, footer="_bin"):
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    with ProgressBar():
        res = xar > threshold
        res = res.chunk(chunks=chunk_dict)
        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def masking(zarr_path, group_target, group_mask, reverse=False, footer="_msk"):

    ds = xr.open_zarr(zarr_path, group=group_target + "/0")
    xar_tgt = ds["data"]

    ds = xr.open_zarr(zarr_path, group=group_mask + "/0")
    xar_msk = ds["data"]

    original_chunks = xar_tgt.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_tgt.dims, original_chunks)}

    with ProgressBar():
        if reverse:
            res = xar_tgt * (1 - (xar_msk > 0))
        else:
            res = xar_tgt * (xar_msk > 0)
        res = res.chunk(chunks=chunk_dict)
        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group_target + footer + "/0", mode="w")


if USE_GPU:
    def _fill_holes(img):
        img = img.astype(np.float32)
        frm = cp.asarray(img.values)

        out = cp.zeros(frm.shape, dtype=frm.dtype)

        ids = cp.unique(frm)
        for i in ids:
            mask = frm == i
            result = binary_fill_holes(mask).astype(frm.dtype)
            out += result * i

        res_array = xr.DataArray(
            out.get(), dims=img.dims, coords=img.coords)

        return res_array
else:
    def _fill_holes(img):
        img = img.astype(np.float32)
        frm = img.values

        out = np.zeros(frm.shape, dtype=frm.dtype)

        ids = np.unique(frm)
        for i in ids:
            mask = frm == i
            result = binary_fill_holes(mask).astype(frm.dtype)
            out += result * i

        res_array = xr.DataArray(
            out, dims=img.dims, coords=img.coords)

        return res_array


def fill_holes(zarr_path, group, footer="_fil"):

    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype=xar.dtype),
        dims=xar.dims, coords=xar.coords)

    with ProgressBar():
        dil = xar.map_blocks(
            _fill_holes, template=template)

        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


if USE_GPU:
    def _remove_edge_mask(img):
        img = img.astype(np.float32)
        frm = cp.asarray(img.values)

        # get the edge of the image
        edge_mask = cp.zeros(frm.shape, dtype=frm.dtype)
        edge_mask[0, :] = 1
        edge_mask[-1, :] = 1
        edge_mask[:, 0] = 1
        edge_mask[:, -1] = 1

        edges = frm * edge_mask
        ids_edge = cp.unique(edges)
        for i in ids_edge:
            mask = frm == i
            frm[mask] = 0

        res_array = xr.DataArray(
            frm.get(), dims=img.dims, coords=img.coords)

        return res_array
else:
    def _remove_edge_mask(img):
        img = img.astype(np.float32)
        frm = img.values

        # get the edge of the image
        edge_mask = np.zeros(frm.shape, dtype=frm.dtype)
        edge_mask[0, :] = 1
        edge_mask[-1, :] = 1
        edge_mask[:, 0] = 1
        edge_mask[:, -1] = 1

        edges = frm * edge_mask
        ids_edge = np.unique(edges)
        for i in ids_edge:
            mask = frm == i
            frm[mask] = 0

        res_array = xr.DataArray(
            frm, dims=img.dims, coords=img.coords)

        return res_array


def remove_edge_mask(zarr_path, group, footer="_egr"):

    ds = xr.open_zarr(zarr_path, group=group + "/0")
    xar = ds["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype=xar.dtype),
        dims=xar.dims, coords=xar.coords)

    with ProgressBar():
        dil = xar.map_blocks(
            _remove_edge_mask, template=template)

        original_chunks = xar.chunks
        dil = dil.chunk(original_chunks)
        ds = dil.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


# ================ Counting ================


def _count_spot(dar_spt, dar_lbl, csv_dir, block_info=None):

    cycle = block_info[0]["chunk-location"][0]
    chunk_y = block_info[0]["chunk-location"][1]
    chunk_x = block_info[0]["chunk-location"][2]

    if USE_GPU:
        label = cp.asarray(dar_lbl[:, :])
        spot = cp.asarray(dar_spt[0, :, :])
    else:
        label = dar_lbl[:, :]
        spot = dar_spt[0, :, :]
    spot_binary = spot > 0

    label_spot = label * spot_binary
    label_spot_flat = label_spot.flatten()

    if USE_GPU:
        unique, counts = cp.unique(label_spot_flat, return_counts=True)
        unique = unique[1:]
        counts = counts[1:]
        df = pd.DataFrame({"seg_id": unique.get(), "count": counts.get()})
    else:
        unique, counts = np.unique(label_spot_flat, return_counts=True)
        unique = unique[1:]
        counts = counts[1:]
        df = pd.DataFrame({"seg_id": unique, "count": counts})

    # add zero count for missing seg_id
    if USE_GPU:
        seg_id = cp.unique(label).get()
        missing_seg_id = cp.array([i for i in seg_id if i not in unique])
        missing_count = cp.zeros(len(missing_seg_id))
        df_missing = pd.DataFrame(
            {"seg_id": missing_seg_id.get(), "count": missing_count.get()})
    else:
        seg_id = np.unique(label)
        missing_seg_id = [i for i in seg_id if i not in unique]
        missing_count = np.zeros(len(missing_seg_id))
        df_missing = pd.DataFrame(
            {"seg_id": missing_seg_id, "count": missing_count})
    df = pd.concat([df, df_missing], axis=0)
    df = df.sort_values(by="seg_id")

    csv_path = os.path.join(
        csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) + ".csv")
    df.to_csv(csv_path, index=False)

    return np.zeros((1, 1, 1), dtype=np.uint8)


def count_spot(zarr_path, group_spot, group_label, footer="_cnt"):

    root = zarr.open(zarr_path)
    zar_spt = root[group_spot + "/0"]["data"]
    dar_spt = da.from_zarr(zar_spt)

    zar_lbl = root[group_label + "/0"]["data"]
    dar_lbl = da.from_zarr(zar_lbl)

    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group_spot + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    with ProgressBar():
        da.map_blocks(
            _count_spot, dar_spt, dar_lbl, csv_dir, dtype=np.uint8).compute()


def segment_info_csv(zarr_path, group, footer="_seg"):
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    root = zarr.open(zarr_path)
    zar = root[group + "/0"]["data"]
    dar = da.from_zarr(zar)

    def _segment_info_csv(dar, csv_dir, block_info=None):
        chunk_y = block_info[0]["chunk-location"][0]
        chunk_x = block_info[0]["chunk-location"][1]

        label = dar[:, :]
        label_ids = np.unique(label)
        label_ids = label_ids[label_ids > 0]
        if len(label_ids) == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        area = np.zeros(len(label_ids))
        centers = np.zeros((len(label_ids), 2))
        for i, label_id in enumerate(label_ids):
            area[i] = (label == label_id).sum()
            centers[i] = centroid(label == label_id)

        df = pd.DataFrame({
            "seg_id": label_ids,
            "area": area,
            "centroid_y": centers[:, 0],
            "centroid_x": centers[:, 1]})

        csv_path = os.path.join(
            csv_dir, str(chunk_y) + "_" + str(chunk_x) + ".csv")
        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1), dtype=np.uint8)

    with ProgressBar():
        da.map_blocks(
            _segment_info_csv, dar, csv_dir, dtype=np.uint8).compute()


def merge_count_csv(
        zarr_path, group, column_names=["seg_id", "count"],
        sort_values=["cycle", "seg_id"]):
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group, "0")
    csv_files = os.listdir(csv_dir)
    dfs = []
    for csv_file in tqdm(csv_files):
        cycle, chunk_y, chunk_x = csv_file.replace(".csv", "").split("_")
        cycle = int(cycle)
        chunk_y = int(chunk_y)
        chunk_x = int(chunk_x)

        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        df["cycle"] = cycle
        df["chunk_y"] = chunk_y
        df["chunk_x"] = chunk_x

        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    df = df[['cycle', 'chunk_y', 'chunk_x'] + column_names]
    df = df.sort_values(sort_values)

    csv_path = zarr_path.replace(".zarr", "_" + group + ".csv")
    df.to_csv(csv_path, index=False)


def merge_segment_info_csv(
        zarr_path, group, column_names=[
            "seg_id", "area", "centroid_y", "centroid_x"],
        sort_values=["seg_id"]):
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group, "0")
    csv_files = os.listdir(csv_dir)
    dfs = []
    for csv_file in tqdm(csv_files):
        chunk_y, chunk_x = csv_file.replace(".csv", "").split("_")
        chunk_y = int(chunk_y)
        chunk_x = int(chunk_x)

        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        df["chunk_y"] = chunk_y
        df["chunk_x"] = chunk_x
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    df = df[['chunk_y', 'chunk_x'] + column_names]
    df = df.sort_values(sort_values)

    csv_path = zarr_path.replace(".zarr", "_" + group + ".csv")
    df.to_csv(csv_path, index=False)


def spot_coords_csv(zarr_path, group, footer="_spt"):

    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    root = zarr.open(zarr_path)
    zar = root[group + "/0"]["data"]
    dar = da.from_zarr(zar)

    def _spot_coords_csv(dar, csv_dir, block_info=None):
        cycle = block_info[0]["chunk-location"][0]
        chunk_y = block_info[0]["chunk-location"][1]
        chunk_x = block_info[0]["chunk-location"][2]

        img = dar[0, :, :]
        y, x = np.where(img > 0)
        z = np.zeros_like(y)
        df = pd.DataFrame({"y": y, "x": x, "z": z})
        df["y"] = df["y"] + 0.5
        df["x"] = df["x"] + 0.5

        if len(df) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        csv_path = os.path.join(
            csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) + ".csv")
        df.to_csv(csv_path, index=False)

        return np.zeros((1, 1, 1), dtype=np.uint8)

    with ProgressBar():
        da.map_blocks(
            _spot_coords_csv, dar, csv_dir, dtype=np.uint8).compute()


# =========== MERFISH ===============


def gaussian_kernel(shape, sigma):
    m, n = [int((ss - 1.) / 2.) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel


if USE_GPU:
    def _merfish_prefilter(img_raw, sigma_high, psf, iterations, sigma_low, mask_size):

        img = cp.asarray(img_raw.astype(np.float32))
        frm = cp.asarray(img[0, 0, :, :])

        # 1) High pass filter
        lowpass = gaussian(frm, sigma=sigma_high, output=None,
                           cval=0, preserve_range=True,
                           truncate=4.0)
        lowpass = cp.clip(lowpass, 0, None)
        highpass = cp.clip(frm - lowpass, 0, None)

        # 2) richardson_lucy_deconv
        im_deconv = 0.5 * cp.ones(frm.shape)
        psf_mirror = cp.asarray(psf[::-1, ::-1])

        eps = cp.finfo(frm.dtype).eps
        for _ in range(iterations):
            x = fftconvolve(im_deconv, cp.asarray(psf), 'same')
            cp.place(x, x == 0, eps)
            relative_blur = highpass / x + eps
            im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

        # 3) Low pass filter
        lowpass = gaussian(
            im_deconv, sigma=sigma_low, output=None,
            cval=0, preserve_range=True, truncate=4.0)

        lowpass = cp.clip(lowpass, 0, None)  # TODO
        footprint = [(cp.ones((1, 3)), mask_size),
                     (cp.ones((3, 1)), mask_size)]
        # 4) remove zero positions
        mask = frm == 0
        mask = binary_dilation(mask, footprint=footprint)
        lowpass[mask] = 0

        # 5) Normalize
        # lowpass_nonzero = lowpass[lowpass > 0]
        # norm = lowpass / np.percentile(lowpass_nonzero, 95) / 1000
        # norm = lowpass
        # img[0, 0, :, :] = norm

        img_raw[0, 0, :, :] = lowpass.get()
        return img_raw
else:
    def _merfish_prefilter(img_raw, sigma_high, psf, iterations, sigma_low, mask_size):

        img = img_raw.astype(np.float32)
        frm = img[0, 0, :, :]

        # 1) High pass filter
        lowpass = gaussian(frm, sigma=sigma_high, output=None,
                           cval=0, preserve_range=True,
                           truncate=4.0)
        lowpass = np.clip(lowpass, 0, None)
        highpass = np.clip(frm - lowpass, 0, None)

        # 2) richardson_lucy_deconv
        im_deconv = 0.5 * np.ones(frm.shape)
        psf_mirror = psf[::-1, ::-1]

        eps = np.finfo(frm.dtype).eps
        for _ in range(iterations):
            x = fftconvolve(im_deconv, psf, 'same')
            np.place(x, x == 0, eps)
            relative_blur = highpass / x + eps
            im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

        # 3) Low pass filter
        lowpass = gaussian(
            im_deconv, sigma=sigma_low, output=None,
            cval=0, preserve_range=True, truncate=4.0)

        lowpass = np.clip(lowpass, 0, None)  # TODO
        footprint = [(np.ones((1, 3)), mask_size),
                     (np.ones((3, 1)), mask_size)]
        # 4) remove zero positions
        mask = frm == 0
        mask = binary_dilation(mask, footprint=footprint)
        lowpass[mask] = 0

        # 5) Normalize
        # lowpass_nonzero = lowpass[lowpass > 0]
        # norm = lowpass / np.percentile(lowpass_nonzero, 95) / 1000
        # norm = lowpass
        # img[0, 0, :, :] = norm

        img_raw[0, 0, :, :] = lowpass
        return img_raw


def merfish_prefilter(zarr_path, group, sigma_high, psf, iterations,
                      sigma_low, mask_size, footer="_mfp"):

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    template = xr.DataArray(
        da.empty_like(xar.data, dtype="float32"),
        dims=xar.dims, coords=xar.coords)

    with ProgressBar():
        flt = xar.map_blocks(_merfish_prefilter, kwargs={
            "sigma_high": sigma_high, "psf": psf, "iterations": iterations,
            "sigma_low": sigma_low, "mask_size": mask_size}, template=template)

        flt = flt.chunk(xar.chunks)
        ds = flt.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def _norm_value(img):
    norm_order = 2
    img = img.stack(traces=("cycle", "round"))

    norm = np.linalg.norm(img.values, ord=norm_order, axis=2)

    dims = ("y", "x")
    coords = {"y": img.coords["y"], "x": img.coords["x"], }
    return xr.DataArray(norm, dims=dims, coords=coords)


def norm_value(zarr_path, group, footer="_nmv"):

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    n_round, n_cycle, n_y, n_x = xar.shape

    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    chunk_dict["round"] = n_round
    chunk_dict["cycle"] = n_cycle

    xar = xar.chunk(chunk_dict)

    new_dims = ("y", "x")
    new_coords = {
        "y": np.arange(n_y), "x": np.arange(n_x), }

    template = xr.DataArray(
        da.zeros((n_y, n_x),
                 chunks=(chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.float32),
        dims=new_dims, coords=new_coords)

    with ProgressBar():
        res = xar.map_blocks(_norm_value, template=template)
        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def divide_by_norm(zarr_path, groups, footer="_nrm"):

    ds = xr.open_zarr(zarr_path, group=groups[0] + "/0")
    xar_flt = ds["data"]
    ds = xr.open_zarr(zarr_path, group=groups[1] + "/0")
    xar_nmv = ds["data"]

    n_round, n_cycle, n_y, n_x = xar_flt.shape

    original_chunks = xar_flt.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_flt.dims, original_chunks)}

    res = xar_flt / xar_nmv

    res = res.chunk(chunk_dict)

    ds = res.to_dataset(name="data")
    ds.to_zarr(zarr_path, group=groups[0] + footer + "/0", mode="w")


def _nearest_neighbor(img, nn):

    pixel_traces = img.stack(traces=("cycle", "round"))
    pixel_traces = pixel_traces.stack(features=("y", "x"))
    pixel_traces = pixel_traces.transpose("features", "traces")

    pixel_traces = pixel_traces.values
    pixel_traces = pixel_traces.astype(np.float32)
    pixel_traces = np.nan_to_num(pixel_traces)

    metric_output, indices = nn.kneighbors(pixel_traces)

    indices = indices.reshape(img.sizes["y"], img.sizes["x"])
    metric_output = metric_output.reshape(
        img.sizes["y"], img.sizes["x"])

    res = np.zeros((2, img.sizes["y"], img.sizes["x"]))
    res[0] = indices
    res[1] = metric_output

    dims = ("iddist", "y", "x")
    coords = {"iddist": np.arange(2),
              "y": img.coords["y"], "x": img.coords["x"], }
    res = xr.DataArray(res, dims=dims, coords=coords)
    return res


def nearest_neighbor(zarr_path, group, code_intensity_path, footer="_nnd"):
    # nearest neighbor

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]

    n_round, n_cycle, n_y, n_x = xar.shape
    original_chunks = xar.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar.dims, original_chunks)}

    chunk_dict["round"] = n_round
    chunk_dict["cycle"] = n_cycle

    xar = xar.chunk(chunk_dict)

    code_intensities = xr.open_dataarray(code_intensity_path)
    code_traces = code_intensities.stack(traces=("c", "r"))
    linear_codes = code_traces.values

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto',
                          metric="euclidean").fit(linear_codes)

    new_dims = ("iddist", "y", "x")
    new_coords = {
        "iddist": np.arange(2),
        "y": np.arange(n_y),
        "x": np.arange(n_x), }

    template = xr.DataArray(
        da.zeros((2, n_y, n_x),
                 chunks=(2, chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.float32),
        dims=new_dims, coords=new_coords)

    with ProgressBar():
        res = xar.map_blocks(
            _nearest_neighbor, kwargs={"nn": nn},
            template=template)

        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


if USE_GPU:
    def _select_decode(img, min_intensity, max_distance,
                       area_limits):

        decoded_img = cp.asarray(img[0, :, :].values)
        dist_img = img[1, :, :]
        norm_img = img[2, :, :]

        label_img = label(decoded_img, connectivity=2)
        label_img[norm_img < min_intensity] = 0
        label_img[dist_img > max_distance] = 0

        props = regionprops_table(label_img, properties=("label", "area"))

        min_area, max_area = area_limits
        df = pd.DataFrame(
            {"label": props["label"].get(), "area": props["area"].get()})
        df = df[(df["area"] >= min_area) & (df["area"] <= max_area)]

        valid_labels = cp.asarray(df["label"].values)
        label_img[~cp.isin(label_img, valid_labels)] = 0

        label_img = label_img > 0
        decoded_img = decoded_img * label_img

        res = np.zeros(img.shape[1:], dtype=np.uint16)
        res = decoded_img.get()
        res = xr.DataArray(res, dims=("y", "x"),
                           coords={"y": img.coords["y"], "x": img.coords["x"]})
        return res
else:
    def _select_decode(img, min_intensity, max_distance,
                       area_limits):

        decoded_img = img[0, :, :].values
        dist_img = img[1, :, :]
        norm_img = img[2, :, :]

        label_img = label(decoded_img, connectivity=2)
        label_img[norm_img < min_intensity] = 0
        label_img[dist_img > max_distance] = 0

        props = regionprops_table(label_img, properties=("label", "area"))

        min_area, max_area = area_limits
        df = pd.DataFrame(
            {"label": props["label"], "area": props["area"]})
        df = df[(df["area"] >= min_area) & (df["area"] <= max_area)]

        valid_labels = np.asarray(df["label"].values)
        label_img[~np.isin(label_img, valid_labels)] = 0

        label_img = label_img > 0
        decoded_img = decoded_img * label_img

        res = np.zeros(img.shape[1:], dtype=np.uint16)
        res = decoded_img
        res = xr.DataArray(res, dims=("y", "x"),
                           coords={"y": img.coords["y"], "x": img.coords["x"]})
        return res


def select_decode(zarr_path, groups, min_intensity, max_distance,
                  area_limits, footer="_dec"):

    root = xr.open_zarr(zarr_path, group=groups[0] + "/0")
    xar_nmv = root["data"]
    root = xr.open_zarr(zarr_path, group=groups[1] + "/0")
    xar_nnd = root["data"]

    original_chunks = xar_nnd.chunks
    chunk_dict = {dim_name: chunk[0]
                  for dim_name, chunk in zip(xar_nnd.dims, original_chunks)}
    n_y, n_x = xar_nmv.shape

    nmv_expanded = xar_nmv.expand_dims(dim={"iddist": [2]})
    xar_in = xr.concat([xar_nnd, nmv_expanded], dim="iddist")
    xar_in = xar_in.chunk({"iddist": 3,
                           "y": chunk_dict["y"], "x": chunk_dict["x"]})

    new_dims = ("y", "x")
    new_coords = {"y": np.arange(n_y), "x": np.arange(n_x), }
    template = xr.DataArray(
        da.empty((n_y, n_x),
                 chunks=(chunk_dict["y"], chunk_dict["x"]),
                 dtype=np.uint16),
        dims=new_dims, coords=new_coords)

    with ProgressBar():
        res = xar_in.map_blocks(
            _select_decode, kwargs={
                "min_intensity": min_intensity,
                "max_distance": max_distance,
                "area_limits": area_limits},
            template=template)

        ds = res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=groups[1] + footer + "/0", mode="a")


# =========== SeqIF ===============

def TCEP_subtraction(zarr_path, group, footer="_sub"):

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    dims = xar.dims
    chunk_sizes = [xar.chunksizes[dim][0] for dim in dims]
    chunk_dict = {dim: size for dim, size in zip(dims, chunk_sizes)}

    xar = xar.astype("float32")
    xar_even = xar.sel(cycle=xar.cycle % 2 == 0)
    xar_odd = xar.sel(cycle=xar.cycle % 2 == 1)
    xar_even.coords["cycle"] = xar_even.coords["cycle"] // 2
    xar_odd.coords["cycle"] = xar_odd.coords["cycle"] // 2
    xar_sub = xar_even - xar_odd
    xar_sub = xar_sub.fillna(0)
    xar_sub = xar_sub.clip(0)
    xar_sub = xar_sub.astype("int16")

    # remove subtract by 0 in tile edge blank
    xar_sub = xar_sub * (xar_odd > 0)
    xar_sub = xar_sub.fillna(0)
    xar_sub = xar_sub.astype("uint16")

    xar_sub = xar_sub.chunk(chunk_dict)
    # chose even cycle
    with ProgressBar():

        # save to zarr
        ds = xar_sub.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def skip_odd_cycle(zarr_path, group, footer="_skc"):

    root = xr.open_zarr(zarr_path, group=group + "/0")
    xar = root["data"]
    dims = xar.dims
    chunk_sizes = [xar.chunksizes[dim][0] for dim in dims]
    chunk_dict = {dim: size for dim, size in zip(dims, chunk_sizes)}

    # chose even cycle
    with ProgressBar():
        xar_res = xar.sel(cycle=xar.cycle % 2 == 0)

        # save to zarr
        xar_res = xar_res.chunk(chunk_dict)
        ds = xar_res.to_dataset(name="data")
        ds.to_zarr(zarr_path, group=group + footer + "/0", mode="w")


def _get_intensity(dar_int, dar_lbl, csv_dir, block_info=None):

    cycle = block_info[0]["chunk-location"][0]
    chunk_y = block_info[0]["chunk-location"][1]
    chunk_x = block_info[0]["chunk-location"][2]

    label = cp.array(dar_lbl[:, :])
    value = cp.array(dar_int[0, :, :])

    label_ids = cp.unique(label)
    label_ids = label_ids[label_ids > 0]
    if len(label_ids) == 0:
        return np.zeros((1, 1, 1), dtype=np.uint8)

    area = cp.zeros(len(label_ids))
    value_sum = cp.zeros(len(label_ids))
    centers = cp.zeros((len(label_ids), 2))
    for i, label_id in enumerate(label_ids):
        area[i] = (label == label_id).sum()
        value_sum[i] = value[label == label_id].sum()
        centers[i] = centroid(label == label_id)

    df = pd.DataFrame({
        "label": label_ids.get(),
        "area": area.get(),
        "value_sum": value_sum.get(),
        "center_y": centers[:, 0].get(),
        "center_x": centers[:, 1].get()})

    csv_path = os.path.join(
        csv_dir, str(cycle) + "_" + str(chunk_y) + "_" + str(chunk_x) + ".csv")

    df.to_csv(csv_path, index=False)

    return np.zeros((1, 1, 1), dtype=np.uint8)


def get_intensity(zarr_path, group_int, group_lbl, footer="_int"):
    root = zarr.open(zarr_path)
    zar_int = root[group_int + "/0"]["data"]
    dar_int = da.from_zarr(zar_int)

    zar_lbl = root[group_lbl + "/0"]["data"]
    dar_lbl = da.from_zarr(zar_lbl)

    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group_int + footer, "0")
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)

    with ProgressBar():
        da.map_blocks(
            _get_intensity, dar_int, dar_lbl, csv_dir, dtype=np.uint8).compute()


def merge_intensity_csv(
        zarr_path, group, column_names=["label", "area", "value_sum",
                                        "center_y", "center_x"],
        sort_values=["cycle", "label"]):
    csv_dir = zarr_path.replace(".zarr", "_csv")
    csv_dir = os.path.join(csv_dir, group, "0")
    csv_files = os.listdir(csv_dir)
    dfs = []
    for csv_file in tqdm(csv_files):
        cycle, chunk_y, chunk_x = csv_file.replace(".csv", "").split("_")
        cycle = int(cycle)
        chunk_y = int(chunk_y)
        chunk_x = int(chunk_x)

        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        df["cycle"] = cycle
        df["chunk_y"] = chunk_y
        df["chunk_x"] = chunk_x

        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    df = df[['cycle', 'chunk_y', 'chunk_x'] + column_names]
    df = df.sort_values(sort_values)

    csv_path = zarr_path.replace(".zarr", "_" + group + ".csv")
    df.to_csv(csv_path, index=False)


def main():
    root_dir = "/spo82/ana/240521_simu/240807/"
    pitch = [0.1650, 0.1, 0.1]
    n_cycle, n_tile_y, n_tile_x, n_z, n_y, n_x = \
        18, 1, 1, 1, 1000, 1000

    sample_name = "Simulated_spot"
    zarr_path = os.path.join(root_dir, sample_name + ".zarr")

    st = time.time()

    # ===== Loading =====
    load_images_tif_cYXzyx(zarr_path, n_z, n_y, n_x)

    # ===== Specific processing =====
    group = "spot"
    ds = xr.open_zarr(zarr_path, group=group + "/0")
    res = ds["data"].isel({"z": 0, "tile_y": 0, "tile_x": 0})
    res = res.drop_vars(["z", "tile_y", "tile_x"])
    res = res.chunk(chunks={"cycle": 1, "y": 1000, "x": 1000})
    res.to_zarr(zarr_path, group=group + "_mip_reg" + "/0", mode="w")

    # ===== Spot detection =====
    NA = 1.4
    wavelength_um = 0.519
    mean_pitch_yx = np.mean(pitch[1:])

    group = "spot_mip_reg"
    dog_sd1, dog_sd2 = dog_sds(NA, wavelength_um, mean_pitch_yx)
    DoG_filter(zarr_path, group, dog_sd1,
               dog_sd2, axes=(1, 2), mask_radius=9)
    group = "spot_mip_reg_dog"
    footprint = local_maxima_footprint(
        NA, wavelength_um, mean_pitch_yx)
    local_maxima(zarr_path, group, footprint, axes=(1, 2))

    factor = 1.5
    # ------- selection ------
    group = "spot_mip_reg_dog_lmx"
    footer = "_ith"  # Intensity THresholding

    with ProgressBar():
        root = xr.open_zarr(zarr_path, group=group + "/0")
        xar = root["data"]
        total = xar.sum().compute()
        count = (xar != 0).sum().compute()
        ave = total / count
        sd = (xar != 0).std().compute()
        threshold = ave + factor * sd
        # convert to zero if below threshold
        res = xar.where(xar > threshold, 0)
        # save as zarr
        res.to_zarr(zarr_path, group=group +
                    footer + "/0", mode="w")

    # ===== Count =====
    group = "spot_mip_reg_dog_lmx_ith"
    spot_coords_csv(zarr_path, group)

    print("Elapsed time: ", time.time() - st)  # 3.8845205307006836 s


if __name__ == "__main__":
    main()

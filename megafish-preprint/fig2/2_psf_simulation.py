import os
import numpy as np
from scipy.ndimage import map_coordinates
from tifffile import imread, TiffWriter
from tqdm import tqdm
import napari
import pandas as pd
from imaris_ims_file_reader.ims import ims


root_dir = "/spo82/ana/240521_simu/240807/"


def rescale_psf(source_image, output_shape):
    """
    Resize a 3D source image to the specified output shape using interpolation.

    Args:
        source_image (numpy.ndarray): The 3D source image.
        output_shape (tuple): The desired output shape (depth, height, width).

    Returns:
        numpy.ndarray: The resized 3D image.
    """
    input_shape = source_image.shape
    depth_indices = np.linspace(0, input_shape[0] - 1, output_shape[0])
    row_indices = np.linspace(0, input_shape[1] - 1, output_shape[1])
    col_indices = np.linspace(0, input_shape[2] - 1, output_shape[2])

    # Create a meshgrid of the indices
    depth_grid, row_grid, col_grid = np.meshgrid(
        depth_indices, row_indices, col_indices, indexing='ij')

    # Interpolate the source image at the calculated grid points
    coords = np.array([depth_grid.ravel(), row_grid.ravel(), col_grid.ravel()])
    resized_image = map_coordinates(
        source_image, coords, order=1).reshape(output_shape)

    return resized_image


def rescale_3d(psf, scale_factor_xy, scale_factor_z):
    """
    Rescales a 3D image using block averaging.

    Args:
    psf (np.ndarray): The 3D array to be rescaled.
    scale_factor (float): The scale factor for rescaling.

    Returns:
    np.ndarray: The rescaled 3D array.
    """
    new_shape = (int(psf.shape[0] * scale_factor_z),
                 int(psf.shape[1] * scale_factor_xy),
                 int(psf.shape[2] * scale_factor_xy))
    psf_rescaled = psf.reshape(
        new_shape[0], psf.shape[0] // new_shape[0],
        new_shape[1], psf.shape[1] // new_shape[1],
        new_shape[2], psf.shape[2] // new_shape[2]).sum(axis=(1, 3, 5))
    return psf_rescaled


def plot_psf(img, location, psf, scale_factor_xy, scale_factor_z, intenisity=1):

    # Calculate integer coordinates and decimal parts
    z_c = int(np.floor(location[0]))
    y_c = int(np.floor(location[1]))
    x_c = int(np.floor(location[2]))
    d_z = int((location[0] - z_c) * 10)
    d_y = int((location[1] - y_c) * 10)
    d_x = int((location[2] - x_c) * 10)

    # Shift the PSF image
    psf_shifted = np.zeros(psf.shape)
    psf_shifted[d_z:psf.shape[0], d_y:psf.shape[1], d_x:psf.shape[2]
                ] = psf[
                    0:psf.shape[0] - d_z,
                    0:psf.shape[1] - d_y,
                    0:psf.shape[2] - d_x]

    psf_rescaled = rescale_3d(psf_shifted, scale_factor_xy, scale_factor_z)

    psf_rescaled_norm = psf_rescaled / psf_rescaled.max()
    psf_rescaled = psf_rescaled_norm * intenisity

    # Add the PSF image to the main image
    rescaled_size = psf_rescaled.shape
    w_z = rescaled_size[0] // 2
    w_y = rescaled_size[1] // 2
    w_x = rescaled_size[2] // 2
    img[z_c - w_z:z_c + w_z,
        y_c - w_y:y_c + w_y,
        x_c - w_x:x_c + w_x] += psf_rescaled

    return img


def main_1():
    # check spots

    # Define parameters
    img_size = (30, 1000, 1000)  # (Z, Y, X)
    psf_even_size = (200, 200, 200)  # Make sure it's even
    scale_factor_xy = 0.1
    scale_factor_z = 0.05

    # Create an empty image
    img = np.zeros(img_size, dtype=np.float32)

    # Load the PSF image
    psf_path = '/spo82/ana/240521_simu/240807/psf_ex488_em519_na1400_ri1401_au1_10nmpix/empsf_vol.tif'
    psf = np.array(imread(psf_path), dtype=np.float32)
    psf_even = rescale_psf(psf, psf_even_size)

    for density in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

        range_min_xy = 10
        range_max_xy = 990
        range_min_z = 10
        range_max_z = 11
        decimal_places = 1

        # Density information
        pitch = 0.1
        area = ((range_max_xy - range_min_xy) * pitch) ** 2  # um^2
        num_sets = int(density * area)  # spots/um^2
        real_density = num_sets / area  # spots/um^2

        print(f"Number of spots: {num_sets}")
        # print(f"Area: {area} um^2")
        # print(f"Density: {density} spots/um^2")
        print(f"Real density: {real_density} spots/um^2")

    [96, 288, 960, 2881, 9604, 28812]

    random_sets = []
    for _ in range(num_sets):
        random_xy = np.random.uniform(range_min_xy, range_max_xy, 2)
        random_z = np.random.uniform(range_min_z, range_max_z, 1)
        random_set = np.concatenate((random_z, random_xy), axis=None)
        random_set = np.round(random_set, decimal_places)
        random_sets.append(random_set.tolist())

    for target_zyx in tqdm(random_sets):
        img = plot_psf(img, target_zyx, psf_even,
                       scale_factor_xy, scale_factor_z, intenisity=80)

    img += np.random.normal(100, 5, img.shape)

    # show in napari
    scale = (0.2, 0.1, 0.1)
    viewer = napari.Viewer()
    viewer.add_image(img, scale=scale, name="spot", colormap="gray",
                     blending="additive", contrast_limits=[80, 200])
    napari.run()


def main_2():
    # make z,y,x coordinates csv file
    n_spots = [96, 288, 960, 2881, 9604, 28812]
    n_repeat = 3

    range_min_xy = 10
    range_max_xy = 990
    range_min_z = 10
    range_max_z = 11
    decimal_places = 1

    pitch = 0.1

    np.random.seed(0)
    for n_spot in n_spots:

        # Density information
        area = ((range_max_xy - range_min_xy) * pitch) ** 2  # um^2
        density = n_spot / area  # spots/um^2
        print(f"Number of spots: {n_spot}")
        print(f"Area: {area} um^2")
        print(f"Density: {density} spots/um^2")

        for i in range(n_repeat):
            random_sets = []
            for _ in range(n_spot):
                random_xy = np.random.uniform(range_min_xy, range_max_xy, 2)
                random_z = np.random.uniform(range_min_z, range_max_z, 1)
                random_set = np.concatenate((random_z, random_xy), axis=None)
                random_set = np.round(random_set, decimal_places)
                random_sets.append(random_set.tolist())

            coord_path = root_dir + f"spot_zyx_num{n_spot}_rep{i + 1}.csv"

            # convert to DataFrame
            df = pd.DataFrame(random_sets, columns=["z", "y", "x"])
            df.to_csv(coord_path, index=False)


def main_3():
    # Calculate noise SD

    img_dir = "/spo82/ana/Microscopy/Fusion2/2024-05-02/"

    dark_file = "Fusion2_dark_200ms_2024-05-02_15.15.44.ims"

    dark_img = ims(img_dir + dark_file)[0, 0, 0, :, :]

    intensity_array = dark_img.flatten()
    noise_sd = np.std(intensity_array)
    print(f"Noise SD: {noise_sd}")
    print(f"Mean intensity: {np.mean(intensity_array)}")

    # Noise SD: 2.688652141267394
    # Mean intensity: 101.94482612609863


def main_4():
    # make simulation image

    # Define parameters
    img_size = (21, 1000, 1000)  # (Z, Y, X)
    psf_even_size = (200, 200, 200)  # Make sure it's even
    scale_factor_xy = 0.1
    scale_factor_z = 0.05

    # Create an empty image
    img = np.zeros(img_size, dtype=np.float32)

    # Load the PSF image
    psf_root = "/spo82/ana/240521_simu/240807/"
    # psf_dirs = ["psf_ex561_em592_na1400_ri1401_au1_10nmpix/",
    #             "psf_ex561_em592_na1400_ri1401_wf_10nmpix/",
    #             "psf_ex637_em671_na1400_ri1401_au1_10nmpix/",
    #             "psf_ex637_em671_na1400_ri1401_wf_10nmpix/"]

    psf_dirs = ["psf_ex488_em519_na1400_ri1401_au1_10nmpix/",]

    for psf_dir in psf_dirs:
        psf_path = psf_root + psf_dir + "obsvol_vol.tif"

        psf = np.array(imread(psf_path), dtype=np.float32)
        psf_even = rescale_psf(psf, psf_even_size)

        n_spots = [96, 288, 960, 2881, 9604, 28812]
        n_repeat = 3

        range_min_xy = 10
        range_max_xy = 990
        range_min_z = 10
        range_max_z = 11
        decimal_places = 1

        pitch = 0.1

        np.random.seed(0)
        for n_spot in tqdm(n_spots, desc=psf_dir):

            # Density information
            # area = ((range_max_xy - range_min_xy) * pitch) ** 2  # um^2
            # density = n_spot / area  # spots/um^2
            # print(f"Number of spots: {n_spot}")
            # print(f"Area: {area} um^2")
            # print(f"Density: {density} spots/um^2")

            for i in tqdm(range(n_repeat), desc=f"num{n_spot}", leave=False):

                random_sets_path = root_dir + \
                    f"spot_zyx_num{n_spot}_rep{i + 1}.csv"

                random_sets = pd.read_csv(random_sets_path).values.tolist()

                # Create an empty image
                img = np.zeros(img_size, dtype=np.float32)

                for target_zyx in tqdm(random_sets, desc=f"rep{i + 1}", leave=False):
                    img = plot_psf(img, target_zyx, psf_even,
                                   scale_factor_xy, scale_factor_z, intenisity=80)

                img += np.random.normal(102, 2.689, img.shape)

                save_dir = root_dir + psf_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + \
                    f"simu_img_num{n_spot}_rep{i + 1}.tif"

                # save image
                with TiffWriter(save_path) as tif:
                    tif.save(img)


def main_5():
    # make MIP image

    # Define parameters
    img_size = (21, 1000, 1000)  # (Z, Y, X)
    psf_even_size = (200, 200, 200)  # Make sure it's even
    scale_factor_xy = 0.1
    scale_factor_z = 0.05

    # Create an empty image
    img = np.zeros(img_size, dtype=np.float32)

    # Load the PSF image
    psf_root = "/spo82/ana/240521_simu/240807/"
    psf_dirs = ["psf_ex488_em519_na1400_ri1401_au1_10nmpix/",]

    for psf_dir in psf_dirs:
        psf_path = psf_root + psf_dir + "obsvol_vol.tif"

        n_spots = [96, 288, 960, 2881, 9604, 28812]
        n_repeat = 3
        for n_spot in tqdm(n_spots, desc=psf_dir):
            for i in tqdm(range(n_repeat), desc=f"num{n_spot}", leave=False):
                save_dir = root_dir + psf_dir
                save_path = save_dir + \
                    f"simu_img_num{n_spot}_rep{i + 1}.tif"
                img = imread(save_path)
                mip = np.max(img, axis=0)
                save_path = save_dir + \
                    f"simu_img_num{n_spot}_rep{i + 1}_mip.tif"
                with TiffWriter(save_path) as tif:
                    tif.save(mip)


if __name__ == '__main__':
    # Uncomment only the process you want to execute

    main_1()
    # main_2()
    # main_3()
    # main_4()
    # main_5()

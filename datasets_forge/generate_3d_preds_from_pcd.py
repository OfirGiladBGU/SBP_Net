import os
import pathlib
from tqdm import tqdm
import numpy as np
import random
from scipy.ndimage import convolve, label, rotate
import math
import shutil
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from configs.configs_parser import DATA_PATH
from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file, get_data_file_stem
from datasets_forge.generate_holes_utils import Data_Type, generate_plane_holes_v6
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


DATASET_PATH = DATA_PATH.joinpath("PipeForge3DPCD")


###################
# Generate Labels #
###################
def expand_connected_neighbors(array: np.ndarray) -> np.ndarray:
    """
    Expands the value of `1` to all 6-connected neighbors in a 3D numpy array.
    Args:
        array (np.ndarray): A 3D numpy array with binary values (0 and 1).

    Returns:
        np.ndarray: A 3D numpy array with the neighbors of all `1` voxels set to `1`.
    """
    # V1: Very slow method
    # Get the shape of the array
    # x_max, y_max, z_max = array.shape
    #
    # # Create a copy of the array to store results
    # expanded_array = array.copy()
    #
    # # Iterate over the array to find all 1's and expand to their 6 neighbors
    # for x in range(x_max):
    #     for y in range(y_max):
    #         for z in range(z_max):
    #             if array[x, y, z] == 1:
    #                 # Update neighbors in 6 directions
    #                 if x > 0: expanded_array[x - 1, y, z] = 1
    #                 if x < x_max - 1: expanded_array[x + 1, y, z] = 1
    #                 if y > 0: expanded_array[x, y - 1, z] = 1
    #                 if y < y_max - 1: expanded_array[x, y + 1, z] = 1
    #                 if z > 0: expanded_array[x, y, z - 1] = 1
    #                 if z < z_max - 1: expanded_array[x, y, z + 1] = 1

    # V2: Faster method
    # Define a kernel for 6-connected neighbors
    kernel = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])

    # Apply the convolution
    convolved = convolve(array, kernel, mode='constant', cval=0)

    # Threshold the result to binary (0 or 1)
    expanded_array = (convolved > 0).astype(np.uint8)

    return expanded_array


# Create new 'labels' folder with numpy data
def convert_originals_data_to_labels_data(save_as_npy: bool = False, points_scale: float = 1.0, voxel_size: float = 1.0,
                                          increase_density: bool = False):
    """
    Converts the original data to discrete data for numpy array, and then save the result in labels folder.
    """
    input_folder = os.path.join(DATASET_PATH, "originals")
    output_folder = os.path.join(DATASET_PATH, "labels")

    # os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]

        numpy_data = convert_data_file_to_numpy(
            data_filepath=data_filepath,
            points_scale=points_scale,
            voxel_size=voxel_size
        )

        if increase_density is True:
            numpy_data = expand_connected_neighbors(array=numpy_data)

        # Save data:
        data_filepath_stem = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folder)
        save_filename = os.path.join(output_folder, data_filepath_stem)

        if save_as_npy is True:
            data_filepath = f"{data_filepath}.npy"
        convert_numpy_to_data_file(
            numpy_data=numpy_data,
            source_data_filepath=data_filepath,
            save_filename=save_filename,
            points_scale=points_scale,
            voxel_size=voxel_size,
            apply_data_threshold=True
        )


# Create new 'preds' folder with holes in numpy data
def convert_labels_data_to_preds_data(save_as_npy: bool = False, config: dict = None):
    """
    Converts the labels data to preds data with holes in numpy array, and then save the result in preds folder.
    :param save_as_npy:
    :param config:
    :return:
    """
    input_folder = os.path.join(DATASET_PATH, "labels")
    output_folder = os.path.join(DATASET_PATH, "preds")

    # os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]

        numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)

        # Generate holes:
        # TODO: implement (Use different method)
        # numpy_data = generate_sphere_holes(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v1(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v2(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v4(numpy_data=numpy_data, config=config)

        # Selected method for final version:
        # NOTE: V5 is too hard for PCD (holes do not overlap)
        numpy_data = generate_plane_holes_v6(numpy_data=numpy_data, config=config)

        # Save data:
        data_filepath_stem = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folder)
        save_filename = os.path.join(output_folder, data_filepath_stem)

        if save_as_npy is True:
            data_filepath = f"{data_filepath}.npy"
        convert_numpy_to_data_file(
            numpy_data=numpy_data,
            source_data_filepath=data_filepath,
            save_filename=save_filename,
            apply_data_threshold=True
        )


def clone_preds_as_evals():
    input_folder = os.path.join(DATASET_PATH, "preds")
    output_folder = os.path.join(DATASET_PATH, "evals")

    shutil.copytree(src=input_folder, dst=output_folder, dirs_exist_ok=True)


def main():
    # From PCD to Numpy with option to go back
    points_scale = 0.25
    voxel_size = 1.0
    increase_density = False

    # # Hospital CUP scale
    # points_scale = 25.0
    # voxel_size = 1.0
    # increase_density = False

    # Generator configuration:
    config = dict(
        data_type=Data_Type.PCD,
        num_of_centers=10,  # Control the number of holes by number of centers
        plane_thickness=[2, 3],  # Control the hole size by plane thickness
        cube_size=[5, 10]  # Control the hole size by cube size
    )

    convert_originals_data_to_labels_data(save_as_npy=True, points_scale=points_scale, voxel_size=voxel_size,
                                          increase_density=increase_density)
    convert_labels_data_to_preds_data(save_as_npy=True, config=config)
    # clone_preds_as_evals()


if __name__ == '__main__':
    main()

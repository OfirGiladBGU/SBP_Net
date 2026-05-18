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
from datasets_forge.generate_holes_utils import Data_Type, generate_plane_holes_v5
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


DATASET_PATH = DATA_PATH.joinpath("PipeForge3DMesh")


###################
# Generate Labels #
###################
# Create new 'labels' folder with numpy data
def convert_originals_data_to_labels_data(save_as_npy: bool = False, mesh_scale: float = 0.5, voxel_size: float = 2.0):
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
            mesh_scale=mesh_scale,
            voxel_size=voxel_size
        )

        # Save data:
        data_filepath_stem = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folder)
        save_filename = os.path.join(output_folder, data_filepath_stem)

        if save_as_npy is True:
            data_filepath = f"{data_filepath}.npy"
        convert_numpy_to_data_file(
            numpy_data=numpy_data,
            source_data_filepath=data_filepath,
            save_filename=save_filename,
            mesh_scale=mesh_scale,
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
        # numpy_data = generate_box_holes(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v1(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v2(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v3(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v4(numpy_data=numpy_data, config=config)
        # numpy_data = generate_plane_holes_v6(numpy_data=numpy_data, config=config)

        # Selected method for final version:
        # NOTE: V6 is too weak assumption for Mesh (holes may overlap)
        numpy_data = generate_plane_holes_v5(numpy_data=numpy_data, config=config)

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
    # From Mesh to Numpy without option to go back
    # mesh_scale = 0.5
    # voxel_size = 2.0

    mesh_scale = 0.25  # Size: ~150
    # mesh_scale = 0.35  # Size: ~200
    # mesh_scale = 0.50  # Size: ~300
    # mesh_scale = 0.70  # Size: ~400
    voxel_size = 1.0

    # Generator configuration:
    config = dict(
        data_type=Data_Type.MESH,
        num_of_centers=10,  # Control the number of holes by number of centers
        plane_thickness=[1, 2],  # Control the hole size by plane thickness
        cube_size=[5, 10]  # Control the hole size by cube size
    )

    convert_originals_data_to_labels_data(save_as_npy=True, mesh_scale=mesh_scale, voxel_size=voxel_size)
    convert_labels_data_to_preds_data(save_as_npy=True, config=config)
    # clone_preds_as_evals()


if __name__ == '__main__':
    main()

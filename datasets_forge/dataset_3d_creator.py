import os
import pathlib
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from configs.configs_parser import *
from datasets.dataset_utils import *

# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


############################
# Repair 3D reconstruction #
############################
# from scipy.ndimage import binary_erosion, binary_dilation
#
#
# def refine_voxel_grid(voxel_grid, erosion_iterations=0, dilation_iterations=1):
#     refined_voxel_grid = voxel_grid
#
#     # Apply binary erosion
#     # refined_voxel_grid = binary_erosion(refined_voxel_grid, iterations=erosion_iterations)
#
#     # Apply binary dilation
#     refined_voxel_grid = binary_dilation(refined_voxel_grid, iterations=dilation_iterations)
#     refined_voxel_grid = refined_voxel_grid.astype(np.uint8)
#
#     return refined_voxel_grid
#
#
# def refine_construction(voxel_grid: np.ndarray):
#     # Extract surface mesh using Marching Cubes
#     voxel_grid_refined = refine_voxel_grid(voxel_grid)
#
#     convert_numpy_to_nii_gz(voxel_grid_refined, save_name="Test")


def create_3d_reconstructions(include_debug: bool = False):
    # Sources
    source_folders = {
        "labels_3d": LABELS_3D
    }

    # Inputs
    input_folders = {
        "labels_2d": LABELS_2D
    }

    # Outputs
    output_folders = {
        "labels_3d_reconstruct": LABELS_3D_RECONSTRUCT
    }

    if include_debug:
        source_folders.update({
            "preds_3d": PREDS_3D,
            "preds_fixed_3d": PREDS_FIXED_3D,
            "preds_advanced_fixed_3d": PREDS_ADVANCED_FIXED_3D
        })

        input_folders.update({
            "preds_2d": PREDS_2D,
            "preds_fixed_2d": PREDS_FIXED_2D,
            "preds_advanced_fixed_2d": PREDS_ADVANCED_FIXED_2D
        })

        output_folders.update({
            "preds_3d_reconstruct": PREDS_3D_RECONSTRUCT,
            "preds_fixed_3d_reconstruct": PREDS_FIXED_3D_RECONSTRUCT,
            "preds_advanced_fixed_3d_reconstruct": PREDS_ADVANCED_FIXED_3D_RECONSTRUCT
        })

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    filepaths_found = list()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.png"))
        filepaths_found.append(len(input_filepaths[key]))

    # Validation
    if len(set(filepaths_found)) != 1:
        raise ValueError("Different number of files found in the Input folders")

    log_data = pd.read_csv(TRAIN_LOG_PATH)

    image_format_map = {
        "labels_2d_format": list(),
        "preds_2d_format": list(),
        "preds_fixed_2d_format": list(),
        "preds_advanced_fixed_2d_format": list()
    }

    # Using iloc to select the first column
    for data_basename in tqdm(log_data.iloc[:, 0]):
        for key, value in input_folders.items():
            data_image_format = os.path.join(value, f"{data_basename}_<VIEW>.png")
            image_format_map[f"{key}_format"].append(data_image_format)

    filepaths_count = len(image_format_map["labels_2d_format"])
    for filepath_idx in tqdm(range(filepaths_count)):
        for key, value in input_folders.items():
            # Get index data
            data_filepath = value
            data_image_format = image_format_map[f"{key}_format"][filepath_idx]

            # Output path
            data_image_relative = pathlib.Path(data_image_format).relative_to(data_filepath)
            key_3d = key.replace("2d", "3d")
            data_image_output_filepath = os.path.join(output_folders[f"{key_3d}_reconstruct"], data_image_relative)

            # 3D Folders
            save_filename = data_image_output_filepath.replace("_<VIEW>.png", "")
            data_3d_path = pathlib.Path(source_folders[key_3d])
            data_3d_relative = str(data_image_relative).replace("_<VIEW>.png", "*")
            source_data_3d_filepath = list(data_3d_path.rglob(data_3d_relative))[0]
            numpy_3d_data = reconstruct_3d_from_2d(
                format_of_2d_images=data_image_format,
                source_data_filepath=source_data_3d_filepath
            )
            convert_numpy_to_data_file(
                numpy_data=numpy_3d_data,
                source_data_filepath=source_data_3d_filepath,
                save_filename=save_filename,
                apply_data_threshold=True
            )

    # format_of_2d_images = r".\parse_labels_mini_cropped_v5\PA000005_vessel_02584_<VIEW>.png"
    # final_data_3d = reconstruct_3d_from_2d(format_of_2d_images)
    #
    # voxel_grid = final_data_3d.astype(np.uint8)
    # refine_construction(voxel_grid)


def create_3d_fusions():
    # Inputs
    input_folders = {
        "labels_3d_reconstruct": LABELS_3D_RECONSTRUCT,
        "preds_3d": PREDS_3D,
        "preds_fixed_3d": PREDS_FIXED_3D,
        "preds_advanced_fixed_3d": PREDS_ADVANCED_FIXED_3D
    }

    # Outputs
    output_folders = {
        "preds_3d_fusion": PREDS_3D_FUSION,
        "preds_fixed_3d_fusion": PREDS_FIXED_3D_FUSION,
        "preds_advanced_fixed_3d_fusion": PREDS_ADVANCED_FIXED_3D_FUSION
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    filepaths_found = list()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
        filepaths_found.append(len(input_filepaths[key]))

    # Validation
    if len(set(filepaths_found)) != 1:
        raise ValueError("Different number of files found in the Input folders")

    filepaths_count = len(input_filepaths["labels_3d_reconstruct"])
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data
        label_3d_reconstruct_filepath = input_filepaths["labels_3d_reconstruct"][filepath_idx]
        label_3d_reconstruct_numpy_data = convert_data_file_to_numpy(data_filepath=label_3d_reconstruct_filepath, apply_data_threshold=True)

        # Loop over options
        for key, value in input_folders.items():
            if key == "labels_3d_reconstruct":
                continue

            # Get index data
            data_filepath = input_filepaths[key][filepath_idx]
            numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_filepath, apply_data_threshold=True)

            # Fusion
            numpy_3d_fusion = np.logical_or(numpy_3d_data, label_3d_reconstruct_numpy_data)
            numpy_3d_fusion = numpy_3d_fusion.astype(np.float32)

            # Output path
            numpy_3d_relative = pathlib.Path(data_filepath).relative_to(input_folders[key])
            numpy_3d_output_filepath = os.path.join(output_folders[f"{key}_fusion"], numpy_3d_relative)

            # 3D Folders - preds fusion
            save_filename = numpy_3d_output_filepath
            convert_numpy_to_data_file(
                numpy_data=numpy_3d_fusion,
                source_data_filepath=data_filepath,
                save_filename=save_filename,
                apply_data_threshold=True
            )


def test_2d_to_3d_and_back(data_3d_folder, data_3d_filepath, cropped_data_path):
    # Use the 2d projections to create a 3d reconstruction
    # Use the 3d reconstruction to create a 2d projection
    # Compare the new 2d projection with the original 2d projection
    data_3d_stem = get_data_file_stem(data_filepath=data_3d_filepath, relative_to=data_3d_folder)
    format_of_2d_images = os.path.join(cropped_data_path, f"{data_3d_stem}_<VIEW>.png")

    # Get the locally saved 2D images
    data_list = get_images_6_views(
        format_of_2d_images=format_of_2d_images,
        convert_to_3d=False,
        source_data_filepath=None
    )

    # Reconstruct 3D from 2D and Project 3D to 2D again
    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    data_3d = reconstruct_3d_from_2d(
        format_of_2d_images=format_of_2d_images,
        source_data_filepath=data_3d_filepath
    )
    projections = project_3d_to_2d(
        data_3d=data_3d,
        projection_options=projection_options,
        source_data_filepath=data_3d_filepath
    )

    # Compare the 2D projections with the original 2D images
    equal_status = True
    for image_idx, image_view in tqdm(enumerate(IMAGES_6_VIEWS)):
        locally_saved_image = data_list[image_idx]
        projection_image = projections[f"{image_view}_image"]
        if (locally_saved_image - projection_image).sum() != 0:
            equal_status = False
            print(f"Error in '{image_view}' view")

    if equal_status is True:
        print("All images are equal")


def main():
    include_debug = False
    create_3d_reconstructions(include_debug=include_debug)
    create_3d_fusions()

    # TODO: DEBUG
    # data_3d_folder = LABELS_3D
    # data_3d_filepath = os.path.join(data_3d_folder, "PA000005_11899.nii.gz")
    # cropped_data_path = LABELS_2D
    # test_2d_to_3d_and_back(
    #     data_3d_folder=data_3d_folder,
    #     data_3d_filepath=data_3d_filepath,
    #     cropped_data_path=cropped_data_path
    # )


if __name__ == "__main__":
    # TODO:
    # 1. Make sure DATASET_FOLDER folder is present in the root directory
    main()

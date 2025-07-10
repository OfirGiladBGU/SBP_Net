import os
import pathlib
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Tuple, List
from skimage import color
import cv2
from collections import defaultdict

from configs.configs_parser import *
from datasets.dataset_utils import *

# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


#########
# Utils #
#########
def crop_mini_cubes(data_3d: np.ndarray,
                    other_data_3d_list: List[np.ndarray] = None,
                    cube_dim: Tuple[int, int, int] = (28, 28, 28),
                    stride_dim: Tuple[int, int, int] = (14, 14, 14),
                    cubes_data: bool = False,
                    index_3d: str = ""):
    # Slower method
    # mini_cubes = list()
    # mini_cubes_data = list()
    #
    # for x in range(0, data_3d.shape[0], step):
    #     for y in range(0, data_3d.shape[1], step):
    #         for z in range(0, data_3d.shape[2], step):
    #             # print(x, y, z)
    #
    #             start_x, start_y, start_z = x, y, z
    #             end_x = min(x + size[0], data_3d.shape[0])
    #             end_y = min(y + size[1], data_3d.shape[1])
    #             end_z = min(z + size[2], data_3d.shape[2])
    #
    #             mini_cube = data_3d[start_x:end_x, start_y:end_y, start_z:end_z]
    #
    #             # Pad with zeros if the cube is smaller than the requested size
    #             pad_x = size[0] - mini_cube.shape[0]
    #             pad_y = size[1] - mini_cube.shape[1]
    #             pad_z = size[2] - mini_cube.shape[2]
    #             mini_cube = np.pad(
    #                 array=mini_cube,
    #                 pad_width=((0, pad_x), (0, pad_y), (0, pad_z)),
    #                 mode='constant',
    #                 constant_values=0
    #             )
    #
    #             # Validate the mini-cube size
    #             if mini_cube.shape != size:
    #                 raise ValueError("[BUG] Invalid Mini Cube Size")
    #
    #             mini_cubes.append(mini_cube)
    #
    #             if cubes_data is True:
    #                 mini_cubes_data.append({
    #                     "index_3d": index_3d,
    #                     "start_x": start_x,
    #                     "start_y": start_y,
    #                     "start_z": start_z,
    #                     "end_x": end_x,
    #                     "end_y": end_y,
    #                     "end_z": end_z,
    #
    #                     # Optional
    #                     "size_x": end_x - start_x,
    #                     "size_y": end_y - start_y,
    #                     "size_z": end_z - start_z
    #                 })


    # Faster method
    # Compute necessary padding to ensure all cubes fit within bounds
    def compute_padding(data_size, cube_dim_size, stride_size):
        remainder = (data_size - cube_dim_size) % stride_size
        pad_size = cube_dim_size - remainder if remainder > 0 else 0
        return tuple([0, pad_size])

    pad_x = compute_padding(data_size=data_3d.shape[0], cube_dim_size=cube_dim[0], stride_size=stride_dim[0])
    pad_y = compute_padding(data_size=data_3d.shape[1], cube_dim_size=cube_dim[1], stride_size=stride_dim[1])
    pad_z = compute_padding(data_size=data_3d.shape[2], cube_dim_size=cube_dim[2], stride_size=stride_dim[2])

    # Apply zero padding to the original array
    padded_data = np.pad(
        array=data_3d,
        pad_width=(pad_x, pad_y, pad_z),
        mode='constant',
        constant_values=0
    )

    if other_data_3d_list is not None:
        other_padded_data_list = [
            np.pad(
                array=other_data_3d,
                pad_width=(pad_x, pad_y, pad_z),
                mode='constant',
                constant_values=0
            )
            for other_data_3d in other_data_3d_list
        ]
    else:
        other_padded_data_list = []

    mini_cubes = []
    other_mini_cubes_list = [[] for _ in other_padded_data_list]
    mini_cubes_data = []

    # Iterate over the padded data with fixed steps
    # Notice: data_3d.shape[i] - cube_dim[i] + 1 is to exclude the last half mini-cube
    for x in range(0, data_3d.shape[0] - cube_dim[0] + 1, stride_dim[0]):
        for y in range(0, data_3d.shape[1] - cube_dim[1] + 1, stride_dim[1]):
            for z in range(0, data_3d.shape[2] - cube_dim[2] + 1, stride_dim[2]):
                # Crop the mini-cube
                mini_cube = padded_data[x:x + cube_dim[0], y:y + cube_dim[1], z:z + cube_dim[2]]
                mini_cubes.append(mini_cube)

                # Validate the mini-cube size
                if mini_cube.shape != cube_dim:
                    raise ValueError("[BUG] Invalid Mini Cube Size")

                for idx, other_padded_data in enumerate(other_padded_data_list):
                    # Crop the mini-cube
                    other_mini_cube = other_padded_data[x:x + cube_dim[0], y:y + cube_dim[1], z:z + cube_dim[2]]
                    other_mini_cubes_list[idx].append(other_mini_cube)

                if cubes_data is True:
                    end_x = min(x + cube_dim[0], data_3d.shape[0])
                    end_y = min(y + cube_dim[1], data_3d.shape[1])
                    end_z = min(z + cube_dim[2], data_3d.shape[2])

                    mini_cubes_data.append({
                        "index_3d": index_3d,
                        "start_x": x,
                        "start_y": y,
                        "start_z": z,
                        "end_x": end_x,
                        "end_y": end_y,
                        "end_z": end_z,

                        # Optional
                        "size_x": end_x - x,
                        "size_y": end_y - y,
                        "size_z": end_z - z
                    })

    # Prepare return values
    if cubes_data is True:
        if other_data_3d_list is None:
            return mini_cubes, mini_cubes_data
        else:
            return mini_cubes, other_mini_cubes_list, mini_cubes_data
    else:
        if other_data_3d_list is None:
            return mini_cubes
        else:
            return mini_cubes, other_mini_cubes_list


def outlier_removal_3d(pred_data: np.ndarray, label_data: np.ndarray):
    # # V1 - CONNECTED COMPONENTS
    # pred_data_fixed = np.logical_and(pred_data, label_data)

    # # V2 - CONNECTED COMPONENTS
    # outlier_data = np.maximum(pred_data - label_data, 0)
    # pred_data_fixed = pred_data - outlier_data

    # # V3 - CONNECTED COMPONENTS
    # pred_data_fixed = np.logical_and(pred_data > 0, label_data > 0)
    # pred_data_fixed = pred_data_fixed.astype(label_data.dtype)

    # # V4 - CONNECTED COMPONENTS
    pred_data_fixed = np.logical_and(pred_data, label_data)
    pred_data_fixed = pred_data_fixed.astype(label_data.dtype)

    # V1 - PATCH HOLES
    # pred_data_fixed = np.where(label_data > 0, pred_data, 0)
    return pred_data_fixed


def outlier_removal_2d(pred_data: np.ndarray, label_data: np.ndarray):
    # V1 - PATCH HOLES
    pred_data_fixed = np.where(pred_data > 0, label_data, 0)
    pred_data_fixed = pred_data_fixed.astype(label_data.dtype)
    return pred_data_fixed


# def image_missing_connected_components_removal(pred, label):
#     pred_missing_components = np.maximum(label - pred, 0)
#
#     threshold_image = pred_missing_components.astype(np.uint8)
#     threshold_image[threshold_image > 0] = 255
#     connectivity = 4
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_image, connectivity, cv2.CV_32S)
#
#     area_threshold = 10
#     for i in range(num_labels):
#         if stats[i, cv2.CC_STAT_AREA] < area_threshold:
#             # Set as 0 pixels of connected component with area smaller than 10 pixels
#             pred_missing_components[labels == i] = 0
#
#     # Remove from the label the missing connected components in the pred that are have area bigger than 10 pixels
#     repaired_label = label - pred_missing_components
#     return repaired_label


#####################
# Create Pred Fixed #
#####################
def create_dataset_with_outliers_removed():
    # Inputs
    input_folders = {
        "labels": LABELS,
        "preds": PREDS
    }

    # Outputs
    output_folders = {
        "preds_fixed": PREDS_FIXED
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    filepaths_counts = []
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
        filepaths_counts.append(len(input_filepaths[key]))

    filepaths_count = max(filepaths_counts)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        label_filepath = input_filepaths["labels"][filepath_idx]
        pred_filepath = input_filepaths["preds"][filepath_idx]

        # Original data
        label_numpy_data = convert_data_file_to_numpy(data_filepath=label_filepath)
        pred_numpy_data = convert_data_file_to_numpy(data_filepath=pred_filepath)

        pred_fixed_numpy_data = outlier_removal_3d(pred_data=pred_numpy_data, label_data=label_numpy_data)
        data_stem = get_data_file_stem(data_filepath=pred_filepath, relative_to=input_folders["preds"])

        save_filename = os.path.join(output_folders["preds_fixed"], data_stem)
        convert_numpy_to_data_file(
            numpy_data=pred_fixed_numpy_data,
            source_data_filepath=pred_filepath,
            save_filename=save_filename,
            apply_data_threshold=True
        )


##################
# 2D Projections #
##################
def create_dataset_depth_2d_projections(data_options: dict):
    """
    Debug function to create 2D projections for the labels, preds, and evals
    :param data_options:
    :return:
    """
    # Inputs
    input_folders = {}

    # Outputs
    output_folders = {}

    for key, value in data_options.items():
        if value is True:
            folder_name = os.path.basename(key)
            input_folders[folder_name] = key
            output_folders[f"{folder_name}_2d"] = os.path.join(DATASET_PATH, f"{folder_name}_2d")

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    filepaths_counts = []
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
        filepaths_counts.append(len(input_filepaths[key]))

    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    filepaths_count = max(filepaths_counts)
    for filepath_idx in tqdm(range(filepaths_count)):
        for key in input_filepaths.keys():
            if filepath_idx >= len(input_filepaths[key]):
                continue  # No more files

            # Get index data:
            data_filepath = input_filepaths[key][filepath_idx]
            output_idx = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folders[key])

            data_numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)
            data_projections = project_3d_to_2d(
                data_3d=data_numpy_data,
                projection_options=projection_options,
                source_data_filepath=data_filepath
            )

            for image_view in IMAGES_6_VIEWS:
                output_2d_format = f"{output_idx}_{image_view}"

                data_image = data_projections[f"{image_view}_image"]
                save_filename = os.path.join(output_folders[f"{key}_2d"], f"{output_2d_format}.png")
                convert_numpy_to_data_file(
                    numpy_data=data_image,
                    source_data_filepath="dumpy.png",
                    save_filename=save_filename
                )


#################
# 3D Components #
#################
def create_data_components(data_options, connectivity_type: int = 26):
    # Inputs
    input_folders = {}

    # Outputs
    output_folders = {}

    if data_options.get(LABELS, False) is True:
        input_folders["labels"] = LABELS
        output_folders["labels_components"] = LABELS_COMPONENTS

    if data_options.get(PREDS, False) is True:
        input_folders["preds"] = PREDS
        output_folders["preds_components"] = PREDS_COMPONENTS

    if data_options.get(PREDS_FIXED, False) is True:
        input_folders["preds_fixed"] = PREDS_FIXED
        output_folders["preds_fixed_components"] = PREDS_FIXED_COMPONENTS

    if data_options.get(EVALS, False) is True:
        input_folders["evals"] = EVALS
        output_folders["evals_components"] = EVALS_COMPONENTS

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    filepaths_counts = []
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
        filepaths_counts.append(len(input_filepaths[key]))

    filepaths_count = max(filepaths_counts)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        for key in input_filepaths.keys():
            if filepath_idx >= len(input_filepaths[key]):
                continue  # No more files

            data_filepath = input_filepaths[key][filepath_idx]

            numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)
            (data_3d_components, _) = connected_components_3d(data_3d=numpy_data, connectivity_type=connectivity_type)

            # Save results
            save_name = data_filepath.relative_to(input_folders[key])
            save_filename = os.path.join(output_folders[f"{key}_components"], save_name)
            convert_numpy_to_data_file(
                numpy_data=data_3d_components,
                source_data_filepath=data_filepath,
                save_filename=save_filename
            )

            # TODO: Debug
            # save_nii_gz_in_identity_affine(numpy_data=data_3d_components, data_filepath=data_filepath,
            #                                save_filename=save_filename)


###############################
# 2D Projections and 3D Cubes #
###############################
def create_2d_projections_and_3d_cubes_for_training():
    # Config
    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    # Inputs
    input_folders = {
        "labels": LABELS,
        "preds": PREDS,
        "preds_fixed": PREDS_FIXED
    }

    # Outputs
    output_folders = {
        # Labels
        "labels_2d": LABELS_2D,
        "labels_3d": LABELS_3D,
        # Preds
        "preds_2d": PREDS_2D,
        "preds_3d": PREDS_3D,
        # Preds Fixed
        "preds_fixed_2d": PREDS_FIXED_2D,
        "preds_fixed_3d": PREDS_FIXED_3D,

        # Preds Advanced Fixed
        "preds_advanced_fixed_2d": PREDS_ADVANCED_FIXED_2D,
        "preds_advanced_fixed_3d": PREDS_ADVANCED_FIXED_3D
    }

    if TASK_TYPE == TaskType.SINGLE_COMPONENT:
        input_folders.update({
            "labels_components": LABELS_COMPONENTS,
            "preds_components": PREDS_COMPONENTS,
            "preds_fixed_components": PREDS_FIXED_COMPONENTS
        })

        output_folders.update({
            # Labels Components
            "labels_components_2d": LABELS_COMPONENTS_2D,
            "labels_components_3d": LABELS_COMPONENTS_3D,
            # Preds Components
            "preds_components_2d": PREDS_COMPONENTS_2D,
            "preds_components_3d": PREDS_COMPONENTS_3D,
            # Preds Fixed Components
            "preds_fixed_components_2d": PREDS_FIXED_COMPONENTS_2D,
            "preds_fixed_components_3d": PREDS_FIXED_COMPONENTS_3D,

            # Preds Advanced Fixed Components
            "preds_advanced_fixed_components_2d": PREDS_ADVANCED_FIXED_COMPONENTS_2D,
            "preds_advanced_fixed_components_3d": PREDS_ADVANCED_FIXED_COMPONENTS_3D
        })

    # Log Data
    log_data = dict()

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

    print("Cropping Mini Cubes...")
    filepaths_count = len(input_filepaths["labels"])
    if STOP_INDEX > 0:
        filepaths_count = min(filepaths_count, STOP_INDEX)
    for filepath_idx in range(filepaths_count):
        if not (START_INDEX <= (filepath_idx + 1)):
            continue

        #############
        # Load Data #
        #############

        # Get index data:
        label_filepath = input_filepaths["labels"][filepath_idx]
        pred_filepath = input_filepaths["preds"][filepath_idx]
        pred_fixed_filepath = input_filepaths["preds_fixed"][filepath_idx]

        # Original data
        label_numpy_data = convert_data_file_to_numpy(data_filepath=label_filepath)
        pred_numpy_data = convert_data_file_to_numpy(data_filepath=pred_filepath)
        pred_fixed_numpy_data = convert_data_file_to_numpy(data_filepath=pred_fixed_filepath, apply_data_threshold=True)

        # TODO: Debug
        # if (pred_numpy_data - pred_fixed_numpy_data).sum() != 0:
        #     raise ValueError("Outlier Removal Failed")

        output_idx = get_data_file_stem(data_filepath=label_filepath, relative_to=input_folders["labels"])
        print(f"[File: {output_idx}, Number: {filepath_idx + 1}/{filepaths_count}]")

        # if str(output_idx) != "46":
        #     continue

        #############
        # Crop Data #
        #############

        # Crop Mini Cubes
        label_cubes, other_cubes_list, cubes_data = crop_mini_cubes(
            data_3d=label_numpy_data,
            other_data_3d_list=[pred_numpy_data, pred_fixed_numpy_data],
            cube_dim=DATA_3D_SIZE,
            stride_dim=DATA_3D_STRIDE,
            cubes_data=True,
            index_3d=f"~{output_idx}"
        )
        pred_cubes = other_cubes_list[0]
        pred_fixed_cubes = other_cubes_list[1]

        if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            # Get index data:
            label_component_filepath = input_filepaths["labels_components"][filepath_idx]
            pred_component_filepath = input_filepaths["preds_components"][filepath_idx]
            pred_fixed_component_filepath = input_filepaths["preds_fixed_components"][filepath_idx]

            # Original data
            labels_component_numpy_data = convert_data_file_to_numpy(data_filepath=label_component_filepath)
            pred_component_numpy_data = convert_data_file_to_numpy(data_filepath=pred_component_filepath)
            pred_fixed_component_numpy_data = convert_data_file_to_numpy(data_filepath=pred_fixed_component_filepath)

            # Crop Mini Cubes
            label_components_cubes, other_cubes_list = crop_mini_cubes(
                data_3d=labels_component_numpy_data,
                other_data_3d_list=[pred_component_numpy_data, pred_fixed_component_numpy_data],
                cube_dim=DATA_3D_SIZE,
                stride_dim=DATA_3D_STRIDE
            )
            pred_components_cubes = other_cubes_list[0]
            pred_fixed_components_cubes = other_cubes_list[1]

        elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
            label_components_cubes = None
            pred_components_cubes = None
            pred_fixed_components_cubes = None

        else:
            raise ValueError("Invalid Task Type")

        # print(f"Total Mini Cubes: {len(label_cubes)}\n")  # Already appears on tqdm

        cubes_count = len(label_cubes)
        cubes_count_digits_count = len(str(cubes_count))
        for cube_idx in tqdm(range(cubes_count)):
            # Get index data:
            label_cube = label_cubes[cube_idx]
            pred_cube = pred_cubes[cube_idx]
            pred_fixed_cube = pred_fixed_cubes[cube_idx]

            cube_idx_str = str(cube_idx).zfill(cubes_count_digits_count)

            # if cube_idx_str != "061":
            #     continue

            # if cube_idx_str != "303":
            #     continue

            # TODO: Add checks if the advanced fixed preds are valid (equal to 2d projections)
            # if cube_idx_str == "3253":
            #     print("Debug")

            #####################
            # Task Filter Crops #
            #####################

            if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                # Get index data:
                label_components_cube = label_components_cubes[cube_idx]
                pred_components_cube = pred_components_cubes[cube_idx]
                pred_fixed_components_cube = pred_fixed_components_cubes[cube_idx]

                # TASK CONDITION: The region has 2 or more different global components
                global_components_3d_indices = list(np.unique(pred_components_cube))
                # global_components_3d_indices = list(np.unique(pred_fixed_components_cube))
                global_components_3d_indices.remove(0)
                global_components_3d_count = len(global_components_3d_indices)
                if global_components_3d_count < 2:
                    continue

                # Log 3D info
                cubes_data[cube_idx].update({
                    # "name": output_3d_format,
                    "pred_global_components": global_components_3d_count,
                })

            elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
                label_components_cube = None
                pred_components_cube = None
                pred_fixed_components_cube = None

                # TASK CONDITION: NONE

            elif TASK_TYPE == TaskType.PATCH_HOLES:
                label_components_cube = None
                pred_components_cube = None
                pred_fixed_components_cube = None

                # TASK CONDITION: NONE

                delta = np.abs(label_cube - pred_fixed_cube) > 0.5
                delta_count = np.count_nonzero(delta)
                # if delta_count == 0:
                #     continue

                # Log 3D info
                cubes_data[cube_idx].update({
                    # "name": output_3d_format,
                    "delta_count": delta_count,
                })
            else:
                raise ValueError("Invalid Task Type")

            ####################
            # Project 3D to 2D #
            ####################

            # Project 3D to 2D (Labels)
            label_projections = project_3d_to_2d(
                data_3d=label_cube,
                projection_options=projection_options,
                component_3d=label_components_cube,
                source_data_filepath=label_filepath
            )

            # Project 3D to 2D (Preds)
            pred_projections = project_3d_to_2d(
                data_3d=pred_cube,
                projection_options=projection_options,
                component_3d=pred_components_cube,
                source_data_filepath=pred_filepath
            )

            # Project 3D to 2D (Preds Fixed)
            pred_fixed_projections = project_3d_to_2d(
                data_3d=pred_fixed_cube,
                projection_options=projection_options,
                component_3d=pred_fixed_components_cube,
                source_data_filepath=pred_filepath
            )

            ########################
            # Density Filter Crops #
            ########################

            condition_list = [True] * len(IMAGES_6_VIEWS)
            for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
                label_image = label_projections[f"{image_view}_image"]
                pred_image = pred_projections[f"{image_view}_image"]
                pred_fixed_image = pred_fixed_projections[f"{image_view}_image"]

                # Repair the labels - TODO: Check how to do smartly
                # label_projections[f"{image_6_view}_repaired_image"] = image_missing_connected_components_removal(
                #     pred=pred_image,
                #     label=label_image
                # )
                # repaired_label_image = label_projections[f"{image_6_view}_repaired_image"]

                if APPLY_MEDIAN_FILTER:
                    label_image = cv2.medianBlur(label_image, ksize=5)
                    pred_image = cv2.medianBlur(pred_image, ksize=5)
                    pred_fixed_image = cv2.medianBlur(pred_fixed_image, ksize=5)
                    pred_fixed_image = outlier_removal_2d(pred_data=pred_fixed_image, label_data=label_image)

                    label_projections[f"{image_view}_image"] = label_image
                    pred_projections[f"{image_view}_image"] = pred_image
                    pred_fixed_projections[f"{image_view}_image"] = pred_fixed_image

                # Check Condition (If condition fails, mark the view as invalid):
                condition = [
                    not (UPPER_THRESHOLD_2D > np.count_nonzero(label_image) > LOWER_THRESHOLD_2D),
                    # not (UPPER_THRESHOLD_2D > np.count_nonzero(pred_image) > LOWER_THRESHOLD_2D),  # Optional
                    not (UPPER_THRESHOLD_2D > np.count_nonzero(pred_fixed_image) > LOWER_THRESHOLD_2D),
                ]

                # Check Condition (If condition fails, skip the current view):
                if any(condition):
                    condition_list[view_idx] = False

                    cubes_data[cube_idx].update({
                        f"{image_view}_valid": False
                    })
                else:
                    cubes_data[cube_idx].update({
                        f"{image_view}_valid": True
                    })

            # Validate that at least 1 condition is met (if not, skip cube data)
            if not any(condition_list):
                continue

            ########################
            # Continuity Fix in 3D #
            ########################

            pred_advanced_fixed_cube = pred_fixed_cube.copy()
            pred_advanced_fixed_components_cube = None

            if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                if APPLY_CONTINUITY_FIX_3D:
                    # Update the pred_advanced_fixed_cube
                    pred_advanced_fixed_cube = components_continuity_3d_single_component(
                        label_cube=label_cube,
                        pred_advanced_fixed_cube=pred_advanced_fixed_cube,
                        reverse_mode=False,
                        connectivity_type=TRAIN_CONNECTIVITY_TYPE_3D,
                        hard_condition=False
                    )

                # Calculate the connected components for the advanced fixed preds
                (pred_advanced_fixed_components_cube, _) = connected_components_3d(data_3d=pred_advanced_fixed_cube)

            elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
                if APPLY_CONTINUITY_FIX_3D:
                    # Update the pred_advanced_fixed_cube
                    pred_advanced_fixed_cube = components_continuity_3d_local_connectivity(
                        label_cube=label_cube,
                        pred_advanced_fixed_cube=pred_advanced_fixed_cube,
                        reverse_mode=False,
                        connectivity_type=TRAIN_CONNECTIVITY_TYPE_3D,
                        hard_condition=False,
                        apply_dilation_scope=BINARY_DILATION
                    )

            elif TASK_TYPE == TaskType.PATCH_HOLES:
                pass

            else:
                raise ValueError("Invalid Task Type")

            # Project 3D to 2D (Preds Fixed Advanced)
            source_data_filepath = pred_fixed_filepath
            pred_advanced_fixed_projections = project_3d_to_2d(
                data_3d=pred_advanced_fixed_cube,
                projection_options=projection_options,
                component_3d=pred_advanced_fixed_components_cube,
                source_data_filepath=source_data_filepath
            )

            ########################
            # Continuity Fix in 2D #
            ########################

            for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
                label_image = label_projections[f"{image_view}_image"]
                pred_advanced_fixed_image = pred_advanced_fixed_projections[f"{image_view}_image"]

                # Used mainly for noisy PCD projections
                if APPLY_MEDIAN_FILTER:
                    pred_advanced_fixed_image = cv2.medianBlur(pred_advanced_fixed_image, ksize=5)
                    pred_advanced_fixed_image = outlier_removal_2d(
                        pred_data=pred_advanced_fixed_image,
                        label_data=label_image
                    )

                    pred_advanced_fixed_projections[f"{image_view}_image"] = pred_advanced_fixed_image

                if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                    if APPLY_CONTINUITY_FIX_2D:
                        # Update the pred_advanced_fixed_image
                        pred_advanced_fixed_image = components_continuity_2d_single_component(
                            label_image=label_image,
                            pred_advanced_fixed_image=pred_advanced_fixed_image,
                            reverse_mode=False,
                            connectivity_type=TRAIN_CONNECTIVITY_TYPE_2D,
                            binary_diff=False,
                            hard_condition=False
                        )
                    pred_advanced_fixed_projections[f"{image_view}_image"] = pred_advanced_fixed_image

                    # Calculate the connected components for the advanced fixed preds
                    labeled_delta = connected_components_2d(data_2d=pred_advanced_fixed_image)[0]
                    pred_advanced_fixed_projections[f"{image_view}_components"] = color.label2rgb(
                        label=labeled_delta
                    ) * 255

                elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
                    if APPLY_CONTINUITY_FIX_2D:
                        # Update the pred_advanced_fixed_image
                        pred_advanced_fixed_image = components_continuity_2d_local_connectivity(
                            label_image=label_image,
                            pred_advanced_fixed_image=pred_advanced_fixed_image,
                            reverse_mode=False,
                            connectivity_type=TRAIN_CONNECTIVITY_TYPE_2D,
                            binary_diff=False,
                            hard_condition=False,
                            apply_dilation_scope=BINARY_DILATION
                        )
                    pred_advanced_fixed_projections[f"{image_view}_image"] = pred_advanced_fixed_image

                elif TASK_TYPE == TaskType.PATCH_HOLES:
                    pass

                else:
                    raise ValueError("Invalid Task Type")

                # Log info
                if np.array_equal(pred_advanced_fixed_image, label_image):
                    cubes_data[cube_idx].update({
                        f"{image_view}_advance_valid": False
                    })
                else:
                    cubes_data[cube_idx].update({
                        f"{image_view}_advance_valid": True
                    })

            ###############
            # Export Data #
            ###############

            folder_2d_map = {
                "labels_2d": label_projections,
                "preds_2d": pred_projections,
                "preds_fixed_2d": pred_fixed_projections,
                "preds_advanced_fixed_2d": pred_advanced_fixed_projections
            }

            folder_2d_components_map = {
                "labels_components_2d": label_projections,
                "preds_components_2d": pred_projections,
                "preds_fixed_components_2d": pred_fixed_projections,
                "preds_advanced_fixed_components_2d": pred_advanced_fixed_projections
            }

            for image_view in IMAGES_6_VIEWS:
                output_2d_format = f"{output_idx}_{cube_idx_str}_{image_view}"

                for key, value in folder_2d_map.items():
                    data_2d = value[f"{image_view}_image"]
                    save_filename = os.path.join(output_folders[key], output_2d_format)
                    convert_numpy_to_data_file(
                        numpy_data=data_2d,
                        source_data_filepath="dumpy.png",
                        save_filename=save_filename
                    )

                if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                    for key, value in folder_2d_components_map.items():
                        data_2d = value[f"{image_view}_components"]
                        save_filename = os.path.join(output_folders[key], output_2d_format)
                        convert_numpy_to_data_file(
                            numpy_data=data_2d,
                            source_data_filepath="dumpy.png",
                            save_filename=save_filename
                        )

                elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
                    pass

                else:
                    raise ValueError("Invalid Task Type")

            output_3d_format = f"{output_idx}_{cube_idx_str}"

            folder_3d_map = {
                "labels_3d": label_cube,
                "preds_3d": pred_cube,
                "preds_fixed_3d": pred_fixed_cube,
                "preds_advanced_fixed_3d": pred_advanced_fixed_cube
            }

            folder_3d_components_map = {
                "labels_components_3d": label_components_cube,
                "preds_components_3d": pred_components_cube,
                "preds_fixed_components_3d": pred_fixed_components_cube,
                "preds_advanced_fixed_components_3d": pred_advanced_fixed_components_cube
            }

            for key, value in folder_3d_map.items():
                if key == "preds_advanced_fixed_3d":
                    source_name = "preds_fixed"
                else:
                    source_name = key.replace("_3d", "")
                source_data_filepath = input_filepaths[source_name][filepath_idx]
                save_filename = os.path.join(output_folders[key], output_3d_format)
                convert_numpy_to_data_file(
                    numpy_data=value,
                    source_data_filepath=source_data_filepath,
                    save_filename=save_filename,
                    apply_data_threshold=True
                )

            if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                for key, value in folder_3d_components_map.items():
                    if key == "preds_advanced_fixed_components_3d":
                        source_name = "preds_fixed_components"
                    else:
                        source_name = key.replace("_3d", "")
                    source_data_filepath = input_filepaths[source_name][filepath_idx]
                    save_filename = os.path.join(output_folders[key], output_3d_format)
                    convert_numpy_to_data_file(
                        numpy_data=value,
                        source_data_filepath=source_data_filepath,
                        save_filename=save_filename
                    )

                (label_components_cube, _) = connected_components_3d(data_3d=label_cube)

                local_components_3d_indices = list(np.unique(label_components_cube))
                local_components_3d_indices.remove(0)
                local_components_3d_count = len(local_components_3d_indices)

                # Log 3D info
                cubes_data[cube_idx].update({
                    "label_local_components": local_components_3d_count
                })

            elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
                pass

            else:
                raise ValueError("Invalid Task Type")

            log_data[output_3d_format] = cubes_data[cube_idx]

            # DEBUG
            # _convert_numpy_to_nii_gz(label_cube, save_name="1")
            # _convert_numpy_to_nii_gz(pred_cube, save_name="2")

        if (filepath_idx + 1) == STOP_INDEX:
            break

    pd.DataFrame(data=log_data).T.to_csv(path_or_buf=TRAIN_LOG_PATH)


def create_2d_projections_and_3d_cubes_for_evaluation(include_labels=False):
    # Config
    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    # Inputs
    input_folders = {
        "evals": EVALS
    }

    # Outputs
    output_folders = {
        # Evals
        "evals_2d": EVALS_2D,
        "evals_3d": EVALS_3D
    }

    if include_labels:
        # Inputs
        input_folders.update({
            "labels": LABELS
        })

        # Outputs
        output_folders.update({
            # Labels
            "labels_2d": LABELS_2D,
            "labels_3d": LABELS_3D
        })

    if TASK_TYPE == TaskType.SINGLE_COMPONENT:
        input_folders.update({
            "evals_components": EVALS_COMPONENTS
        })

        output_folders.update({
            # Evals Components
            "evals_components_2d": EVALS_COMPONENTS_2D,
            "evals_components_3d": EVALS_COMPONENTS_3D
        })

        if include_labels:
            input_folders.update({
                "labels_components": LABELS_COMPONENTS
            })

            output_folders.update({
                # Labels Components
                "labels_components_2d": LABELS_COMPONENTS_2D,
                "labels_components_3d": LABELS_COMPONENTS_3D
            })

    # Log Data
    log_data = dict()

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Evals cubes backup (for density filter)
    if include_labels:
        evals_cubes_backup = defaultdict(dict)
    else:
        evals_cubes_backup = None

    # Get the filepaths
    input_filepaths = dict()
    filepaths_found = list()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
        filepaths_found.append(len(input_filepaths[key]))

    # Validation
    if len(set(filepaths_found)) != 1:
        raise ValueError("Different number of files found in the Input folders")

    # Using loop as the cases are symmetric
    for data_key in input_folders.keys():
        print(f"Cropping Mini Cubes... [{data_key}]")
        filepaths_count = len(input_filepaths[data_key])
        if STOP_INDEX > 0:
            filepaths_count = min(filepaths_count, STOP_INDEX)
        for filepath_idx in range(filepaths_count):
            if not (START_INDEX <= (filepath_idx + 1)):
                continue

            #############
            # Load Data #
            #############

            # Get index data:
            data_filepath = input_filepaths[data_key][filepath_idx]

            # Original data
            data_numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)

            output_idx = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folders[data_key])
            print(f"[File: {output_idx}, Number: {filepath_idx + 1}/{filepaths_count}]")

            #############
            # Crop Data #
            #############

            data_cubes, cubes_data = crop_mini_cubes(
                data_3d=data_numpy_data,
                cube_dim=DATA_3D_SIZE,
                stride_dim=DATA_3D_STRIDE,
                cubes_data=True,
                index_3d=f"~{output_idx}"
            )

            if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                # Get index data:
                data_component_filepath = input_filepaths[f"{data_key}_components"][filepath_idx]

                # Original data
                data_component_numpy_data = convert_data_file_to_numpy(data_filepath=data_component_filepath)

                # Crop Mini Cubes
                data_components_cubes = crop_mini_cubes(
                    data_3d=data_component_numpy_data,
                    cube_dim=DATA_3D_SIZE,
                    stride_dim=DATA_3D_STRIDE
                )

            elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
                data_component_filepath = None
                data_components_cubes = None

            else:
                raise ValueError("Invalid Task Type")

            # print(f"Total Mini Cubes: {len(data_cubes)}\n")  # Already appears on tqdm

            cubes_count = len(data_cubes)
            cubes_count_digits_count = len(str(cubes_count))
            for cube_idx in tqdm(range(cubes_count)):
                # Get index data:
                data_cube = data_cubes[cube_idx]

                cube_idx_str = str(cube_idx).zfill(cubes_count_digits_count)

                # TODO: enable 2 modes
                if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                    # Get index data:
                    data_components_cube = data_components_cubes[cube_idx]

                    # TASK CONDITION: The region has 2 or more different global components
                    global_components_3d_indices = list(np.unique(data_components_cube))
                    global_components_3d_indices.remove(0)
                    global_components_3d_count = len(global_components_3d_indices)
                    if global_components_3d_count < 2:
                        continue

                    # Log 3D info
                    cubes_data[cube_idx].update({
                        # "name": output_3d_format,
                        "eval_global_components": global_components_3d_count,
                    })

                elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
                    data_components_cube = None

                    # TASK CONDITION: NONE

                elif TASK_TYPE == TaskType.PATCH_HOLES:
                    data_components_cube = None

                    # TASK CONDITION: NONE

                else:
                    raise ValueError("Invalid Task Type")

                ####################
                # Project 3D to 2D #
                ####################

                # Project 3D to 2D (data)
                data_projections = project_3d_to_2d(
                    data_3d=data_cube,
                    projection_options=projection_options,
                    source_data_filepath=data_filepath,
                    component_3d=data_components_cube
                )

                ########################
                # Density Filter Crops #
                ########################

                condition_list = [True] * len(IMAGES_6_VIEWS)
                for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
                    # Calculate condition according to evals only!
                    if evals_cubes_backup is not None and data_key == "evals":
                        data_image = data_projections[f"{image_view}_image"]
                        evals_cubes_backup[cube_idx][f"{image_view}_image"] = data_image
                    else:
                        data_image = evals_cubes_backup[cube_idx][f"{image_view}_image"]

                    condition = [
                        not (UPPER_THRESHOLD_2D > np.count_nonzero(data_image) > LOWER_THRESHOLD_2D)
                    ]

                    # Check Condition (If condition fails, skip the current view):
                    if any(condition):
                        condition_list[view_idx] = False

                        cubes_data[cube_idx].update({
                            f"{image_view}_valid": False
                        })
                    else:
                        cubes_data[cube_idx].update({
                            f"{image_view}_valid": True
                        })

                # Validate that at least 1 condition is met (if not, pop cube data)
                if not any(condition_list):
                    continue

                ###############
                # Export Data #
                ###############

                for image_view in IMAGES_6_VIEWS:
                    output_2d_format = f"{output_idx}_{cube_idx_str}_{image_view}"

                    # 2D Folders - data
                    data_2d = data_projections[f"{image_view}_image"]
                    save_filename = os.path.join(output_folders[f"{data_key}_2d"], output_2d_format)
                    convert_numpy_to_data_file(
                        numpy_data=data_2d,
                        source_data_filepath="dumpy.png",
                        save_filename=save_filename
                    )

                    if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                        # 2D Folders - data components
                        data_components_2d = data_projections[f"{image_view}_components"]
                        save_filename = os.path.join(output_folders[f"{data_key}_components_2d"], output_2d_format)
                        convert_numpy_to_data_file(
                            numpy_data=data_components_2d,
                            source_data_filepath="dumpy.png",
                            save_filename=save_filename
                        )

                    elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
                        pass

                    else:
                        raise ValueError("Invalid Task Type")

                output_3d_format = f"{output_idx}_{cube_idx_str}"

                # 3D Folders - data
                save_filename = os.path.join(output_folders[f"{data_key}_3d"], output_3d_format)
                convert_numpy_to_data_file(
                    numpy_data=data_cube,
                    source_data_filepath=data_filepath,
                    save_filename=save_filename,
                    apply_data_threshold=True
                )

                if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                    # 3D Folders - data components
                    save_filename = os.path.join(output_folders[f"{data_key}_components_3d"], output_3d_format)
                    convert_numpy_to_data_file(
                        numpy_data=data_components_cube,
                        source_data_filepath=data_component_filepath,
                        save_filename=save_filename
                    )

                elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
                    pass

                else:
                    raise ValueError("Invalid Task Type")

                log_data[output_3d_format] = cubes_data[cube_idx]

            if (filepath_idx + 1) == STOP_INDEX:
                break

        # Only evals log is getting exported
        if data_key == "evals":
            pd.DataFrame(data=log_data).T.to_csv(path_or_buf=EVAL_LOG_PATH)


def main():
    # TODO: Required for training with all modes
    if os.path.exists(PREDS_FIXED) is False:
        create_dataset_with_outliers_removed()

    # data_options = {
    #     LABELS: True,
    #     PREDS: False,
    #     PREDS_FIXED: False,
    #     EVALS: False
    # }

    # TODO: Required for training with TaskType.SINGLE_COMPONENT
    # connectivity_type = 26
    # create_data_components(data_options=data_options, connectivity_type=connectivity_type)

    # TODO: DEBUG
    # create_dataset_depth_2d_projections(data_options=data_options)

    create_2d_projections_and_3d_cubes_for_training()
    # create_2d_projections_and_3d_cubes_for_evaluation(include_labels=True)


if __name__ == "__main__":
    # TODO: Add classifier model to find cubes with holes better - model 2D to 1D

    # TODO:
    # 1. Make sure DATASET_FOLDER folder is present in the root directory
    main()

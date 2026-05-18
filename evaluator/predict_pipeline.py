import argparse
import os
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import convolve
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import sys
root_path = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(root_path)

from configs.configs_parser import *
from datasets.dataset_utils import *
from models.model_list import init_model
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d

#########
# Utils #
#########

from evaluator.offline_utils import offline_preprocess_2d, offline_preprocess_3d
from evaluator.online_utils import online_preprocess_2d, online_preprocess_3d

def postprocess_2d(data_3d_stem: str,
                   data_2d_input: torch.Tensor,
                   data_2d_output: torch.Tensor,
                   apply_input_merge_2d: bool = False,
                   apply_noise_filter_2d: bool = False,
                   hard_noise_filter_2d: bool = True,
                   connectivity_type_2d: int = 4,
                   log_data: pd.DataFrame = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert (1, 6, w, h) to (6, w, h)
    if data_2d_input.shape[0] == 1 and data_2d_input.shape[1] == 6:
        data_2d_input = data_2d_input.squeeze(0)
        data_2d_output = data_2d_output.squeeze(0)
    # Convert (6, 1, w, h) to (6, w, h)
    elif data_2d_input.shape[0] == 6 and data_2d_input.shape[1] == 1:
        data_2d_input = data_2d_input.squeeze(1)
        data_2d_output = data_2d_output.squeeze(1)
    # Apply no change to (6, w, h)
    elif data_2d_input.shape[0] == 6 and len(data_2d_input.shape) == 3:
        pass
    else:
        raise ValueError("Invalid shape")

    if log_data is not None:
        # Replace invalid data with original input
        for idx, image_view in enumerate(IMAGES_6_VIEWS):
            view_advance_valid = f"{image_view}_advance_valid"
            if view_advance_valid in log_data.columns:
                data_2d_matching_rows = log_data[log_data[log_data.columns[0]] == data_3d_stem]
                data_2d_row = list(data_2d_matching_rows.iterrows())[0][1]
                if data_2d_row[view_advance_valid] is False:
                    data_2d_output[idx] = data_2d_input[idx].clone()
                else:
                    pass
            else:
                # print(f"No '{view_advance_valid}' column in log csv data")
                pass  # For Eval data there is no advance_valid column
    else:  # TODO: use classifier results
        pass

    if apply_input_merge_2d is True:
        data_2d_output = data_2d_input + torch.where(data_2d_input == 0, data_2d_output, 0)

    if apply_noise_filter_2d is True:
        for idx in range(len(IMAGES_6_VIEWS)):
            data_2d_input_idx = np.round(data_2d_input[idx].cpu().numpy() * 255).astype(np.uint8)
            data_2d_output_idx = np.round(data_2d_output[idx].cpu().numpy() * 255).astype(np.uint8)

            # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            #     filtered_output = components_continuity_2d_single_component(
            #         label_image=data_2d_input_idx,
            #         pred_advanced_fixed_image=data_2d_output_idx,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_2d,
            #         binary_diff=True,
            #         hard_condition=hard_noise_filter_2d,
            #         apply_dilation_scope=BINARY_DILATION
            #     )
            # elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
            #     filtered_output = components_continuity_2d_local_connectivity(
            #         label_image=data_2d_input_idx,
            #         pred_advanced_fixed_image=data_2d_output_idx,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_2d,
            #         binary_diff=True,
            #         hard_condition=hard_noise_filter_2d,
            #         apply_dilation_scope=BINARY_DILATION
            #     )
            # else:
            #     filtered_output = data_2d_output_idx

            filtered_output = components_continuity_2d_local_connectivity(
                label_image=data_2d_input_idx,
                pred_advanced_fixed_image=data_2d_output_idx,
                reverse_mode=False,
                connectivity_type=connectivity_type_2d,
                binary_diff=True,
                hard_condition=hard_noise_filter_2d,
                apply_dilation_scope=BINARY_DILATION
            )

            data_2d_output[idx] = torch.Tensor(filtered_output / 255.0).to(data_2d_output.device)

    return data_2d_input, data_2d_output


def debug_2d(data_3d_stem: str, data_2d_input: torch.Tensor, data_2d_output: torch.Tensor):
    data_2d_input_copy = data_2d_input.clone().cpu().numpy()
    data_2d_output_copy = data_2d_output.clone().cpu().numpy()

    columns = 6
    rows = 2
    fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
    ax = list()

    # 2D Input
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 0 * columns + j + 1))
        numpy_image = data_2d_input_copy[j]
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    # 2D Output
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 1 * columns + j + 1))
        numpy_image = data_2d_output_copy[j]
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d_debug")
    save_filename = os.path.join(save_path, data_3d_stem)
    os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)


def export_output_2d(data_3d_stem, data_3d_filepath, data_2d_output: torch.Tensor):
    numpy_data_2d_output = data_2d_output.clone().cpu().numpy()

    source_data_filepath = "dummy.png"
    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d")
    for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
        numpy_image = numpy_data_2d_output[view_idx]
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)

        save_filename = os.path.join(save_path, f"{data_3d_stem}_{image_view}_output")
        convert_numpy_to_data_file(
            numpy_data=numpy_image,
            source_data_filepath=source_data_filepath,
            save_filename=save_filename,
        )


def naive_noise_filter(data_3d_original: np.ndarray, data_3d_input: np.ndarray):
    # Define a 3x3x3 kernel that will be used to check 6 neighbors (left, right, up, down, front, back)
    kernel = np.zeros((3, 3, 3), dtype=int)

    # Set only the 6-connectivity neighbors in the kernel
    kernel[1, 0, 1] = 1  # Left
    kernel[1, 2, 1] = 1  # Right
    kernel[0, 1, 1] = 1  # Up
    kernel[2, 1, 1] = 1  # Down
    kernel[1, 1, 0] = 1  # Front
    kernel[1, 1, 2] = 1  # Back

    # Convolve the binary array with the kernel to count neighbors
    neighbors_count = convolve(data_3d_input, kernel, mode='constant', cval=0)

    # Filter out voxels that have no neighboring voxels (i.e., neighbors_count == 0)
    filtered_data_3d_input = np.where((data_3d_input == 1) & (neighbors_count > 0), 1, 0)

    return filtered_data_3d_input


def postprocess_3d(data_3d_input: torch.Tensor,
                   data_3d_output: torch.Tensor,
                   apply_threshold_3d: bool = True,
                   threshold_3d: float = 0.5,
                   apply_input_merge_3d: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    data_3d_input = data_3d_input.squeeze().squeeze()
    data_3d_output = data_3d_output.squeeze().squeeze()

    # TODO: Threshold
    if apply_threshold_3d:
        apply_threshold(data_3d_output, threshold=threshold_3d, keep_values=False)

    if apply_input_merge_3d is True:
        data_3d_output = data_3d_input + torch.where(data_3d_input == 0, data_3d_output, 0)

    return data_3d_input, data_3d_output


def debug_3d(data_3d_stem, data_3d_filepath, data_3d_input: torch.Tensor):
    data_3d_input = data_3d_input.clone().cpu().numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d_debug")
    save_filename = os.path.join(save_path, f"{data_3d_stem}_input")
    convert_numpy_to_data_file(
        numpy_data=data_3d_input,
        source_data_filepath=data_3d_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )


def export_output_3d(data_3d_stem, data_3d_filepath, data_3d_output: torch.Tensor):
    numpy_data_3d_output = data_3d_output.cpu().numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")
    save_filename = os.path.join(save_path, f"{data_3d_stem}_output")
    convert_numpy_to_data_file(
        numpy_data=numpy_data_3d_output,
        source_data_filepath=data_3d_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )


##################
# Core Functions #
##################
def init_pipeline_models(args: argparse.Namespace):
    # Load 1D model

    # Load 2D model
    if len(args.model_2d) > 0:
        args.input_size = args.input_size_model_2d
        args.model = args.model_2d
        model_2d = init_model(args=args)
        if WEIGHTS_2D_PATH is not None:
            model_2d_weights_filepath = WEIGHTS_2D_PATH
        else:
            weights_name = f"Network_{DATASET_OUTPUT_FOLDER}_{model_2d.model_name}.pth"
            model_2d_weights_filepath = os.path.join(ROOT_PATH, "weights", weights_name)
        model_2d.load_state_dict(state_dict=torch.load(f=model_2d_weights_filepath, map_location=args.device))
        model_2d.eval()
        model_2d.to(args.device)
        args.model_2d_class = model_2d
    else:
        pass

    # Load 3D model
    if len(args.model_3d) > 0:
        args.input_size = args.input_size_model_3d
        args.model = args.model_3d
        model_3d = init_model(args=args)
        if WEIGHTS_3D_PATH is not None:
            model_3d_weights_filepath = WEIGHTS_3D_PATH
        else:
            weights_name = f"Network_{DATASET_OUTPUT_FOLDER}_{model_3d.model_name}.pth"
            model_3d_weights_filepath = os.path.join(ROOT_PATH, "weights", weights_name)
        model_3d.load_state_dict(state_dict=torch.load(f=model_3d_weights_filepath, map_location=args.device))
        model_3d.eval()
        model_3d.to(args.device)
        args.model_3d_class = model_3d
    else:
        pass


def single_predict(args: argparse.Namespace,
                   data_3d_filepath,
                   projections_data=None,  # For online mode only
                   data_3d_folder=None, data_2d_folder=None,  # For offline mode only
                   log_data=None, enable_debug=True,
                   run_2d_flow=True, run_3d_flow=True,
                   export_2d=True, export_3d=True):
    if args.mode == "offline" and not os.path.exists(data_3d_filepath):
        raise ValueError(f"File: {data_3d_filepath} doesn't exist!")

    # CONFIGS
    apply_fusion = APPLY_FUSION

    apply_input_merge_2d = APPLY_INPUT_MERGE_2D
    apply_threshold_2d = APPLY_THRESHOLD_2D
    threshold_2d = THRESHOLD_2D
    apply_noise_filter_2d = APPLY_NOISE_FILTER_2D
    hard_noise_filter_2d = HARD_NOISE_FILTER_2D
    connectivity_type_2d = PREDICT_CONNECTIVITY_TYPE_2D

    apply_input_merge_3d = APPLY_INPUT_MERGE_3D
    apply_threshold_3d = APPLY_THRESHOLD_3D
    threshold_3d = THRESHOLD_3D
    apply_noise_filter_3d = APPLY_NOISE_FILTER_3D
    hard_noise_filter_3d = HARD_NOISE_FILTER_3D
    connectivity_type_3d = PREDICT_CONNECTIVITY_TYPE_3D

    # INPUTS
    data_3d_filepath = str(data_3d_filepath)

    if args.mode == "offline":
        data_3d_folder = str(data_3d_folder)
        data_2d_folder = str(data_2d_folder)

    data_3d_stem = get_data_file_stem(data_filepath=data_3d_filepath, relative_to=data_3d_folder)

    # os.makedirs(PREDICT_PIPELINE_RESULTS_PATH, exist_ok=True)

    if args.input_size_model_2d[0] == 6 and len(args.input_size_model_2d) == 3:
        apply_batch_merge = True
    else:
        apply_batch_merge = False

    with torch.no_grad():
        # TODO: Support 1D model TBD

        ##############
        # 2D Section #
        ##############

        if run_2d_flow is True:
            # Preprocess 2D
            if args.mode == "offline":
                data_2d_input = offline_preprocess_2d(
                    data_3d_stem=data_3d_stem,
                    data_2d_folder=data_2d_folder,
                    apply_batch_merge=apply_batch_merge
                )
            elif args.mode == "online":
                data_2d_input = online_preprocess_2d(
                    data_3d_stem=data_3d_stem,
                    projections_data=projections_data,
                    apply_batch_merge=apply_batch_merge
                )
            else:
                raise ValueError(f"Mode '{args.mode}' is not supported!")

            # Predict 2D
            data_2d_input = data_2d_input.to(args.device)
            if len(args.model_2d) > 0:
                data_2d_output = args.model_2d_class(data_2d_input)

                # Handle additional tasks
                if "confidence map" in getattr(args.model_2d_class, "additional_tasks", list()):
                    data_2d_output, data_2d_output_confidence = data_2d_output
                    data_2d_output = torch.where(data_2d_output_confidence > 0.5, data_2d_output, 0)
            else:
                data_2d_output = data_2d_input.clone()

            # Postprocess 2D
            (data_2d_input, data_2d_output) = postprocess_2d(
                data_3d_stem=data_3d_stem,
                data_2d_input=data_2d_input,
                data_2d_output=data_2d_output,
                apply_input_merge_2d=apply_input_merge_2d,
                apply_noise_filter_2d=apply_noise_filter_2d,
                hard_noise_filter_2d=hard_noise_filter_2d,
                connectivity_type_2d=connectivity_type_2d,
                log_data=log_data
            )

            # DEBUG
            if enable_debug is True:
                debug_2d(data_3d_stem=data_3d_stem, data_2d_input=data_2d_input, data_2d_output=data_2d_output)

            if export_2d is True:
                export_output_2d(
                    data_3d_stem=data_3d_stem,
                    data_3d_filepath=data_3d_filepath,
                    data_2d_output=data_2d_output
                )
        else:
            data_2d_output = None

        ##############
        # 3D Section #
        ##############

        if run_3d_flow is True:
            # Preprocess 3D
            if args.mode == "offline":
                data_3d_input = offline_preprocess_3d(
                    data_3d_filepath=data_3d_filepath,
                    data_2d_output=data_2d_output,
                    apply_threshold_2d=apply_threshold_2d,
                    threshold_2d=threshold_2d,
                    apply_fusion=apply_fusion,
                    apply_noise_filter_3d=apply_noise_filter_3d,
                    hard_noise_filter_3d=hard_noise_filter_3d,
                    connectivity_type_3d=connectivity_type_3d
                )
            elif args.mode == "online":
                data_3d_input = online_preprocess_3d(
                    data_3d_filepath=data_3d_filepath,
                    projections_data=projections_data,
                    data_2d_output=data_2d_output,
                    apply_threshold_2d=apply_threshold_2d,
                    threshold_2d=threshold_2d,
                    apply_fusion=apply_fusion,
                    apply_noise_filter_3d=apply_noise_filter_3d,
                    hard_noise_filter_3d=hard_noise_filter_3d,
                    connectivity_type_3d=connectivity_type_3d
                )
            else:
                raise ValueError(f"Mode '{args.mode}' is not supported!")

            # Predict 3D
            data_3d_input = data_3d_input.to(args.device)
            if len(args.model_3d) > 0:
                data_3d_output = args.model_3d_class(data_3d_input)
            else:
                data_3d_output = data_3d_input.clone()

            # Postprocess 3D
            (data_3d_input, data_3d_output) = postprocess_3d(
                data_3d_input=data_3d_input,
                data_3d_output=data_3d_output,
                apply_threshold_3d=apply_threshold_3d,
                threshold_3d=threshold_3d,
                apply_input_merge_3d=apply_input_merge_3d
            )

            # DEBUG
            if enable_debug is True:
                debug_3d(data_3d_stem=data_3d_stem, data_3d_filepath=data_3d_filepath, data_3d_input=data_3d_input)
                # TODO: Add matplotlib export

            if export_3d is True:
                export_output_3d(
                    data_3d_stem=data_3d_stem,
                    data_3d_filepath=data_3d_filepath,
                    data_3d_output=data_3d_output
                )
        else:
            pass

# TODO: Debug Tools
def test_single_predict(args: argparse.Namespace):
    data_type = DataType.TRAIN
    # data_type = DataType.EVAL

    # base_filename = "PA000005_11899.nii.gz"
    # base_filename = "PA000078_11996.nii.gz"
    # base_filename = "47_52.npy"
    base_filename = "46_053.npy"

    if data_type == DataType.TRAIN:
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        # data_3d_folder = PREDS_3D
        # data_2d_folder = PREDS_2D

        data_3d_folder = PREDS_FIXED_3D
        data_2d_folder = PREDS_FIXED_2D

        # data_3d_folder = PREDS_ADVANCED_FIXED_3D
        # data_2d_folder = PREDS_ADVANCED_FIXED_2D
    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        data_3d_folder = EVALS_3D
        data_2d_folder = EVALS_2D

    # Get filepaths
    data_3d_filepath = os.path.join(data_3d_folder, base_filename)
    single_predict(
        args=args,
        data_3d_filepath=data_3d_filepath,
        data_3d_folder=data_3d_folder,
        data_2d_folder=data_2d_folder,
        log_data=log_data
    )


def full_predict(args: argparse.Namespace,
                 data_3d_stem, data_type: DataType, log_data=None, data_3d_folder=None, data_2d_folder=None,
                 run_2d_flow=True, run_3d_flow=True, export_2d=True, export_3d=True):
    # LOAD DATA #
    if data_type == DataType.TRAIN:
        if log_data is None:
            log_data = pd.read_csv(TRAIN_LOG_PATH)

        if data_3d_folder is None:
            # data_3d_folder = PREDS_3D
            data_3d_folder = PREDS_FIXED_3D
            # data_3d_folder = PREDS_ADVANCED_FIXED_3D

        if data_2d_folder is None:
            # data_2d_folder = PREDS_2D
            data_2d_folder = PREDS_FIXED_2D
            # data_2d_folder = PREDS_ADVANCED_FIXED_2D

    else:
        if log_data is None:
            log_data = pd.read_csv(EVAL_LOG_PATH)

        if data_3d_folder is None:
            data_3d_folder = EVALS_3D

        if data_2d_folder is None:
            data_2d_folder = EVALS_2D

    # Get filepaths (based on folder)
    # data_3d_filepaths = pathlib.Path(data_3d_folder).glob(f"{data_3d_stem}_*.*")
    # data_3d_filepaths = sorted(data_3d_filepaths)

    # Get filepath (based on csv)
    data_3d_cube_filepaths = []
    col_0 = log_data.columns[0]
    for row_idx, row in log_data.iterrows():
        # Skip non relevant rows
        if data_3d_stem != str(row[col_0]).rsplit("_", maxsplit=1)[0]:
            continue

        data_3d_cube_filepath = list(pathlib.Path(data_3d_folder).glob(f"{row[col_0]}.*"))[0]
        data_3d_cube_filepaths.append(data_3d_cube_filepath)

    # START #
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Started Predict... "
        f"(Timestamp: {start_timestamp})"
    )

    # Single-threading - Sequential
    # for data_3d_cube_filepath in tqdm(data_3d_cube_filepaths):
    #     single_predict(
    #         args = args,
    #         data_3d_filepath=data_3d_cube_filepath,
    #         data_3d_folder=data_3d_folder,
    #         data_2d_folder=data_2d_folder,
    #         log_data=log_data,
    #         enable_debug=True,
    #         run_2d_flow=run_2d_flow,
    #         run_3d_flow=run_3d_flow,
    #         export_2d=export_2d,
    #         export_3d=export_3d
    #     )

    # Multi-threading
    futures = []
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        for data_3d_cube_filepath in data_3d_cube_filepaths:
            futures.append(
                executor.submit(
                    single_predict,
                    args=args,
                    data_3d_filepath=data_3d_cube_filepath,
                    data_3d_folder=data_3d_folder,
                    data_2d_folder=data_2d_folder,
                    log_data=log_data,
                    enable_debug=False,
                    run_2d_flow=run_2d_flow,
                    run_3d_flow=run_3d_flow,
                    export_2d=export_2d,
                    export_3d=export_3d
                )
            )
        # "Join" on all tasks by waiting for each future to complete.
        for future in tqdm(futures):
            future.result()  # This will block until the future is done.

    end_time = datetime.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Completed Predict... "
        f"(Timestamp: {end_timestamp}, Full Predict Time Elapsed: {end_time - start_time})"
    )


def full_merge(data_3d_stem, data_type: DataType, log_data=None, source_data_3d_folder=None):
    merge_with_source = MERGE_WITH_SOURCE

    # LOAD DATA #
    if data_type == DataType.TRAIN:
        if log_data is None:
            log_data = pd.read_csv(TRAIN_LOG_PATH)

        if source_data_3d_folder is None:
            source_data_3d_folder = PREDS
            # source_data_3d_folder = PREDS_FIXED
            # source_data_3d_folder = PREDS_ADVANCED_FIXED

    else:
        if log_data is None:
            log_data = pd.read_csv(EVAL_LOG_PATH)

        if source_data_3d_folder is None:
            source_data_3d_folder = EVALS

    # Input 3D object
    data_3d_filepath = list(pathlib.Path(source_data_3d_folder).glob(f"{data_3d_stem}*"))
    if len(data_3d_filepath) == 1:
        input_filepath = data_3d_filepath[0]
    else:
        raise ValueError(f"Expected 1 input files for '{data_3d_stem}' but got '{len(data_3d_filepath)}'.")

    # Pipeline Predicts
    predict_folder = PREDICT_PIPELINE_RESULTS_PATH
    predict_filepaths = sorted(pathlib.Path(os.path.join(predict_folder, "output_3d")).glob(f"{data_3d_stem}_*_output.*"))
    # predict_filepaths = sorted(list(pathlib.Path(PREDS_FIXED_3D).glob(f"{data_3d_stem}*.*")))

    # Pipeline Merge output path
    output_folder = MERGE_PIPELINE_RESULTS_PATH

    # os.makedirs(output_folder, exist_ok=True)

    input_data = convert_data_file_to_numpy(data_filepath=input_filepath, apply_data_threshold=True)
    if not merge_with_source:
        input_data = np.zeros_like(input_data)

    first_column = log_data.columns[0]
    regex_pattern = f"{data_3d_stem}_.*"
    matching_rows = log_data[log_data[first_column].str.contains(regex_pattern, regex=True, na=False)]

    # START #
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Started Merge... "
        f"(Timestamp: {start_timestamp})"
    )

    # Process the matching rows
    for row_idx, (_, row) in enumerate(matching_rows.iterrows()):
        predict_filepath = predict_filepaths[row_idx]
        predict_data = convert_data_file_to_numpy(data_filepath=predict_filepath, apply_data_threshold=True)

        start_x, end_x, start_y, end_y, start_z, end_z = (
            row["start_x"], row["end_x"], row["start_y"], row["end_y"], row["start_z"], row["end_z"]
        )

        size_x, size_y, size_z = end_x - start_x, end_y - start_y, end_z - start_z

        # Perform the logical OR operation on the specific region
        input_data[start_x:end_x, start_y:end_y, start_z:end_z] = np.logical_or(
            input_data[start_x:end_x, start_y:end_y, start_z:end_z],
            predict_data[:size_x, :size_y, :size_z]
        )

    # Save the final result
    save_filename = os.path.join(output_folder, data_3d_stem)
    convert_numpy_to_data_file(
        numpy_data=input_data,
        source_data_filepath=input_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )

    end_time = datetime.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Completed Merge... "
        f"(Timestamp: {end_timestamp}, Full Merge Time Elapsed: {end_time - start_time})"
    )

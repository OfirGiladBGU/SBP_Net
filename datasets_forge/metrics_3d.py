import os
import pathlib
import numpy as np
from statistics import mean
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from configs.configs_parser import DATA_PATH
from datasets.dataset_utils import convert_data_file_to_numpy, get_data_file_stem, connected_components_3d
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d

DATASET_PATH = DATA_PATH.joinpath("PipeForge3DPCD")

"""
This script calculates metrics of the holes in the 3D data files in a specified dataset.
"""


def calculate_metrics_3d(data_3d_stem, input_folder, target_folder, apply_abs=True):
    # Input
    input_filepath = list(pathlib.Path(input_folder).glob(f"{data_3d_stem}*.*"))[0]

    # Ground Truth
    target_filepath = list(pathlib.Path(target_folder).glob(f"{data_3d_stem}*.*"))[0]

    #############
    # Load Data #
    #############
    input_data_3d = convert_data_file_to_numpy(data_filepath=input_filepath, apply_data_threshold=True)
    target_data_3d = convert_data_file_to_numpy(data_filepath=target_filepath, apply_data_threshold=True)

    if apply_abs:
        # delta_binary = (abs(target_data_3d - input_data_3d) > 0.5).astype(np.int16)
        delta_binary = np.logical_xor(target_data_3d, input_data_3d).astype(np.int16)
    else:
        delta_binary = ((target_data_3d - input_data_3d) > 0.5).astype(np.int16)

    connectivity_type = 26

    # Identify connected components in delta_binary
    delta_labeled, delta_num_components = connected_components_3d(
        data_3d=delta_binary,
        connectivity_type=connectivity_type
    )

    # Iterate through connected components in delta_binary
    results_list = {
        "Hole Count": [delta_num_components],
        "Mean Hole Size": []
    }
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.int16)
        results_list["Mean Hole Size"].append(np.sum(component_mask))

    output_dict = {}
    for key in results_list.keys():
        output_dict[key] = mean(results_list[key]) if results_list[key] else 0

    # Print results
    print_str = "Stats:\n"
    for key, value in output_dict.items():
        print_str += f"AVG {key}: {value}\n"
    print(print_str)
    return output_dict


def main():
    apply_abs = True

    # Baseline data
    target_folder = DATASET_PATH.joinpath("labels")

    # Input data
    input_folder = DATASET_PATH.joinpath("preds")
    # input_folder = DATASET_PATH.joinpath("preds_fixed")

    # Get 3d data filepaths
    data_3d_filepaths = list(input_folder.rglob("*.*"))
    data_3d_stem_list = []
    for data_filepath in data_3d_filepaths:
        data_3d_stem_list.append(get_data_file_stem(data_filepath=data_filepath, relative_to=input_folder))
    data_3d_stem_count = len(data_3d_stem_list)

    # Calculate metrics for each 3D data file
    outputs = {}
    for idx, data_3d_stem in enumerate(data_3d_stem_list):
        print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Predicting...")
        output = calculate_metrics_3d(
            data_3d_stem=data_3d_stem,
            input_folder=input_folder,
            target_folder=target_folder,
            apply_abs=apply_abs
        )

        for key, value in output.items():
            if key in outputs:
                outputs[key].append(value)
            else:
                outputs[key] = [value]

    output_str = "[AVG RESULTS]\n"
    for key, value in outputs.items():
        output_str += f"AVG {key}: {mean(value)}\n"
    print(output_str)


if __name__ == '__main__':
    main()
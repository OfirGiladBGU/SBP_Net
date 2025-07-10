import argparse
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime

from configs.configs_parser import *
from datasets.dataset_utils import *
from evaluator.predict_pipeline import init_pipeline_models, single_predict, full_merge
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


##################
# Core Functions #
##################
def test_single_predict():
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
    data_3d_filepath = pathlib.Path(data_3d_folder).joinpath(base_filename)
    single_predict(
        args=args,
        data_3d_filepath=data_3d_filepath,
        data_3d_folder=data_3d_folder,
        data_2d_folder=data_2d_folder,
        log_data=log_data
    )


def full_predict(data_3d_stem, data_type: DataType, log_data=None, data_3d_folder=None, data_2d_folder=None,
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
    #         data_3d_filepath=data_3d_cube_filepath,
    #         data_3d_folder=data_3d_folder,
    #         data_2d_folder=data_2d_folder,
    #         log_data=log_data,
    #         enable_debug=True
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


def full_folder_predict(data_type: DataType):
    ################
    # Prepare Data #
    ################
    if data_type == DataType.TRAIN:
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        # source_data_3d_folder = PREDS
        source_data_3d_folder = PREDS_FIXED

        # data_3d_folder = LABELS_3D  # SANITY
        # data_3d_folder = PREDS_3D
        data_3d_folder = PREDS_FIXED_3D
        # data_3d_folder = PREDS_ADVANCED_FIXED_3D

        # data_2d_folder = LABELS_2D  # SANITY
        # data_2d_folder = PREDS_2D
        data_2d_folder = PREDS_FIXED_2D
        # data_2d_folder = PREDS_ADVANCED_FIXED_2D
    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        source_data_3d_folder = EVALS

        data_3d_folder = EVALS_3D

        data_2d_folder = EVALS_2D

    index_3d_uniques = log_data["index_3d"].unique()
    data_3d_stem_list = [data_3d_stem[1:] for data_3d_stem in index_3d_uniques]

    # Evaluate on Training - Test Data
    if data_3d_folder != EVALS_3D:
        split_percentage = 0.9
        index_3d_split_index = min(round(len(index_3d_uniques) * split_percentage), len(index_3d_uniques) - 1)

        # train_stems = data_3d_stem_list[:index_3d_split_index]
        test_stems = data_3d_stem_list[index_3d_split_index:]

        data_3d_stem_list = test_stems

    data_3d_stem_count = len(data_3d_stem_list)

    for idx, data_3d_stem in enumerate(data_3d_stem_list):
        print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Predicting...")
        full_predict(
            data_3d_stem=data_3d_stem,
            data_type=data_type,
            log_data=log_data,
            data_3d_folder=data_3d_folder,
            data_2d_folder=data_2d_folder,
            export_2d=False
        )

        print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Merging...")
        full_merge(
            data_3d_stem=data_3d_stem,
            data_type=data_type,
            log_data=log_data,
            source_data_3d_folder=source_data_3d_folder
        )


def main():
    init_pipeline_models(args=args)

    # TODO: Requires Model Init
    # test_single_predict()

    data_type = DataType.TRAIN
    # data_3d_stem = "PA000005"
    # data_3d_stem = "Hospital CUP 1in3"

    # full_predict(data_3d_stem=data_3d_stem, data_type=data_type, export_2d=False)
    # full_merge(data_3d_stem=data_3d_stem, data_type=data_type)

    full_folder_predict(data_type=data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run the prediction pipeline')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA predicting')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model-1d', type=str, default="", metavar='N',
                        help='Which 1D model to use')
    parser.add_argument('--input-size-model-1d', type=tuple, default=(1, 32, 32), metavar='N',
                        help='Which input size the 1D model should to use')
    parser.add_argument('--model-2d', type=str, default="", metavar='N',
                        help='Which 2D model to use')
    parser.add_argument('--input-size-model-2d', type=tuple, default=(1, 32, 32), metavar='N',
                        help='Which input size the 2D model should to use')
    parser.add_argument('--model-3d', type=str, default="", metavar='N',
                        help='Which 3D model to use')
    parser.add_argument('--input-size-model-3d', type=tuple, default=(1, 32, 32, 32), metavar='N',
                        help='Which input size the 3D model should to use')

    args = parser.parse_args()
    args.mode = "offline"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # Custom Edit:

    # args.model_1d = "vit_2d_to_1d"
    # args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_2d = "ae_6_2d_to_6_2d"
    # args.input_size_model_2d = (6, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    args.model_2d = "ae_2d_to_2d"
    args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_3d = "ae_3d_to_3d"
    # args.input_size_model_3d = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

    main()

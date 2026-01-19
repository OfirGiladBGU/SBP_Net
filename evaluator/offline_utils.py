import os
import pathlib
import torch
import numpy as np
from torchvision import transforms

from configs.configs_parser import BINARY_DILATION, IMAGES_6_VIEWS
from datasets.dataset_utils import *
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


def offline_preprocess_2d(data_3d_stem: str,
                          data_2d_folder: str,
                          apply_batch_merge: bool = False) -> torch.Tensor:
    data_2d_stem = f"{data_3d_stem}_<VIEW>"

    # # Get relative path parts
    # relative_filepath = data_3d_filepath.relative_to(CROPS_PATH)
    # relative_filepath_parts = list(relative_filepath.parts)
    #
    # # Update relative path parts to the relevant 2D images path
    # relative_filepath_parts[0] = relative_filepath_parts[0].replace("3d", "2d")
    # relative_filepath_parts[-1] = f"{data_2d_stem}.png"
    # format_of_2d_images_relative_filepath = pathlib.Path(*relative_filepath_parts)
    # format_of_2d_images = os.path.join(CROPS_PATH, format_of_2d_images_relative_filepath)

    format_of_2d_images = os.path.join(data_2d_folder, f"{data_2d_stem}.png")

    # Projections 2D
    data_2d_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        image_data = convert_data_file_to_numpy(data_filepath=image_path)
        torch_image = transforms.ToTensor()(image_data)
        if apply_batch_merge is True:
            torch_image = torch_image.squeeze(0)
        data_2d_list.append(torch_image)

    # Shape: (1, 6, w, h)
    if apply_batch_merge is True:
        data_2d_input = torch.stack(data_2d_list).unsqueeze(0)
    # Shape: (6, 1, w, h)
    else:
        data_2d_input = torch.stack(data_2d_list)
    return data_2d_input


def offline_preprocess_3d(data_3d_filepath: str,
                          data_2d_output: torch.Tensor = None,
                          apply_threshold_2d: bool = False,
                          threshold_2d: float = 0.2,
                          apply_fusion: bool = False,
                          apply_noise_filter_3d: bool = False,
                          hard_noise_filter_3d: bool = True,
                          connectivity_type_3d: int = 6) -> torch.Tensor:
    pred_3d = convert_data_file_to_numpy(data_filepath=data_3d_filepath, apply_data_threshold=True)

    # 2D flow was disabled
    if data_2d_output is None:
        data_3d_input = pred_3d

    # Use 2D flow results
    else:
        data_2d_output = data_2d_output.cpu().numpy()

        # TODO: Threshold
        if apply_threshold_2d:
            apply_threshold(data_2d_output, threshold=threshold_2d, keep_values=True)

        # Reconstruct 3D
        data_3d_list = list()
        for idx, image_view in enumerate(IMAGES_6_VIEWS):
            numpy_image = data_2d_output[idx] * 255
            data_3d = reverse_rotations(numpy_image=numpy_image, view_type=image_view, source_data_filepath=data_3d_filepath)
            data_3d_list.append(data_3d)

        data_3d_reconstruct = data_3d_list[0]
        for i in range(1, len(data_3d_list)):
            data_3d_reconstruct = np.logical_or(data_3d_reconstruct, data_3d_list[i])
        data_3d_reconstruct = data_3d_reconstruct.astype(np.float32)
        apply_threshold(data_3d_reconstruct, threshold=0.5, keep_values=False)

        # Fusion 3D
        if apply_fusion is True:
            data_3d_fusion = np.logical_or(data_3d_reconstruct, pred_3d)
            data_3d_fusion = data_3d_fusion.astype(np.float32)
            apply_threshold(data_3d_fusion, threshold=0.5, keep_values=False)
            data_3d_input = data_3d_fusion
        else:
            data_3d_input = data_3d_reconstruct

        if apply_noise_filter_3d is True:
            # data_3d_input = naive_noise_filter(data_3d_original=pred_3d, data_3d_input=data_3d_input)

            # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            #     data_3d_input = components_continuity_3d_single_component(
            #         label_cube=pred_3d,
            #         pred_advanced_fixed_cube=data_3d_input,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_3d,
            #         hard_condition=hard_noise_filter_3d,
            #         apply_dilation_scope=BINARY_DILATION
            #     )
            # elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
            #     data_3d_input = components_continuity_3d_local_connectivity(
            #         label_cube=pred_3d,
            #         pred_advanced_fixed_cube=data_3d_input,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_3d,
            #         hard_condition=hard_noise_filter_3d,
            #         apply_dilation_scope=BINARY_DILATION
            #     )
            # else:
            #     pass

            data_3d_input = components_continuity_3d_local_connectivity(
                label_cube=pred_3d,
                pred_advanced_fixed_cube=data_3d_input,
                reverse_mode=False,
                connectivity_type=connectivity_type_3d,
                hard_condition=hard_noise_filter_3d,
                apply_dilation_scope=BINARY_DILATION
            )

    data_3d_input = torch.Tensor(data_3d_input).unsqueeze(0).unsqueeze(0)
    return data_3d_input
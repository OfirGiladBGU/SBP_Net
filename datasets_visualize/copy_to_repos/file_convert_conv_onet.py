import os
import pathlib
import sys
root_path = str(pathlib.Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_path)

from datasets.dataset_utils import get_data_file_extension, convert_data_file_to_numpy, convert_numpy_to_data_file, get_data_file_stem


def convert():
    """
    Convert input from one format to another
    """
    numpy_data = convert_data_file_to_numpy(
        data_filepath=data_filepath,
        **src_kwargs
    )

    source_extension = get_data_file_extension(data_filepath=source_data_filepath)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        input_stem = get_data_file_stem(data_filepath=data_filepath)
        save_filename = str(pathlib.Path(save_dir).joinpath(f"{input_stem}{source_extension}"))
    else:
        input_extension = get_data_file_extension(data_filepath=data_filepath)
        save_filename = data_filepath.replace(input_extension, source_extension)

    convert_numpy_to_data_file(
        numpy_data=numpy_data,
        source_data_filepath=source_data_filepath,
        save_filename=save_filename,
        **dst_kwargs
    )


if __name__ == '__main__':
    # Input
    # src_folder = f"{root_path}/data/parse2022/labels"
    # dst_folder = f"{root_path}/../convolutional_occupancy_networks/data/parse_labels"

    src_folder = f"{root_path}/data/parse2022/preds_fixed"
    save_dir = f"{root_path}/../convolutional_occupancy_networks/data/parse_preds_fixed"

    filepaths = sorted(pathlib.Path(src_folder).glob("*.*"))
    for idx, fp in enumerate(filepaths):
        if idx >= 50:
            break
        print(f"Converting file {fp} ({idx+1}/{len(filepaths)})")

        data_filepath = fp

        source_data_filepath = r"dummy.obj"
        # source_data_filepath = r"dummy.pcd"

        src_kwargs = dict(
            # Mesh
            mesh_scale=1.0,
            voxel_size=1.0,

            # PCD
            # points_scale=1.0,
            # voxel_size=1.0
        )

        dst_kwargs = dict(
            # Mesh
            mesh_scale=1.0,
            voxel_size=1.0

            # PCD
            # points_scale=0.25,
            # voxel_size=1.0
        )

        convert()


if __name__ == '__main__':
    # src_folder = f"{root_path}/data/parse2022/labels"
    # dst_folder = f"{root_path}/../convolutional_occupancy_networks/data/parse_labels"

    src_folder = f"{root_path}/data/parse2022/preds_fixed"
    save_dir = f"{root_path}/../convolutional_occupancy_networks/data/parse_preds_fixed"

    src_files = sorted(pathlib.Path(src_folder).glob("*.*"))
    idx = 0
    for src_f in src_files:
        idx += 1
        print(f"Processing file {idx}: {src_f}")
        if idx > 50:
            break

        # Input
        data_filepath = src_f
        base_name = data_filepath.stem
        source_data_filepath = "dummy.npz"
        save_filename = dst_folder + f"/{base_name}.npz"

        # if os.path.exists(save_filename):
        #     print(f"File already exists: {save_filename}")
        #     continue

        # source_data_filepath = r"dummy.pcd"

        src_kwargs = dict(
            # Mesh
            mesh_scale=1.0,
            voxel_size=1.0,

            # PCD
            # points_scale=1.0,
            # voxel_size=1.0
        )

        dst_kwargs = dict(
            # Mesh
            # mesh_scale=0.25,
            mesh_scale=1.0,
            voxel_size=1.0

            # PCD
            # points_scale=0.25,
            # voxel_size=1.0
        )

        convert2()
        # break

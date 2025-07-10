import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from datasets.dataset_utils import get_data_file_extension, convert_data_file_to_numpy, convert_numpy_to_data_file


def convert():
    numpy_data = convert_data_file_to_numpy(
        data_filepath=data_filepath,
        **src_kwargs
    )

    input_extension = get_data_file_extension(data_filepath=data_filepath)
    source_extension = get_data_file_extension(data_filepath=source_data_filepath)
    save_filename = data_filepath.replace(input_extension, source_extension)
    convert_numpy_to_data_file(
        numpy_data=numpy_data,
        source_data_filepath=source_data_filepath,
        save_filename=save_filename,
        **dst_kwargs
    )


if __name__ == '__main__':
    # Input
    data_filepath = r".\output.npy"
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
        mesh_scale=0.25,
        voxel_size=1.0

        # PCD
        # points_scale=0.25,
        # voxel_size=1.0
    )

    convert()

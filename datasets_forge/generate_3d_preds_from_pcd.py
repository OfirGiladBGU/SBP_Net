import os
import pathlib
from tqdm import tqdm
import numpy as np
import random
from scipy.ndimage import convolve, label, rotate
import math
import shutil

from configs.configs_parser import DATA_PATH
from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file, get_data_file_stem
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


##################
# Generate Preds #
##################

# ASSUMPTION: NONE
def generate_sphere_holes(numpy_data: np.ndarray):
    # SPHERE: Random place, controllable hole size and no checking for new connected components

    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            radius = random.randint(3, 5)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]  # Get the coordinates

            for i in range(max(0, x - radius), min(numpy_data.shape[0], x + radius + 1)):
                for j in range(max(0, y - radius), min(numpy_data.shape[1], y + radius + 1)):
                    for k in range(max(0, z - radius), min(numpy_data.shape[2], z + radius + 1)):
                        if (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2 <= radius ** 2:
                            numpy_data[i, j, k] = 0

    return numpy_data


# ASSUMPTION: NONE
def generate_plane_holes(numpy_data: np.ndarray):
    # PLANE: Random all directions, controllable hole size and no checking for new connected components

    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            size = random.randint(0, 1)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]  # Get the coordinates

            # Define cube boundaries
            x_min = max(0, x - size)
            x_max = min(numpy_data.shape[0], x + size + 1)
            y_min = max(0, y - size)
            y_max = min(numpy_data.shape[1], y + size + 1)
            z_min = max(0, z - size)
            z_max = min(numpy_data.shape[2], z + size + 1)

            # Set the cube to black
            numpy_data[x_min:x_max, :, :] = 0  # Modify along the YZ planes
            numpy_data[:, y_min:y_max, :] = 0  # Modify along the XZ planes
            numpy_data[:, :, z_min:z_max] = 0  # Modify along the XY planes

            # Set all points on the same x, y, and z planes to black
            numpy_data[x, :, :] = 0  # Set all points on the plane parallel to YZ to black
            numpy_data[:, y, :] = 0  # Set all points on the plane parallel to XZ to black
            numpy_data[:, :, z] = 0  # Set all points on the plane parallel to XY to black


# ASSUMPTION: NONE
def generate_plane_holes_v2(numpy_data: np.ndarray):
    # PLANE: Random box selection, 1 pixel hole size and no checking for new connected components

    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]

            # Define a random cube size
            size = random.randint(5, 10)  # Random size for the cube

            # Define cube boundaries
            x_min = max(0, x - size)
            x_max = min(numpy_data.shape[0], x + size + 1)
            y_min = max(0, y - size)
            y_max = min(numpy_data.shape[1], y + size + 1)
            z_min = max(0, z - size)
            z_max = min(numpy_data.shape[2], z + size + 1)

            # Select a random plane axis and angle
            plane_axis = random.choice(["XY", "YZ", "XZ"])
            angle = random.choice([45, 90])

            # Create a cube area
            cube = numpy_data[x_min:x_max, y_min:y_max, z_min:z_max]

            # Apply a plane crop inside the cube
            for i in range(cube.shape[0]):
                for j in range(cube.shape[1]):
                    for k in range(cube.shape[2]):
                        # Translate local cube coordinates to global coordinates
                        global_x = x_min + i
                        global_y = y_min + j
                        global_z = z_min + k

                        # Check if the point lies on the plane
                        if plane_axis == "XY":
                            if angle == 90:
                                if global_x == x or global_y == y:
                                    numpy_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if math.isclose(global_x - x, global_y - y, abs_tol=1):
                                    numpy_data[global_x, global_y, global_z] = 0

                        elif plane_axis == "YZ":
                            if angle == 90:
                                if global_y == y or global_z == z:
                                    numpy_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if math.isclose(global_y - y, global_z - z, abs_tol=1):
                                    numpy_data[global_x, global_y, global_z] = 0

                        elif plane_axis == "XZ":
                            if angle == 90:
                                if global_x == x or global_z == z:
                                    numpy_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if math.isclose(global_x - x, global_z - z, abs_tol=1):
                                    numpy_data[global_x, global_y, global_z] = 0

    return numpy_data


# ASSUMPTION: The structure is a single connected component
def generate_plane_holes_v4(numpy_data: np.ndarray):
    # PLANE: Random box selection and controllable hole size

    num_of_centers = 10
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            plane_thickness = random.randint(3, 5)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]

            # Define a random cube size
            size = random.randint(10, 20)  # Random size for the cube

            # Define cube boundaries
            x_min = max(0, x - size)
            x_max = min(numpy_data.shape[0], x + size + 1)
            y_min = max(0, y - size)
            y_max = min(numpy_data.shape[1], y + size + 1)
            z_min = max(0, z - size)
            z_max = min(numpy_data.shape[2], z + size + 1)

            # Select a random plane axis and angle
            plane_axis = random.choice(["XY", "YZ", "XZ"])
            angle = random.choice([45, 90])

            # Create a cube area
            cube = numpy_data[x_min:x_max, y_min:y_max, z_min:z_max]

            # Copy the original data for testing
            test_data = numpy_data.copy()

            # Apply a plane crop inside the cube
            for i in range(cube.shape[0]):
                for j in range(cube.shape[1]):
                    for k in range(cube.shape[2]):
                        # Translate local cube coordinates to global coordinates
                        global_x = x_min + i
                        global_y = y_min + j
                        global_z = z_min + k

                        # Check if the point lies on the plane
                        if plane_axis == "XY":
                            if angle == 90:
                                if abs(global_x - x) < plane_thickness or abs(global_y - y) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if abs(global_x - x - (global_y - y)) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0

                        elif plane_axis == "YZ":
                            if angle == 90:
                                if abs(global_y - y) < plane_thickness or abs(global_z - z) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if abs(global_y - y - (global_z - z)) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0

                        elif plane_axis == "XZ":
                            if angle == 90:
                                if abs(global_x - x) < plane_thickness or abs(global_z - z) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0
                            elif angle == 45:
                                if abs(global_x - x - (global_z - z)) < plane_thickness:
                                    test_data[global_x, global_y, global_z] = 0

            numpy_data = test_data  # Apply the crop

    return numpy_data


def generate_plane_holes_v6(numpy_data: np.ndarray):
    # PLANE: Random box selection and controllable hole size without overlapping holes

    num_of_centers = 10
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points
    created_holes = []

    def is_overlapping(new_hole, existing_holes):
        for hole in existing_holes:
            if not (
                new_hole[0][1] < hole[0][0] or new_hole[0][0] > hole[0][1] or
                new_hole[1][1] < hole[1][0] or new_hole[1][0] > hole[1][1] or
                new_hole[2][1] < hole[2][0] or new_hole[2][0] > hole[2][1]
            ):
                return True
        return False

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            plane_thickness = random.randint(2, 3)

            success = False
            while not success:
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                x, y, z = random_point[0], random_point[1], random_point[2]

                # Define a random cube size
                size = random.randint(5, 10)  # Random size for the cube

                # Define cube boundaries
                x_min = max(0, x - size)
                x_max = min(numpy_data.shape[0], x + size + 1)
                y_min = max(0, y - size)
                y_max = min(numpy_data.shape[1], y + size + 1)
                z_min = max(0, z - size)
                z_max = min(numpy_data.shape[2], z + size + 1)

                new_hole = ((x_min, x_max), (y_min, y_max), (z_min, z_max))

                if is_overlapping(new_hole, created_holes):
                    continue

                # Select a random plane axis and angle
                plane_axis = random.choice(["XY", "YZ", "XZ"])
                angle = random.choice([45, 90])

                # Copy the original data for testing
                test_data = numpy_data.copy()

                # Apply a plane crop inside the cube
                for i in range(x_min, x_max):
                    for j in range(y_min, y_max):
                        for k in range(z_min, z_max):
                            if plane_axis == "XY":
                                if angle == 90:
                                    if abs(i - x) < plane_thickness or abs(j - y) < plane_thickness:
                                        test_data[i, j, k] = 0
                                elif angle == 45:
                                    if abs(i - x - (j - y)) < plane_thickness:
                                        test_data[i, j, k] = 0

                            elif plane_axis == "YZ":
                                if angle == 90:
                                    if abs(j - y) < plane_thickness or abs(k - z) < plane_thickness:
                                        test_data[i, j, k] = 0
                                elif angle == 45:
                                    if abs(j - y - (k - z)) < plane_thickness:
                                        test_data[i, j, k] = 0

                            elif plane_axis == "XZ":
                                if angle == 90:
                                    if abs(i - x) < plane_thickness or abs(k - z) < plane_thickness:
                                        test_data[i, j, k] = 0
                                elif angle == 45:
                                    if abs(i - x - (k - z)) < plane_thickness:
                                        test_data[i, j, k] = 0

                # Skip if the new hole creates new connected components
                numpy_data = test_data  # Apply the crop
                created_holes.append(new_hole)
                success = True

    return numpy_data


# Create new 'preds' folder with holes in numpy data
def convert_labels_data_to_preds_data(save_as_npy: bool = False):
    """
    Converts the labels data to preds data with holes in numpy array, and then save the result in preds folder.
    :param save_as_npy:
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
        # generate_sphere_holes(numpy_data=numpy_data)
        # generate_plane_holes(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v2(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v4(numpy_data=numpy_data)
        numpy_data = generate_plane_holes_v6(numpy_data=numpy_data)

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

    convert_originals_data_to_labels_data(save_as_npy=True, points_scale=points_scale, voxel_size=voxel_size,
                                          increase_density=increase_density)
    convert_labels_data_to_preds_data(save_as_npy=True)
    # clone_preds_as_evals()


if __name__ == '__main__':
    main()

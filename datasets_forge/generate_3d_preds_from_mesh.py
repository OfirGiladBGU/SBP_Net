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
from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file, get_data_file_stem, connected_components_3d
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
def generate_plane_holes_v3(numpy_data: np.ndarray):
    # PLANE: Random box selection and 1 pixel hole size

    num_of_centers = 10
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            success = False
            attempts = 0

            while not success and attempts < 10:  # Retry up to 10 times
                attempts += 1

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
                                    if global_x == x or global_y == y:
                                        test_data[global_x, global_y, global_z] = 0
                                elif angle == 45:
                                    if math.isclose(global_x - x, global_y - y, abs_tol=1):
                                        test_data[global_x, global_y, global_z] = 0

                            elif plane_axis == "YZ":
                                if angle == 90:
                                    if global_y == y or global_z == z:
                                        test_data[global_x, global_y, global_z] = 0
                                elif angle == 45:
                                    if math.isclose(global_y - y, global_z - z, abs_tol=1):
                                        test_data[global_x, global_y, global_z] = 0

                            elif plane_axis == "XZ":
                                if angle == 90:
                                    if global_x == x or global_z == z:
                                        test_data[global_x, global_y, global_z] = 0
                                elif angle == 45:
                                    if math.isclose(global_x - x, global_z - z, abs_tol=1):
                                        test_data[global_x, global_y, global_z] = 0

                # Check if the crop created new connected components
                (_, original_components) = connected_components_3d(data_3d=numpy_data, connectivity_type=26)
                (_, new_components) = connected_components_3d(data_3d=test_data, connectivity_type=26)

                if new_components > original_components:
                    numpy_data = test_data  # Apply the crop
                    success = True

    return numpy_data


# ASSUMPTION: The structure is a single connected component
def generate_plane_holes_v4(numpy_data: np.ndarray):
    # PLANE: Random box selection and controllable hole size

    num_of_centers = 10
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            plane_thickness = random.randint(1, 2)

            success = False
            attempts = 0

            while not success and attempts < 10:  # Retry up to 10 times
                attempts += 1

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

                # Check if the crop created new connected components
                (_, original_components) = connected_components_3d(data_3d=numpy_data, connectivity_type=26)
                (_, new_components) = connected_components_3d(data_3d=test_data, connectivity_type=26)

                if new_components > original_components:
                    numpy_data = test_data  # Apply the crop
                    success = True

    return numpy_data


# ASSUMPTION: LOCAL DISCONNECTION
def generate_plane_holes_v5(numpy_data: np.ndarray):
    # PLANE: Random box selection and controllable hole size

    num_of_centers = 10
    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            plane_thickness = random.randint(1, 2)

            success = False
            # attempts = 0

            while not success: # and attempts < 10:  # Retry up to 10 times
                # attempts += 1

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

                # Check if the crop created new connected components in cube area
                (_, original_components) = connected_components_3d(data_3d=cube, connectivity_type=26)
                test_cube = test_data[x_min:x_max, y_min:y_max, z_min:z_max]
                (_, new_components) = connected_components_3d(data_3d=test_cube, connectivity_type=26)

                if new_components > original_components:
                    numpy_data = test_data  # Apply the crop
                    success = True

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
            plane_thickness = random.randint(1, 2)

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

                # Check if the crop created new connected components in cube area
                (_, original_components) = connected_components_3d(data_3d=numpy_data[x_min:x_max, y_min:y_max, z_min:z_max], connectivity_type=26)
                (_, new_components) = connected_components_3d(data_3d=test_data[x_min:x_max, y_min:y_max, z_min:z_max], connectivity_type=26)

                if new_components > original_components:
                    numpy_data = test_data  # Apply the crop
                    created_holes.append(new_hole)
                    success = True

    return numpy_data


# ASSUMPTION: The structure is a single connected component
def generate_box_holes(numpy_data: np.ndarray):
    # BOX: Random place, controllable hole size

    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)  # Identify parts

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            disconnected = False

            while not disconnected:
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                x, y, z = random_point

                # Randomly select box size to ensure meaningful disconnection
                size_x = random.randint(5, 6)  # Size along X-axis
                size_y = random.randint(5, 6)  # Size along Y-axis
                size_z = random.randint(5, 6)  # Size along Z-axis

                # Define box boundaries, ensuring they stay within array bounds
                x_min = max(0, x - size_x // 2)
                x_max = min(numpy_data.shape[0], x + size_x // 2 + 1)
                y_min = max(0, y - size_y // 2)
                y_max = min(numpy_data.shape[1], y + size_y // 2 + 1)
                z_min = max(0, z - size_z // 2)
                z_max = min(numpy_data.shape[2], z + size_z // 2 + 1)

                # Create a copy of the data to test the disconnection
                test_data = numpy_data.copy()
                test_data[x_min:x_max, y_min:y_max, z_min:z_max] = 0.0

                # Check if the number of connected components increases
                labeled, num_components = label(test_data > 0.5)
                original_components = label(numpy_data > 0.5)[1]

                if num_components > original_components:
                    # Apply the disconnection if it creates new components
                    numpy_data[x_min:x_max, y_min:y_max, z_min:z_max] = 0.0
                    disconnected = True

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
        # numpy_data = generate_plane_holes_v3(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v4(numpy_data=numpy_data)
        numpy_data = generate_plane_holes_v5(numpy_data=numpy_data)
        # numpy_data = generate_plane_holes_v6(numpy_data=numpy_data)
        # numpy_data = generate_box_holes(numpy_data=numpy_data)

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

    convert_originals_data_to_labels_data(save_as_npy=True, mesh_scale=mesh_scale, voxel_size=voxel_size)
    convert_labels_data_to_preds_data(save_as_npy=True)
    # clone_preds_as_evals()


if __name__ == '__main__':
    main()

import random
import math
import numpy as np
from scipy.ndimage import label
from datasets.dataset_utils import connected_components_3d


class Data_Type:
    MESH = "Mesh"
    PCD = "PCD"


#####################
# Archived Funcions #
#####################

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


#################
# Core Funcions #
#################

# ASSUMPTION: NONE
def generate_plane_holes_v1(numpy_data: np.ndarray, config: dict):
    # PLANE: Random all directions, controllable hole size and no checking for new connected components

    # data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 5)
    # plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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

    return numpy_data


# ASSUMPTION: NONE
def generate_plane_holes_v2(numpy_data: np.ndarray, config: dict):
    # PLANE: Random box selection, 1 pixel hole size and no checking for new connected components

    # data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 5)
    # plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]

            # Define a random cube size
            size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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
def generate_plane_holes_v3(numpy_data: np.ndarray, config: dict):
    # PLANE: Random box selection and 1 pixel hole size

    data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 10)
    # plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

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
                size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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

                # Handle data type specific checks for applying the crop
                if data_type == Data_Type.MESH:
                    # Check if the crop created new connected components
                    (_, original_components) = connected_components_3d(data_3d=numpy_data, connectivity_type=26)
                    (_, new_components) = connected_components_3d(data_3d=test_data, connectivity_type=26)

                    if new_components > original_components:
                        numpy_data = test_data  # Apply the crop
                        success = True
                elif data_type == Data_Type.PCD:
                    # For PCD, we skip the connected components check and directly apply the crop
                    numpy_data = test_data  # Apply the crop
                    success = True
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

    return numpy_data


# ASSUMPTION: The structure is a single connected component
def generate_plane_holes_v4(numpy_data: np.ndarray, config: dict):
    # PLANE: Random box selection and controllable hole size

    data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 10)
    plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            # NOTE: Control the hole size by plane thickness
            plane_thickness = random.randint(plane_thickness[0], plane_thickness[1])

            success = False
            attempts = 0

            while not success and attempts < 10:  # Retry up to 10 times
                attempts += 1

                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                x, y, z = random_point[0], random_point[1], random_point[2]

                # Define a random cube size
                size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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

                # Handle data type specific checks for applying the crop
                if data_type == Data_Type.MESH:
                    # Check if the crop created new connected components
                    (_, original_components) = connected_components_3d(data_3d=numpy_data, connectivity_type=26)
                    (_, new_components) = connected_components_3d(data_3d=test_data, connectivity_type=26)

                    if new_components > original_components:
                        numpy_data = test_data  # Apply the crop
                        success = True
                elif data_type == Data_Type.PCD:
                    # For PCD, we skip the connected components check and directly apply the crop
                    numpy_data = test_data  # Apply the crop
                    success = True
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

    return numpy_data


# ASSUMPTION: LOCAL DISCONNECTION
def generate_plane_holes_v5(numpy_data: np.ndarray, config: dict):
    # PLANE: Random box selection and controllable hole size

    data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 10)
    plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

    white_points = np.argwhere(numpy_data > 0.5)  # Find all white points

    if len(white_points) > 0:
        for _ in range(num_of_centers):
            # NOTE: Control the hole size by plane thickness
            plane_thickness = random.randint(plane_thickness[0], plane_thickness[1])

            success = False
            # attempts = 0

            while not success: # and attempts < 10:  # Retry up to 10 times
                # attempts += 1

                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                x, y, z = random_point[0], random_point[1], random_point[2]

                # Define a random cube size
                size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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

                # Handle data type specific checks for applying the crop
                if data_type == Data_Type.MESH:
                    # Check if the crop created new connected components in cube area
                    (_, original_components) = connected_components_3d(data_3d=cube, connectivity_type=26)
                    test_cube = test_data[x_min:x_max, y_min:y_max, z_min:z_max]
                    (_, new_components) = connected_components_3d(data_3d=test_cube, connectivity_type=26)

                    if new_components > original_components:
                        numpy_data = test_data  # Apply the crop
                        success = True
                elif data_type == Data_Type.PCD:
                    # For PCD, we skip the connected components check and directly apply the crop
                    numpy_data = test_data  # Apply the crop
                    success = True
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

    return numpy_data


def generate_plane_holes_v6(numpy_data: np.ndarray, config: dict):
    # PLANE: Random box selection and controllable hole size without overlapping holes

    data_type = config.get("data_type", None)
    num_of_centers = config.get("num_of_centers", 10)
    plane_thickness = config.get("plane_thickness", [1, 2])
    cube_size = config.get("cube_size", [5, 10])

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
            # NOTE: Control the hole size by plane thickness
            plane_thickness = random.randint(plane_thickness[0], plane_thickness[1])

            success = False
            while not success:
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                x, y, z = random_point[0], random_point[1], random_point[2]

                # Define a random cube size
                size = random.randint(cube_size[0], cube_size[1])  # Random size for the cube

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

                # Handle data type specific checks for applying the crop
                if data_type == Data_Type.MESH:
                    # Check if the crop created new connected components in cube area
                    (_, original_components) = connected_components_3d(data_3d=numpy_data[x_min:x_max, y_min:y_max, z_min:z_max], connectivity_type=26)
                    (_, new_components) = connected_components_3d(data_3d=test_data[x_min:x_max, y_min:y_max, z_min:z_max], connectivity_type=26)

                    if new_components > original_components:
                        numpy_data = test_data  # Apply the crop
                        created_holes.append(new_hole)
                        success = True
                elif data_type == Data_Type.PCD:
                    # For PCD, we skip the connected components check and directly apply the crop
                    numpy_data = test_data  # Apply the crop
                    success = True
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

    return numpy_data

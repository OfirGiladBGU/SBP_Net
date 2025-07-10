import os
import pathlib
from tqdm import tqdm
import numpy as np
import random
import cv2
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from configs.configs_parser import DATA_CROPS_PATH
from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file, get_data_file_stem

# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d

CROPS_PATH = os.path.join(DATA_CROPS_PATH, "parse2022_custom")

##################
# Generate Preds #
##################
def generate_circle_holes(numpy_data: np.ndarray):
    # CIRCLE: Random place, controllable hole size and no checking for new connected components

    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            radius = random.randint(3, 5)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y = random_point[0], random_point[1]  # Get the coordinates

            for i in range(max(0, x - radius), min(numpy_data.shape[0], x + radius + 1)):
                for j in range(max(0, y - radius), min(numpy_data.shape[1], y + radius + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                        numpy_data[i, j] = 0

    return numpy_data


def generate_line_holes_v1(numpy_data: np.ndarray, line_width=1):
    # Get initial number of connected components
    num_labels, _ = cv2.connectedComponents((numpy_data > 0).astype(np.uint8))

    tries = 10
    current_try = 0

    # Find all points where the pixel value is greater than 0 (foreground)
    black_points = np.argwhere(numpy_data > 0)
    while True:
        if len(black_points) < 2:
            raise ValueError("Not enough black points to create a line cut.")

        if current_try >= tries:
            raise ValueError(f"Failed to create a line cut after {tries} tries.")

        # Select two random points to define the line
        random_point1 = tuple(random.choice(black_points))
        random_point2 = tuple(random.choice(black_points))

        # Create a mask for the line cut
        line_mask = np.zeros_like(numpy_data, dtype=np.uint8)
        cv2.line(line_mask, random_point1[::-1], random_point2[::-1], 255, thickness=line_width)

        # Apply the line cut
        modified_data = numpy_data.copy()
        modified_data[line_mask > 0] = 0

        # Check the number of connected components after the cut
        new_num_labels, _ = cv2.connectedComponents((modified_data > 0).astype(np.uint8))

        if new_num_labels == num_labels + 1:
            # If the cut created a new connected component, finalize the change
            return modified_data
        else:
            current_try += 1
            # Otherwise, undo the cut by continuing the loop
            continue


# Create new 'preds' folder with holes in numpy data
def convert_labels_data_to_preds_data():
    input_folder = os.path.join(CROPS_PATH, "labels_2d_v6")
    output_folder = os.path.join(CROPS_PATH, "preds_fixed_2d_v6")

    # os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]

        numpy_2d_data = convert_data_file_to_numpy(data_filepath=data_filepath)
        # numpy_2d_data = cv2.imread(str(data_filepath))
        # numpy_2d_data = cv2.cvtColor(numpy_2d_data, cv2.COLOR_BGR2GRAY)

        # Generate holes:
        # TODO: implement (Use different method)
        # generate_circle_holes(numpy_data=numpy_data)
        numpy_2d_data = generate_line_holes_v1(numpy_data=numpy_2d_data)

        # Save data:
        data_filepath_stem = get_data_file_stem(data_filepath=data_filepath, relative_to=input_folder)
        save_filename = os.path.join(output_folder, data_filepath_stem)

        convert_numpy_to_data_file(
            numpy_data=numpy_2d_data,
            source_data_filepath=data_filepath,
            save_filename=save_filename,
            apply_data_threshold=True
        )
        # cv2.imwrite(f"{save_filename}.png", numpy_2d_data)


def main():
    convert_labels_data_to_preds_data()


if __name__ == '__main__':
    main()

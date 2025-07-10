import os
import pathlib
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from configs.configs_parser import DATA_PATH


def apply_filters():
    # Load the image
    image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, ksize=gaussian_blur_ksize, sigmaX=sigma_x)

    # Apply Bilateral Filter
    bilateral_filter = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Apply Median Filter
    median_filter = cv2.medianBlur(image, ksize=median_blur_ksize)

    # Display the results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(gaussian_blur, cmap='gray')
    axes[1].set_title('Gaussian Blur')
    axes[1].axis('off')

    axes[2].imshow(bilateral_filter, cmap='gray')
    axes[2].set_title('Bilateral Filter')
    axes[2].axis('off')

    axes[3].imshow(median_filter, cmap='gray')
    axes[3].set_title('Median Filter')
    axes[3].axis('off')

    plt.show()


if __name__ == '__main__':
    image_filepath = os.path.join(DATA_PATH, r"PipeForge3DPCD\labels_2d\01_back.png")

    # Gaussian Blur parameters
    gaussian_blur_ksize = (5, 5)  # Kernel size for Gaussian Blur
    sigma_x = 0  # Standard deviation in X direction

    # Bilateral Filter parameters
    d = 9  # Diameter of pixel neighborhood
    sigma_color = 75  # Filter sigma in color space
    sigma_space = 75  # Filter sigma in coordinate space

    # Median Filter parameters
    median_blur_ksize = 5  # 3

    apply_filters()

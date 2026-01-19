#!/usr/bin/env python3
"""
Script to convert black pixels to white in PNG images.
Supports threshold parameter to control which pixels are considered "black".
"""

import argparse
import numpy as np
from PIL import Image
import os


def convert_black_to_white(image_path, output_path=None, threshold=50):
    """
    Convert black pixels to white in a PNG image.
    
    Args:
        image_path (str): Path to input PNG image
        output_path (str): Path for output image. If None, adds '_converted' suffix
        threshold (int): Threshold value (0-255). Pixels below this value are considered black
    
    Returns:
        str: Path to the output image
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert to RGBA if not already (to handle transparency)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Create mask for "black" pixels (RGB values below threshold)
    # For RGB channels only (ignore alpha)
    rgb_array = img_array[:, :, :3]
    
    # Pixels are considered black if ALL RGB values are below threshold
    black_mask = np.all(rgb_array < threshold, axis=2)
    
    # Convert black pixels to white (255, 255, 255)
    img_array[black_mask, :3] = 255
    
    # Convert back to PIL Image
    result_img = Image.fromarray(img_array, 'RGBA')
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_converted{ext}"
    
    # Save the result
    result_img.save(output_path)
    
    # Count converted pixels
    num_converted = np.sum(black_mask)
    total_pixels = black_mask.size
    
    print(f"Converted {num_converted} black pixels out of {total_pixels} total pixels")
    print(f"Output saved to: {output_path}")
    
    return output_path


def main(input_image, output_path=None, threshold=50, show_preview=False):
    """
    Main function to convert black pixels to white.
    
    Args:
        input_image (str): Path to input PNG image
        output_path (str): Output image path (default: adds '_converted' suffix to input)
        threshold (int): Threshold for black pixels (0-255)
        show_preview (bool): Whether to show before/after preview
    """
    # Validate inputs
    if not os.path.exists(input_image):
        print(f"Error: Input file '{input_image}' does not exist")
        return 1
    
    if not input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Warning: Input file is not a PNG/JPG. Results may vary.")
    
    if not (0 <= threshold <= 255):
        print("Error: Threshold must be between 0 and 255")
        return 1
    
    try:
        # Convert the image
        output_result = convert_black_to_white(
            input_image,
            output_path,
            threshold
        )
        
        # Show preview if requested
        if show_preview:
            show_preview_images(input_image, output_result)
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    return 0


def show_preview_images(input_path, output_path):
    """Show before/after preview using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        
        # Load images
        img_before = Image.open(input_path)
        img_after = Image.open(output_path)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(img_before)
        ax1.set_title("Before")
        ax1.axis('off')
        
        ax2.imshow(img_after)
        ax2.set_title("After")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error showing preview: {e}")


if __name__ == "__main__":
    # ==================== EDIT THESE PARAMETERS ====================
    
    # Input image path
    INPUT_IMAGE = r"combined_3d_results.png"
    
    # Output image path (None for auto-generated name with '_converted' suffix)
    OUTPUT_PATH = r"combined_3d_results_converted.png"  # or r"path/to/output.png"
    
    # Threshold for black pixels (0-255)
    # Lower values = only very dark pixels
    # Higher values = includes darker grays
    THRESHOLD = 10
    
    # Show before/after preview (requires matplotlib)
    SHOW_PREVIEW = False
    
    # ================================================================
    
    exit(main(INPUT_IMAGE, OUTPUT_PATH, THRESHOLD, SHOW_PREVIEW))
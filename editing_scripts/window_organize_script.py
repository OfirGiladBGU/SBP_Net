import os
from PIL import Image


def load_and_sort_images(folder):
    return sorted([(f, os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.png')])


def group_by_name(images):
    # groups = {}
    # for fname, path in images:
    #     base = fname.rsplit('_p', 1)[0]
    #     groups.setdefault(base, []).append((fname, path))
    groups = {'all': list(images)}
    return groups


def stack_rows(image_groups, white_space=5, target_size=(100, 100), crop_window=None):
    """Create a list of row images from grouped images.

    crop_window: optional tuple (x1, y1, x2, y2) in pixels to crop each resized image.
    The crop is applied after resizing. The box will be clamped to image bounds.
    """
    rows = []
    for name in sorted(image_groups.keys()):
        sorted_imgs = sorted(image_groups[name], key=lambda x: x[0])
        processed = []
        for _, p in sorted_imgs:
            img = Image.open(p).resize(target_size, Image.BICUBIC)
            if crop_window is not None:
                x1, y1, x2, y2 = crop_window
                # clamp to image bounds and use ints
                x1c = max(0, int(x1))
                y1c = max(0, int(y1))
                x2c = min(img.width, int(x2))
                y2c = min(img.height, int(y2))
                if x2c > x1c and y2c > y1c:
                    img = img.crop((x1c, y1c, x2c, y2c))
                # else: invalid window -> keep full resized image
            processed.append(img)

        if not processed:
            continue

        n_imgs = len(processed)
        row_height = max(im.height for im in processed)
        row_width = sum(im.width for im in processed) + (n_imgs - 1) * white_space
        row_img = Image.new('RGB', (row_width, row_height), (0, 0, 0))  # black background
        x_offset = 0
        for i, img in enumerate(processed):
            # vertically align at top; leave background for differing heights
            row_img.paste(img, (x_offset, 0))
            x_offset += img.width
            if i < n_imgs - 1:
                spacer = Image.new('RGB', (white_space, row_height), (255, 255, 255))
                row_img.paste(spacer, (x_offset, 0))
                x_offset += white_space
        rows.append(row_img)
    return rows


def compose_image(folders, white_space=5, red_line_height=5, row_spacing=5, target_size=(100, 100),
                  crop_window=None):
    all_rows = []
    for i, folder in enumerate(folders):
        images = load_and_sort_images(folder)
        grouped = group_by_name(images)
        rows = stack_rows(grouped, white_space, target_size, crop_window=crop_window)
        for j, row in enumerate(rows):
            all_rows.append(row)
            # Add horizontal white space between rows (but not after last row or after a red line)
            if j < len(rows) - 1:
                spacer = Image.new('RGB', (row.width, row_spacing), (255, 255, 255))
                all_rows.append(spacer)
        if i < len(folders) - 1:
            max_width = max(row.width for row in rows)
            red_line = Image.new('RGB', (max_width, red_line_height), (255, 255, 255))
            all_rows.append(red_line)
    final_width = max(row.width for row in all_rows)
    total_height = sum(row.height for row in all_rows)
    final_img = Image.new('RGB', (final_width, total_height), (0, 0, 0))
    y_offset = 0
    for row in all_rows:
        final_img.paste(row, (0, y_offset))
        y_offset += row.height
    return final_img


# Usage
folders = [
    r'E:\AllProjects\PycharmProjects\TreesAutoEncoder\ORDERED\PARSE',
    r'E:\AllProjects\PycharmProjects\TreesAutoEncoder\ORDERED\MESH',
    r'E:\AllProjects\PycharmProjects\TreesAutoEncoder\ORDERED\PCD'
]
output_image = compose_image(folders, target_size=(500, 500), crop_window=(0, 230, 150, 380))
output_image.save("sota_zoom.png")

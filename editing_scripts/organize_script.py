import os
from PIL import Image


def load_and_sort_images(folder):
    return sorted([(f, os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.png')])


def group_by_name(images):
    groups = {}
    for fname, path in images:
        base = fname.rsplit('_p', 1)[0]
        groups.setdefault(base, []).append((fname, path))
    # groups = {'all': list(images)}
    return groups


def stack_rows(image_groups, white_space=5, target_size=(100, 100)):
    rows = []
    for name in sorted(image_groups.keys()):
        sorted_imgs = sorted(image_groups[name], key=lambda x: x[0])
        imgs = [Image.open(p).resize(target_size, Image.BICUBIC) for _, p in sorted_imgs]
        n_imgs = len(imgs)
        row_width = n_imgs * target_size[0] + (n_imgs - 1) * white_space
        row_img = Image.new('RGB', (row_width, target_size[1]), (0, 0, 0))  # black background
        x_offset = 0
        for i, img in enumerate(imgs):
            row_img.paste(img, (x_offset, 0))
            x_offset += target_size[0]
            if i < n_imgs - 1:
                # Draw vertical white space
                for x in range(white_space):
                    for y in range(target_size[1]):
                        row_img.putpixel((x_offset + x, y), (255, 255, 255))
                x_offset += white_space
        rows.append(row_img)
    return rows


def compose_image(folders, white_space=5, red_line_height=5, row_spacing=5, target_size=(100, 100)):
    all_rows = []
    for i, folder in enumerate(folders):
        images = load_and_sort_images(folder)
        grouped = group_by_name(images)
        rows = stack_rows(grouped, white_space, target_size)
        for j, row in enumerate(rows):
            all_rows.append(row)
            # Add horizontal white space between rows (but not after last row or after a red line)
            if j < len(rows) - 1:
                spacer = Image.new('RGB', (row.width, row_spacing), (255, 255, 255))
                all_rows.append(spacer)
        if i < len(folders) - 1:
            max_width = max(row.width for row in rows)
            red_line = Image.new('RGB', (max_width, red_line_height), (255, 0, 0))
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
output_image = compose_image(folders)
output_image.save("output_collage.png")

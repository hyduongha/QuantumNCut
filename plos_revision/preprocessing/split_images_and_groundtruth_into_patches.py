import os
import numpy as np
from PIL import Image
import pandas as pd

# Read the label mask from a SEG file.
# Each data record contains: label, row, start column, and inclusive end column.
def load_seg_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_index = None
    width = height = 0

    # Read the image dimensions and locate the start of the data section.
    for i, line in enumerate(lines):
        if "data" in line.lower():
            data_index = i + 1
            break
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line.startswith("height"):
            height = int(line.split()[1])

    if data_index is None:
        raise ValueError("The 'data' section was not found in the SEG file!")

    labels = np.zeros((height, width), dtype=np.int32)

    # Expand each run into pixel labels. Label 0 is retained as a regular label.
    for line in lines[data_index:]:
        values = line.strip().split()
        if len(values) != 4:
            continue
        label, row, start_col, end_col = map(int, values)
        labels[row, start_col:end_col+1] = label

    return labels

# Write a patch label array to a SEG file using run-length encoding.
def save_seg_file(mask_array, file_path):
    h, w = mask_array.shape
    unique_labels = np.unique(mask_array)
    num_segments = len(unique_labels)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"width {w}\n")
        f.write(f"height {h}\n")
        f.write(f"segments {num_segments}\n")
        f.write("data\n")
        for row in range(h):
            col = 0
            while col < w:
                label = mask_array[row, col]
                start_col = col
                while col < w and mask_array[row, col] == label:
                    col += 1
                end_col = col-1
                f.write(f"{label} {row} {start_col} {end_col}\n")

def main():
    # Input directories: image files and matching SEG ground-truth files.
    image_path = r"F:\Nhu Y_khong xoa_2\groundtruth_images_MaSS13K"
    seg_path = r"F:\Nhu Y_khong xoa_2\groundtruth_masks_MaSS13K"
    # Output directories for image patches and ground-truth patches.
    image_out_dir = r"F:\Nhu Y_khong xoa_2\8x16_MaSS13K\split_images"
    mask_out_dir = r"F:\Nhu Y_khong xoa_2\8x16_MaSS13K\split_masks"
    if not os.path.isdir(image_path):
        print(f"❌ Directory {image_path} does not exist!")
        exit()

    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ No image files were found in {image_path}!")
        exit()

    for idx, file_name in enumerate(image_files, start=1):
        image_path_i = os.path.join(image_path, file_name)
        name, ext = os.path.splitext(file_name)
        seg_path_i = os.path.join(seg_path, name + ".seg")

        # Read the image and convert it to RGB.
        image = Image.open(image_path_i).convert("RGB")
        img_np = np.array(image)
        img_h, img_w = img_np.shape[:2]

        # Load the original ground-truth mask from the matching SEG file.
        mask_np = load_seg_file(seg_path_i)

        # Check that the ground-truth mask and image have identical dimensions.
        assert mask_np.shape == (img_h, img_w), "Ground truth and image dimensions do not match!"

        # Patch dimensions: width = 8 pixels and height = 16 pixels.
        patch_w, patch_h = 8, 16

        # Extract non-overlapping image/mask pairs, from left to right and top to bottom.
        count = 0
        for y in range(0, img_h, patch_h):
            for x in range(0, img_w, patch_w):
                patch_img = img_np[y:y+patch_h, x:x+patch_w]
                patch_mask = mask_np[y:y+patch_h, x:x+patch_w]

                # Skip incomplete patches at the right and bottom image boundaries.
                if patch_img.shape[:2] != (patch_h, patch_w):
                    continue
                # Determine the number of segments from the ground-truth patch,
                # including label 0. Append this number to both output filenames.
                unique_labels = np.unique(patch_mask)
                num_segments = len(unique_labels)
                os.makedirs(os.path.join(image_out_dir, name), exist_ok=True)
                os.makedirs(os.path.join(mask_out_dir, name), exist_ok=True)
                # Save paired patches with the same zero-padded patch index.
                Image.fromarray(patch_img).save(f"{image_out_dir}/{name}/{name}_{count:05d}_{num_segments}.png")
                save_seg_file(patch_mask, f"{mask_out_dir}/{name}/{name}_{count:05d}_{num_segments}.seg")
                count += 1

if __name__ == "__main__":
    main()

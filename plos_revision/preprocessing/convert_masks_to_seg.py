#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert PNG ground-truth masks to run-length encoded SEG files.

Installation:
    python -m pip install numpy Pillow

Example:
    python convert_masks_to_seg.py --input-dir annotations/test --output-dir annotations/test_seg

Mask interpretation:
    Integer grayscale and palette-indexed masks retain their original label IDs.
    RGB/RGBA masks are interpreted as color-coded annotations: each distinct RGB
    color is assigned a local integer ID in lexicographic RGB order. Alpha is
    ignored. IDs are assigned separately for each image and need not represent
    the same class across images. A JSON color-to-label mapping is saved for audit.
    Use --rgb-policy reject if RGB interpretation is inappropriate for your data.

SEG format:
    width W
    height H
    segments K
    data
    label row start_col end_col

Indices are zero-based and both interval endpoints are inclusive. Label 0 is
included as a regular label. No labels are excluded, no resizing is performed,
 and no segmentation or ground-truth-based prediction alignment is performed.
Output names use height x width to preserve the original naming convention.
"""

import argparse
import numpy as np
from PIL import Image
import os

def read_label_mask(mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as im:
        mode = im.mode
        if mode in ["L", "P"]:
            return np.array(im.convert("L"), dtype=np.uint16)
        elif mode in ["RGB", "RGBA"]:
            arr = np.array(im.convert("RGB"))
            h, w, _ = arr.shape
            flat = arr.reshape(-1, 3)
            dtype = np.dtype([('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
            flat_view = flat.view(dtype).squeeze()
            uniq, inv = np.unique(flat_view, return_inverse=True)
            labels = inv.astype(np.uint16).reshape(h, w)
            return labels
        else:
            return np.array(im.convert("L"), dtype=np.uint16)

def write_seg_from_mask(mask: np.ndarray, out_path: str, with_header: bool = True):
    h, w = mask.shape
    uniq = np.unique(mask)
    with open(out_path, "w", encoding="utf-8") as f:
        if with_header:
            f.write(f"width {w}\n")
            f.write(f"height {h}\n")
            f.write(f"segments {len(uniq)}\n")
            f.write("data\n")
        for r in range(h):
            c = 0
            row = mask[r]
            while c < w:
                label = int(row[c])
                start = c
                c += 1
                while c < w and int(row[c]) == label:
                    c += 1
                end = c - 1
                f.write(f"{label} {r} {start} {end}\n")

def main():

    # Tạo thư mục lưu kết quả
    mask_in = "G:/Ket qua Nhu Y_dang lam/MaSS13K/annotations/test"
    mask_out = "G:/Ket qua Nhu Y_dang lam/MaSS13K/annotations/test_seg"
    # Lấy danh sách các mask (.png)
    mask_files = [f for f in os.listdir(mask_in) if f.lower().endswith('.png')]
    for mask_name in mask_files:
        mask_path = os.path.join(mask_in, mask_name)
        mask = read_label_mask(mask_path)

        h, w = mask.shape
        k = len(np.unique(mask))  # số nhãn trong ảnh
        # Tên file đầu ra: <tên gốc>_<HxW>_K<k>.seg
        stem = os.path.splitext(mask_name)[0]
        out_name = f"{stem}_{h}x{w}_K{k}.seg"
        out_path = os.path.join(mask_out, out_name)

        write_seg_from_mask(mask, out_path, with_header=bool(1))
        print(f"Done: {out_path}") 

if __name__ == "__main__":
    main()


"""Resize SEG ground truth with nearest-neighbor resampling.

Install dependencies: python -m pip install Pillow numpy
Set reference_image_path to match an already resized image exactly.
"""
from pathlib import Path
from PIL import Image
import numpy as np


def read_seg(input_path):
    """Read inclusive label/row/start/end runs, preserving label zero."""
    width = height = None
    records = []
    reading_data = False
    with open(input_path, 'r', encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if not reading_data:
                key = parts[0].lower()
                if key == 'width':
                    width = int(parts[1])
                elif key == 'height':
                    height = int(parts[1])
                elif key == 'data':
                    reading_data = True
                continue
            if len(parts) != 4:
                raise ValueError(f'Line {line_number}: expected four columns')
            records.append(tuple(map(int, parts)))
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError('Missing or invalid width/height')
    groundtruth = np.full((height, width), -1, dtype=np.int32)
    for label, row, start, end in records:
        if label < 0 or label > np.iinfo(np.int32).max:
            raise ValueError(f'Invalid label: {label}')
        if not (0 <= row < height and 0 <= start <= end < width):
            raise ValueError(f'Invalid run: {label} {row} {start} {end}')
        if np.any(groundtruth[row, start:end + 1] != -1):
            raise ValueError(f'Overlapping runs in row {row}')
        groundtruth[row, start:end + 1] = label
    missing = int(np.sum(groundtruth == -1))
    if missing:
        raise ValueError(f'{missing} pixels have no label')
    return groundtruth


def write_seg(groundtruth, output_path):
    """Write SEG runs with inclusive end columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = groundtruth.shape
    with output_path.open('w', encoding='utf-8', newline='\n') as file:
        file.write(f'width {width}\nheight {height}\nsegments {len(np.unique(groundtruth))}\ndata\n')
        for row_index, row_data in enumerate(groundtruth):
            start = 0
            label = row_data[0]
            for col in range(1, width):
                if row_data[col] != label:
                    file.write(f'{int(label)} {row_index} {start} {col - 1}\n')
                    start, label = col, row_data[col]
            file.write(f'{int(label)} {row_index} {start} {width - 1}\n')


def resize_groundtruth_by_factor(input_path, output_path, factor=10,
                                reference_image_path=None):
    """Match a reference image's size, or divide dimensions by factor."""
    groundtruth = read_seg(input_path)
    old_height, old_width = groundtruth.shape
    if reference_image_path is not None:
        with Image.open(reference_image_path) as image:
            new_width, new_height = image.size
    else:
        if factor <= 0:
            raise ValueError('factor must be greater than zero')
        # Match resize_image_by_factor.py, including its minimum size of one.
        new_width = max(1, round(old_width / factor))
        new_height = max(1, round(old_height / factor))
    resized = np.asarray(
        Image.fromarray(groundtruth).resize(
            (new_width, new_height), resample=Image.Resampling.NEAREST
        ), dtype=np.int32
    )
    write_seg(resized, output_path)
    print(f'Kích thước ground truth gốc: {old_width} × {old_height} px')
    print(f'Kích thước ground truth mới: {new_width} × {new_height} px')
    print(f'Các nhãn trước resize: {np.unique(groundtruth).tolist()}')
    print(f'Các nhãn sau resize: {np.unique(resized).tolist()}')
    print(f'Đã lưu tại: {output_path}')


if __name__ == '__main__':
    resize_groundtruth_by_factor(
        input_path=r'D:\Nhu Y_khong xoa_1\00204_b01_20230106_yinchuan_id000_3024x4032_K6.seg',
        output_path=r'D:\Nhu Y_khong xoa_1\Ncut_Quantum\Thuc nghiem theo yeu cau Review\00204_b01_20230106_yinchuan_id000_3024x4032_K6.seg',
        factor=10,
        # To match the resized image exactly, replace None with its file path.
        reference_image_path=None,
    )

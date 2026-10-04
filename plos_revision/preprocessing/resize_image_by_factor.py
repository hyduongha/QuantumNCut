"""Resize an image by a factor using Pillow's Lanczos resampling."""

from pathlib import Path
from PIL import Image


def resize_by_factor(input_path, output_path, factor=10):
    """Divide both image dimensions by factor and save the resized image."""
    if factor <= 0:
        raise ValueError("factor must be greater than zero")

    output_path = Path(output_path)
    with Image.open(input_path) as image:
        old_width, old_height = image.size
        new_width = max(1, round(old_width / factor))
        new_height = max(1, round(old_height / factor))
        resized_image = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in (".jpg", ".jpeg"):
            if resized_image.mode not in ("RGB", "L"):
                resized_image = resized_image.convert("RGB")
            resized_image.save(output_path, quality=95)
        else:
            resized_image.save(output_path)

    print(f"Kích thước gốc: {old_width} × {old_height} px")
    print(f"Kích thước mới: {new_width} × {new_height} px")
    print(f"Đã lưu tại: {output_path}")


if __name__ == "__main__":
    resize_by_factor(
        input_path=r"D:\Nhu Y_khong xoa_1\36103_b12_20230324_chengdu_id000_6936x9248_K5.jpg",
        output_path=r"D:\Nhu Y_khong xoa_1\Ncut_Quantum\Ncut_36103_b12_20230324_chengdu_id000_6936x9248_K5\Ncut_resize_10\36103_b12_20230324_chengdu_id000_6936x9248_K5.jpg",
        factor=10,
    )

# ============================================================

# RECONSTRUCT A FULL-SIZE .SEG FILE FROM ONE-TO-ONE-RELABELED PATCHES

#

# - Process only the specified datasets.

# - Read only image dimensions and the header from ground truth.

# - Read patches produced by remap_seg_labels_one_to_one.py from seg_remap_one_to_one.

# - Sort patches by index and assemble them into a new label matrix.

# - Require complete patches and divisible dimensions; do not fill labels from GT.

# - Assemble patches directly, without boundary merging or full-image remapping.
# - Export the full-size .seg file and a color segmentation .jpg image.

# ============================================================

from pathlib import Path

import random

import numpy as np

from PIL import Image

# ============================================================

# 1. CONFIGURATION

# ============================================================

# Directory containing full-size ground-truth files.

#

# Supported directory layouts:

#

# GT_ROOT/

# ├── 06935_b06_....seg

# └── 06948_b06_....seg

#

GT_ROOT = Path(

    r"F:\Nhu Y_khong xoa_2\groundtruth_masks_MaSS13K"

)

# Root directory containing prediction datasets

PREDICTION_ROOT = Path(

    r"D:\NhuY_khongxoa_4_tmp\ncut_kq"

)

# Process only datasets listed here

DATASET_FOLDER_NAMES = (

    "06115_b06_20230210_tianjin_id2974_2736x3648_K3",

    #"05913_b06_20230210_tianjin_id2968_3648x2736_K4",

    # Add other datasets here:

    # "06956_b06_20230210_tianjin_id3002_2736x3648_K4",

)

# Methods for patch reconstruction

METHOD_NAMES = (

#    "QL",

    "Ncut",

#    "QPE",

#    "IQPE",

    "Quantum",

)

# Directory containing remapped patches

INPUT_PATCH_ROOT_NAME = "seg_remap_one_to_one"

# Directory containing reconstructed results

OUTPUT_ROOT_NAME = "seg_merge_one_to_one"

# Dimensions of each patch:

# 8 pixels wide and 16 pixels high

PATCH_WIDTH = 8

PATCH_HEIGHT = 16

# Seed used to generate label colors

COLOR_SEED = 12345

# ============================================================

# 2. READ A SEG FILE INTO A LABEL MATRIX

# ============================================================

def read_seg_to_mask(seg_path):

    """

    Read a .seg file and return:

        header_lines

        width

        height

        mask

    Each data line has the following format:

        label row start_col end_col

    end_col is inclusive.

    """

    seg_path = Path(seg_path)

    with open(

        seg_path,

        "r",

        encoding="utf-8",

        errors="replace"

    ) as file:

        lines = file.read().splitlines()

    width = None

    height = None

    data_index = None

    header = []

    for index, line in enumerate(lines):

        stripped = line.strip()

        stripped_lower = stripped.lower()

        header.append(line)

        if stripped_lower == "data":

            data_index = index + 1

            break

        if stripped_lower.startswith("width "):

            width = int(stripped.split()[1])

        elif stripped_lower.startswith("height "):

            height = int(stripped.split()[1])

    if (

        width is None

        or height is None

        or data_index is None

    ):

        raise ValueError(

            "SEG file is missing width, height, or data:\n"

            f"{seg_path}"

        )

    # Initialize the label matrix

    mask = np.full(

        (height, width),

        fill_value=-1,

        dtype=np.int32

    )

    # Read the data following the "data" line

    for line in lines[data_index:]:

        parts = line.strip().split()

        if len(parts) != 4:

            continue

        try:

            label, row, start_col, end_col = map(

                int,

                parts

            )

        except ValueError:

            continue

        if not 0 <= row < height:

            continue

        # Clamp coordinates to the image dimensions

        start_col = max(

            0,

            min(width - 1, start_col)

        )

        end_col = max(

            0,

            min(width - 1, end_col)

        )

        if label == -1:

            raise ValueError(f"Unlabeled pixel marker (-1) in {seg_path}.")

        if start_col <= end_col:

            mask[

                row,

                start_col:end_col + 1

            ] = label

    if np.any(mask == -1):

        raise ValueError(f"Patch SEG does not cover every pixel: {seg_path}")

    # The header already includes the "data" line

    return (

        header[:data_index],

        width,

        height,

        mask

    )

# ============================================================

# 3. ENCODE A MATRIX ROW AS RUNS

# ============================================================

def mask_row_to_runs(

    row_array: np.ndarray,

    row_index: int

):

    """

    Convert a mask row into runs:

        label row start end

    end is written as an inclusive index.

    """

    runs = []

    width = row_array.shape[0]

    column = 0

    while column < width:

        label = int(row_array[column])

        start = column

        while (

            column < width

            and int(row_array[column]) == label

        ):

            column += 1

        end = column - 1

        runs.append(

            (

                label,

                row_index,

                start,

                end

            )

        )

    return runs

# ============================================================

# 4. WRITE A LABEL MATRIX TO A SEG FILE

# ============================================================

def write_seg_from_mask(

    mask: np.ndarray,

    output_path,

    base_header_lines

):

    """

    Write the mask to a full-size .seg file.

    Preserve the ground-truth header and update:

        width

        height

        segments

    Overwrite the file if it already exists.

    """

    output_path = Path(output_path)

    height, width = mask.shape

    unique_labels = np.unique(mask)

    segment_count = len(unique_labels)

    new_header = []

    for line in base_header_lines:

        stripped_lower = line.strip().lower()

        if stripped_lower.startswith("width "):

            new_header.append(

                f"width {width}"

            )

        elif stripped_lower.startswith("height "):

            new_header.append(

                f"height {height}"

            )

        elif stripped_lower.startswith("segments "):

            new_header.append(

                f"segments {segment_count}"

            )

        else:

            new_header.append(line)

    # Ensure that the header contains the data line

    has_data = any(

        line.strip().lower() == "data"

        for line in new_header

    )

    if not has_data:

        new_header.append("data")

    # Create the output directory

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output_path,

        "w",

        encoding="utf-8",

        newline="\n"

    ) as file:

        file.write(

            "\n".join(new_header) + "\n"

        )

        for row_index in range(height):

            runs = mask_row_to_runs(

                mask[row_index],

                row_index

            )

            for label, row, start, end in runs:

                file.write(

                    f"{label} {row} {start} {end}\n"

                )

# ============================================================

# 5. CREATE A COLOR SEGMENTATION IMAGE

# ============================================================

def save_segmentation_jpg(

    mask: np.ndarray,

    output_jpg_path,

    seed: int = 12345

):

    """

    Assign a color to each label and save a JPG image.

    """

    output_jpg_path = Path(

        output_jpg_path

    )

    height, width = mask.shape

    unique_labels = np.unique(mask)

    random_generator = random.Random(seed)

    palette = {}

    for label in unique_labels:

        # Avoid excessively dark colors

        palette[int(label)] = (

            random_generator.randint(40, 255),

            random_generator.randint(40, 255),

            random_generator.randint(40, 255),

        )

    rgb_image = np.zeros(

        (height, width, 3),

        dtype=np.uint8

    )

    for label in unique_labels:

        rgb_image[mask == label] = (

            palette[int(label)]

        )

    output_jpg_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    Image.fromarray(rgb_image).save(

        output_jpg_path,

        quality=95

    )

    print(

        f"✅ JPG saved: {output_jpg_path}\n"

        f"   Number of labels: {len(unique_labels)}"

    )

# ============================================================

# 6. EXTRACT THE INDEX FROM A PATCH FILENAME

# ============================================================

def get_patch_index(

    patch_file_name: str,

    base_name: str

):

    """

    Read the index from the patch filename.

    Expected patch filename format:

        base_name_00005_1.seg

    Where:

        00005 is the patch index

        1 is the segment count or the final filename field

    Return None if the filename is invalid.

    """

    patch_path = Path(patch_file_name)

    if patch_path.suffix.lower() != ".seg":

        return None

    stem = patch_path.stem

    # Split from right to left into:

    # [base_name, idx, segments]

    parts = stem.rsplit("_", 2)

    if len(parts) != 3:

        return None

    patch_base_name = parts[0]

    patch_index_text = parts[1]

    if patch_base_name != base_name:

        return None

    try:

        return int(patch_index_text)

    except ValueError:

        return None

# ============================================================

# 7. RECONSTRUCT PATCHES FOR ONE METHOD

# ============================================================

def merge_subset_patches_into_gt_and_export(

    gt_seg_path,

    patch_folder,

    output_seg_path,

    output_jpg_path,

    patch_width: int = 8,

    patch_height: int = 16,

    seed: int = 12345,

):

    """Assemble all patches into a new label matrix using their creation indices.

    GT supplies only the header and dimensions; GT labels are not used.
    Raise an error for nondivisible dimensions or missing, duplicate, or invalid patches.
    """

    gt_seg_path = Path(gt_seg_path)

    patch_folder = Path(patch_folder)

    output_seg_path = Path(output_seg_path)

    output_jpg_path = Path(output_jpg_path)

    if not gt_seg_path.is_file():

        raise FileNotFoundError(

            f"Ground-truth file not found:\n"

            f"{gt_seg_path}"

        )

    if not patch_folder.is_dir():

        raise FileNotFoundError(

            f"Patch directory not found:\n"

            f"{patch_folder}"

        )

    # Read metadata for the full-size image

    # Read only the header and dimensions; do not read ground-truth labels.
    gt_header = []
    width = height = None
    with gt_seg_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.rstrip("\n")
            gt_header.append(line)
            parts = line.strip().split()
            if parts and parts[0].lower() == "width":
                width = int(parts[1])
            elif parts and parts[0].lower() == "height":
                height = int(parts[1])
            elif parts and parts[0].lower() == "data":
                break
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("The header has missing or invalid width/height values.")
    if width % patch_width or height % patch_height:
        raise ValueError("Image dimensions must be divisible by the patch dimensions.")
    big_mask = np.full((height, width), -1, dtype=np.int32)

    base_name = gt_seg_path.stem

    # Number of complete patches horizontally and vertically

    patches_per_row = (

        width // patch_width

    )

    patches_per_column = (

        height // patch_height

    )

    if (

        patches_per_row == 0

        or patches_per_column == 0

    ):

        raise ValueError(

            "The ground-truth image is smaller than a patch."

        )

    # Valid patch indices:

    # 0 <= idx < max_full_index

    max_full_index = (

        patches_per_row

        * patches_per_column

    )

    used_count = 0

    skipped_name_count = 0

    skipped_index_count = 0

    skipped_size_count = 0

    error_count = 0

    patch_files = sorted(

        path

        for path in patch_folder.iterdir()

        if (

            path.is_file()

            and path.suffix.lower() == ".seg"

        )

    )

    # Sort by numerical index rather than lexicographic filename order.
    indexed_files = []
    for path in patch_files:
        index = get_patch_index(path.name, base_name)
        if index is None:
            raise ValueError(f"Invalid patch filename: {path.name}")
        indexed_files.append((index, path))
    indexed_files.sort(key=lambda item: item[0])
    if len(indexed_files) != max_full_index:
        raise ValueError(
            f"Expected {max_full_index} patches, found {len(indexed_files)} patches."
        )
    for expected_index, (index, path) in enumerate(indexed_files):
        if index != expected_index:
            raise ValueError(f"Missing or duplicate patch index: expected {expected_index}, found {index}.")
    patch_files = [path for index, path in indexed_files]

    for patch_path in patch_files:

        patch_index = get_patch_index(

            patch_file_name=patch_path.name,

            base_name=base_name

        )

        # The filename does not match the expected format

        if patch_index is None:

            skipped_name_count += 1

            continue

        # Use only complete patches;

        # exclude boundary areas outside the divisible region

        if (

            patch_index < 0

            or patch_index >= max_full_index

        ):

            skipped_index_count += 1

            continue

        try:

            (

                _,

                patch_file_width,

                patch_file_height,

                patch_mask

            ) = read_seg_to_mask(

                patch_path

            )

        except Exception as error:

            error_count += 1

            print(

                f"❌ Unable to read patch:\n"

                f"   {patch_path}\n"

                f"   Error: {error}"

            )

            continue

        # Patches must have dimensions of 8x16

        if (

            patch_file_width != patch_width

            or patch_file_height != patch_height

        ):

            skipped_size_count += 1

            print(

                f"⚠️ Incorrect patch dimensions: "

                f"{patch_path.name}\n"

                f"   Actual: "

                f"{patch_file_width}x{patch_file_height}\n"

                f"   Required: "

                f"{patch_width}x{patch_height}"

            )

            continue

        # Determine the patch position in the full-size image

        y_start = (

            patch_index // patches_per_row

        ) * patch_height

        x_start = (

            patch_index % patches_per_row

        ) * patch_width

        # Place the patch at its corresponding position in the new matrix

        big_mask[

            y_start:y_start + patch_height,

            x_start:x_start + patch_width

        ] = patch_mask

        used_count += 1

        if used_count % 5000 == 0:

            print(

                f"Assembled {used_count} patches..."

            )

    if used_count != max_full_index or np.any(big_mask == -1):
        raise ValueError("Not all valid patches were assembled; incomplete results will not be exported.")

    skipped_total = (

        skipped_name_count

        + skipped_index_count

        + skipped_size_count

        + error_count

    )

    print(

        "\n✅ Patch reconstruction completed:"

        f"\n   Ground Truth       : {gt_seg_path}"

        f"\n   Patch directory    : {patch_folder}"

        f"\n   Patches found      : {len(patch_files)}"

        f"\n   Patches used       : {used_count}"

        f"\n   Invalid filenames  : {skipped_name_count}"

        f"\n   Invalid indices    : {skipped_index_count}"

        f"\n   Incorrect sizes    : {skipped_size_count}"

        f"\n   Patch errors       : {error_count}"

        f"\n   Total skipped      : {skipped_total}"

        "\n   The image is reconstructed entirely from patches; no GT labels are used for filling."

    )

    # Export the full-size SEG file

    write_seg_from_mask(

        mask=big_mask,

        output_path=output_seg_path,

        base_header_lines=gt_header

    )

    print(

        f"✅ Full-size SEG saved: {output_seg_path}"

    )

    # Export the color JPG image

    save_segmentation_jpg(

        mask=big_mask,

        output_jpg_path=output_jpg_path,

        seed=seed

    )

    return {

        "patch_files": len(patch_files),

        "used": used_count,

        "skipped": skipped_total,

        "errors": error_count,

    }

# ============================================================

# 8. FIND THE FULL-SIZE GROUND-TRUTH FILE

# ============================================================

def find_ground_truth_seg(

    dataset_name: str

):

    """

    Find the ground-truth file by dataset name.

    Supported locations:

    1. GT_ROOT/dataset_name.seg

    2. GT_ROOT/dataset_name/dataset_name.seg

    If neither location contains the file, search for the exact filename

    dataset_name.seg in subdirectories of GT_ROOT.

    """

    direct_file = (

        GT_ROOT / f"{dataset_name}.seg"

    )

    if direct_file.is_file():

        return direct_file

    nested_file = (

        GT_ROOT

        / dataset_name

        / f"{dataset_name}.seg"

    )

    if nested_file.is_file():

        return nested_file

    # Recursively search for the exact filename

    matches = list(

        GT_ROOT.rglob(

            f"{dataset_name}.seg"

        )

    )

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        print(

            "⚠️ Multiple ground-truth files found "

            "with the same name:"

        )

        for path in matches:

            print(f"   {path}")

        print(

            "This dataset will be skipped to avoid "

            "selecting the wrong ground-truth file."

        )

    return None

# ============================================================

# 9. PROCESS ONE DATASET

# ============================================================

def process_one_dataset(

    dataset_name: str

):

    """

    Process four methods for one dataset:

        QL

        Ncut

        QPE

        IQPE

        Quantum

    """

    dataset_folder = (

        PREDICTION_ROOT / dataset_name

    )

    if not dataset_folder.is_dir():

        print(

            "⚠️ Dataset not found:\n"

            f"   {dataset_folder}"

        )

        return {

            "processed": False,

            "methods": 0,

            "errors": 1,

        }

    input_patch_root = (

        dataset_folder

        / INPUT_PATCH_ROOT_NAME

    )

    if not input_patch_root.is_dir():

        print(

            f"⚠️ Directory not found "

            f"'{INPUT_PATCH_ROOT_NAME}':\n"

            f"   {input_patch_root}"

        )

        return {

            "processed": False,

            "methods": 0,

            "errors": 1,

        }

    ground_truth_seg = find_ground_truth_seg(

        dataset_name

    )

    if ground_truth_seg is None:

        print(

            "⚠️ Full-size ground truth not found "

            "for dataset:\n"

            f"   {dataset_name}"

        )

        return {

            "processed": False,

            "methods": 0,

            "errors": 1,

        }

    output_root = (

        dataset_folder / OUTPUT_ROOT_NAME

    )

    output_root.mkdir(

        parents=True,

        exist_ok=True

    )

    method_count = 0

    error_count = 0

    print("\n" + "=" * 100)

    print(f"RECONSTRUCTING DATASET: {dataset_name}")

    print(f"Ground Truth : {ground_truth_seg}")

    print(f"Patch root   : {input_patch_root}")

    print(f"Output root  : {output_root}")

    print("=" * 100)

    for method_name in METHOD_NAMES:

        patch_folder = (

            input_patch_root / method_name

        )

        if not patch_folder.is_dir():

            print(

                f"⚠️ Missing patch directory for "

                f"{method_name}:\n"

                f"   {patch_folder}"

            )

            continue

        # Save directly to the configured output directory.

        # do not create QL, L, QPE, or IQPE subdirectories

        output_seg_path = (

            output_root

            / f"{dataset_name}_{method_name}.seg"

        )

        output_jpg_path = (

            output_root

            / f"{dataset_name}_{method_name}.jpg"

        )

        print("\n" + "-" * 100)

        print(f"METHOD: {method_name}")

        print(f"Patch : {patch_folder}")

        print(f"SEG   : {output_seg_path}")

        print(f"JPG   : {output_jpg_path}")

        print("-" * 100)

        try:

            merge_subset_patches_into_gt_and_export(

                gt_seg_path=ground_truth_seg,

                patch_folder=patch_folder,

                output_seg_path=output_seg_path,

                output_jpg_path=output_jpg_path,

                patch_width=PATCH_WIDTH,

                patch_height=PATCH_HEIGHT,

                seed=COLOR_SEED,

            )

            method_count += 1

        except Exception as error:

            error_count += 1

            print(

                f"❌ Error reconstructing method "

                f"{method_name}:\n"

                f"   Dataset: {dataset_name}\n"

                f"   Error: {error}"

            )

    return {

        "processed": True,

        "methods": method_count,

        "errors": error_count,

    }

# ============================================================

# 10. MAIN PROGRAM

# ============================================================

def main():

    if not GT_ROOT.is_dir():

        raise FileNotFoundError(

            "Ground-truth directory not found:\n"

            f"{GT_ROOT}"

        )

    if not PREDICTION_ROOT.is_dir():

        raise FileNotFoundError(

            "Prediction directory not found:\n"

            f"{PREDICTION_ROOT}"

        )

    # Remove empty and duplicate names

    selected_dataset_names = list(

        dict.fromkeys(

            dataset_name.strip()

            for dataset_name

            in DATASET_FOLDER_NAMES

            if dataset_name.strip()

        )

    )

    if not selected_dataset_names:

        print(

            "⚠️ No datasets specified in "

            "DATASET_FOLDER_NAMES."

        )

        return

    print("\n" + "=" * 100)

    print("DATASETS TO RECONSTRUCT")

    print("=" * 100)

    for index, dataset_name in enumerate(

        selected_dataset_names,

        start=1

    ):

        print(f"{index}. {dataset_name}")

    processed_dataset_count = 0

    total_method_count = 0

    total_error_count = 0

    for dataset_name in selected_dataset_names:

        result = process_one_dataset(

            dataset_name

        )

        if result["processed"]:

            processed_dataset_count += 1

        total_method_count += result["methods"]

        total_error_count += result["errors"]

    print("\n" + "=" * 100)

    print("ALL PROCESSING COMPLETED!")

    print("=" * 100)

    print(

        f"Number of requested datasets: "

        f"{len(selected_dataset_names)}"

    )

    print(

        f"Number of processed datasets: "

        f"{processed_dataset_count}"

    )

    print(

        f"Number of method results generated: "

        f"{total_method_count}"

    )

    print(

        f"Total number of errors: "

        f"{total_error_count}"

    )

    print(

        f"\nResults are stored in directory "

        f"'{OUTPUT_ROOT_NAME}' for each dataset."

    )

if __name__ == "__main__":

    main()

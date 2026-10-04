# Process all selected images

# ============================================================

# REMAP PREDICTED SEG LABELS TO GROUND-TRUTH LABELS

#

# Ground-truth directory structure:

#

# split_masks/

# ├── 05823_b06_.../

# │   ├── image01.seg

# │   ├── image02.seg

# │   └── ...

# ├── 05913_b06_.../

# └── ...

#

# Prediction directory structure:

#

# KQ_QL_QPE_Test2_1_50/

# ├── 05823_b06_.../

# │   └── seg_format/

# │       ├── QL/

# │       ├── Ncut/

# │       ├── QPE/

# │       └── IQPE/

# │       └── Quantum/

# └── ...

#

# Output directory structure:

#

# KQ_QL_QPE_Test2_1_50/

# ├── 05823_b06_.../

# │   └── seg_remap_one_to_one/

# │       ├── QL/

# │       ├── Ncut/

# │       ├── QPE/

# │       └── IQPE/

# │       └── Quantum/

# └── ...

# ============================================================



from scipy.optimize import linear_sum_assignment

from pathlib import Path



import numpy as np





# ============================================================

# 1. PATH CONFIGURATION

# ============================================================



# Root directory containing ground-truth datasets

GT_ROOT = Path(

    r"F:\Nhu Y_khong xoa_2\8x16_MaSS13K\split_masks"

)



# Root directory containing prediction datasets

PREDICTION_ROOT = Path(

    r"D:\NhuY_khongxoa_4_tmp\ncut_kq"

)



# Process only the datasets listed below

DATASET_FOLDER_NAMES = (

    "06115_b06_20230210_tianjin_id2974_2736x3648_K3",

    #"05913_b06_20230210_tianjin_id2968_3648x2736_K4",





    # Add other dataset directories here:

    # "06956_b06_20230210_tianjin_id3002_2736x3648_K4",

)



# Input subdirectory containing files with corrected end-column indices

INPUT_SEG_FOLDER_NAME = "seg_format"



# Output subdirectory for remapped SEG files

OUTPUT_SEG_FOLDER_NAME = "seg_remap_one_to_one"



# Methods to process

METHOD_NAMES = (

    #"QL",

    "Ncut",

    #"QPE",

    #"IQPE",

    "Quantum",

)



SEG_EXTENSION = ".seg"





# ============================================================

# 2. READ SEG FILES

# ============================================================



def read_seg(path: Path):

    """Read SEG headers, dimensions, and (label, row, start, end) records.
    End columns are inclusive. Malformed data records are skipped."""



    path = Path(path)



    with open(

        path,

        "r",

        encoding="utf-8",

        errors="replace"

    ) as file:

        lines = [

            line.rstrip("\n")

            for line in file

        ]



    width = None

    height = None

    data_index = None



    for index, line in enumerate(lines):



        stripped = line.strip()



        if stripped.startswith("width "):

            width = int(stripped.split()[1])



        elif stripped.startswith("height "):

            height = int(stripped.split()[1])



        elif stripped == "data":

            data_index = index

            break



    if (

        data_index is None

        or width is None

        or height is None

    ):

        raise ValueError(

            "File SEG không hợp lệ, thiếu "

            f"width, height hoặc data:\n{path}"

        )



    # Retain the complete header, including the data marker

    header_lines = lines[:data_index + 1]



    runs = []



    for line in lines[data_index + 1:]:



        stripped = line.strip()



        if not stripped:

            continue



        parts = stripped.split()



        # Read only records containing four fields

        if len(parts) != 4:

            continue



        try:

            label, row, start, end = map(int, parts)



            runs.append(

                (label, row, start, end)

            )



        except ValueError:

            # Skip records that do not contain four integers

            continue



    return (

        header_lines,

        width,

        height,

        runs

    )





# ============================================================

# 3. DECODE RUNS INTO A LABEL ARRAY

# ============================================================



def seg_runs_to_label_image(

    width: int,

    height: int,

    runs

):

    """Decode runs into an H x W integer label array.
    End columns are inclusive; uncovered pixels retain label -1.
    Invalid rows are skipped and column indices are clamped to image bounds."""



    label_image = np.full(

        (height, width),

        fill_value=-1,

        dtype=np.int32

    )



    for label, row, start, end in runs:



        # Skip records with an invalid row index

        if not 0 <= row < height:

            continue



        # Clamp start and end columns to the image bounds

        start_fixed = max(

            0,

            min(width - 1, start)

        )



        end_fixed = max(

            0,

            min(width - 1, end)

        )



        if start_fixed <= end_fixed:



            label_image[

                row,

                start_fixed:end_fixed + 1

            ] = label



    return label_image





# ============================================================

# 4. ENCODE THE LABEL ARRAY AS RUNS

# ============================================================



def label_image_to_runs(label_image):

    """Encode each row as (label, row, start, end) runs with inclusive endpoints."""



    height, width = label_image.shape



    runs = []



    for row_index in range(height):



        row = label_image[row_index]



        start = 0

        current_label = row[0]



        for column_index in range(1, width):



            if row[column_index] != current_label:



                runs.append(

                    (

                        int(current_label),

                        row_index,

                        start,

                        column_index - 1

                    )

                )



                start = column_index

                current_label = row[column_index]



        # Append the final run of the row

        runs.append(

            (

                int(current_label),

                row_index,

                start,

                width - 1

            )

        )



    return runs





# ============================================================

# 5. WRITE SEG FILES

# ============================================================



def write_seg(

    path: Path,

    header_lines,

    width: int,

    height: int,

    label_image

):

    """Write a label array to SEG, updating width, height, and segment count.
    Existing output files are overwritten."""



    path = Path(path)



    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )



    runs = label_image_to_runs(

        label_image

    )



    # Count distinct labels after remapping

    unique_labels = np.unique(label_image)

    segment_count = len(unique_labels)



    new_header = []



    for line in header_lines:



        stripped = line.strip()



        if stripped.startswith("width "):

            new_header.append(

                f"width {width}"

            )



        elif stripped.startswith("height "):

            new_header.append(

                f"height {height}"

            )



        elif stripped.startswith("segments "):

            new_header.append(

                f"segments {segment_count}"

            )



        else:

            new_header.append(line)



    output_lines = list(new_header)



    for label, row, start, end in runs:

        output_lines.append(

            f"{label} {row} {start} {end}"

        )



    # Write mode replaces an existing output file

    with open(

        path,

        "w",

        encoding="utf-8",

        newline="\n"

    ) as file:



        file.write(

            "\n".join(output_lines) + "\n"

        )





# ============================================================

# 6. FIND A LABEL MAPPING

# ============================================================



def best_label_mapping(ground_truth_image, prediction_image):
    """Find the injective label mapping with maximum total pixel overlap.

    All labels, including label 0, participate. Raise an error if an
    injective mapping is impossible or either SEG mask is incomplete.
    """
    if ground_truth_image.shape != prediction_image.shape:
        raise ValueError("Prediction and ground-truth masks have different shapes.")
    if np.any(ground_truth_image == -1) or np.any(prediction_image == -1):
        raise ValueError("An input SEG file leaves pixels without a label (-1).")

    prediction_labels, pred_inverse = np.unique(
        prediction_image, return_inverse=True
    )
    ground_truth_labels, gt_inverse = np.unique(
        ground_truth_image, return_inverse=True
    )
    n_pred = len(prediction_labels)
    n_gt = len(ground_truth_labels)
    if n_pred > n_gt:
        raise ValueError(
            "One-to-one mapping is impossible: "
            f"{n_pred} predicted labels but only {n_gt} ground-truth labels."
        )

    # Count overlapping pixels, including predicted/ground-truth label 0.
    overlap = np.zeros((n_pred, n_gt), dtype=np.int64)
    np.add.at(overlap, (pred_inverse, gt_inverse), 1)

    # A rectangular assignment chooses a distinct GT label for every
    # predicted label and maximizes the sum of matched pixel counts.
    # It replaces factorial enumeration and has no arbitrary label limit.
    row_indices, col_indices = linear_sum_assignment(overlap, maximize=True)
    if len(row_indices) != n_pred:
        raise ValueError("Could not assign every predicted label one-to-one.")
    return {
        int(prediction_labels[row]): int(ground_truth_labels[col])
        for row, col in zip(row_indices, col_indices)
    }


# ============================================================

# 7. APPLY THE LABEL MAPPING

# ============================================================



def remap_prediction(

    prediction_image,

    mapping

):

    """Replace prediction label IDs according to the supplied mapping.
    Masks are computed from the original array to avoid cascading replacements."""



    output_image = prediction_image.copy()



    for prediction_label, ground_truth_label in (

        mapping.items()

    ):



        output_image[

            prediction_image == prediction_label

        ] = ground_truth_label



    return output_image





# ============================================================

# 8. LIST SEG FILES

# ============================================================



def get_seg_files(folder: Path):

    """List SEG files directly inside a directory, ignoring extension case."""



    folder = Path(folder)



    if not folder.is_dir():

        return []



    return sorted(

        file_path

        for file_path in folder.iterdir()

        if (

            file_path.is_file()

            and file_path.suffix.lower()

            == SEG_EXTENSION

        )

    )





# ============================================================

# 9. PROCESS A PREDICTION DIRECTORY

# ============================================================



def remap_folder(

    ground_truth_folder: Path,

    prediction_folder: Path,

    output_folder: Path

):

    """Remap predictions against ground-truth files with matching complete names.
    Iterate over predictions; unused ground-truth files are ignored.
    Skip missing pairs and dimension mismatches, and report processing statistics."""



    ground_truth_folder = Path(

        ground_truth_folder

    )



    prediction_folder = Path(

        prediction_folder

    )



    output_folder = Path(

        output_folder

    )



    # Create the output directory if necessary

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )



    stats = {

        "prediction_files": 0,

        "processed": 0,

        "changed": 0,

        "missing_ground_truth": 0,

        "size_mismatch": 0,

        "errors": 0,

    }



    if not prediction_folder.is_dir():



        print(

            "⚠️ Không tồn tại thư mục dự đoán:\n"

            f"   {prediction_folder}"

        )



        return stats



    if not ground_truth_folder.is_dir():



        print(

            "⚠️ Không tồn tại thư mục Ground Truth:\n"

            f"   {ground_truth_folder}"

        )



        return stats



    # --------------------------------------------------------

    # Iterate over prediction files

    # --------------------------------------------------------



    prediction_files = get_seg_files(

        prediction_folder

    )



    stats["prediction_files"] = len(

        prediction_files

    )



    if not prediction_files:



        print(

            "⚠️ Không có file .seg trong:\n"

            f"   {prediction_folder}"

        )



        return stats



    # Index ground-truth files by their complete filenames.

    # Use lower() for case-insensitive filename matching.

    ground_truth_lookup = {

        file_path.name.lower(): file_path

        for file_path

        in get_seg_files(ground_truth_folder)

    }



    for prediction_path in prediction_files:



        # Find the ground-truth file with the same complete filename

        ground_truth_path = (

            ground_truth_lookup.get(

                prediction_path.name.lower()

            )

        )



        # Skip predictions with no matching ground-truth file

        if ground_truth_path is None:



            stats["missing_ground_truth"] += 1



            print(

                "⚠️ Không tìm thấy Ground Truth "

                "cho file dự đoán:\n"

                f"   {prediction_path.name}"

            )



            continue



        try:

            (

                gt_header,

                gt_width,

                gt_height,

                gt_runs

            ) = read_seg(

                ground_truth_path

            )



            (

                pred_header,

                pred_width,

                pred_height,

                pred_runs

            ) = read_seg(

                prediction_path

            )



            # Check that prediction and ground-truth dimensions match

            if (

                gt_width,

                gt_height

            ) != (

                pred_width,

                pred_height

            ):



                stats["size_mismatch"] += 1



                print(

                    "⚠️ Kích thước không giống nhau, "

                    "bỏ qua:\n"

                    f"   File: {prediction_path.name}\n"

                    f"   GT  : {gt_width}x{gt_height}\n"

                    f"   Pred: {pred_width}x{pred_height}"

                )



                continue



            # Decode the ground-truth label array

            ground_truth_image = (

                seg_runs_to_label_image(

                    gt_width,

                    gt_height,

                    gt_runs

                )

            )



            # Decode the prediction label array

            prediction_image = (

                seg_runs_to_label_image(

                    pred_width,

                    pred_height,

                    pred_runs

                )

            )



            # Determine the label mapping

            mapping = best_label_mapping(

                ground_truth_image,

                prediction_image

            )



            # Apply the label mapping

            remapped_image = remap_prediction(

                prediction_image,

                mapping

            )



            output_path = (

                output_folder

                / prediction_path.name

            )



            # Use the prediction file header

            write_seg(

                path=output_path,

                header_lines=pred_header,

                width=pred_width,

                height=pred_height,

                label_image=remapped_image

            )



            stats["processed"] += 1



            if not np.array_equal(

                remapped_image,

                prediction_image

            ):

                stats["changed"] += 1



            print(

                f"✅ {prediction_path.name}\n"

                f"   Mapping: {mapping}\n"

                f"   Output : {output_path}"

            )



        except Exception as error:



            stats["errors"] += 1



            print(

                "❌ Lỗi khi xử lý file:\n"

                f"   Pred: {prediction_path}\n"

                f"   GT  : {ground_truth_path}\n"

                f"   Lỗi : {error}"

            )



    print(

        "\n"

        f"Hoàn thành thư mục {prediction_folder.name}:\n"

        f"  File dự đoán        : "

        f"{stats['prediction_files']}\n"

        f"  Đã xử lý            : "

        f"{stats['processed']}\n"

        f"  Có thay đổi nhãn    : "

        f"{stats['changed']}\n"

        f"  Thiếu Ground Truth  : "

        f"{stats['missing_ground_truth']}\n"

        f"  Sai kích thước      : "

        f"{stats['size_mismatch']}\n"

        f"  File bị lỗi         : "

        f"{stats['errors']}\n"

        f"  Kết quả             : "

        f"{output_folder}"

    )



    return stats





# ============================================================

# 10. MAIN PROGRAM

# ============================================================



# ============================================================

# 10. MAIN PROGRAM

# ============================================================



def main():



    # --------------------------------------------------------

    # A. CHECK THE GROUND-TRUTH DIRECTORY

    # --------------------------------------------------------



    if not GT_ROOT.exists():

        raise FileNotFoundError(

            "Không tìm thấy thư mục Ground Truth:\n"

            f"{GT_ROOT}"

        )



    if not GT_ROOT.is_dir():

        raise NotADirectoryError(

            "Đường dẫn Ground Truth không phải thư mục:\n"

            f"{GT_ROOT}"

        )



    # --------------------------------------------------------

    # B. CHECK THE PREDICTION DIRECTORY

    # --------------------------------------------------------



    if not PREDICTION_ROOT.exists():

        raise FileNotFoundError(

            "Không tìm thấy thư mục dự đoán:\n"

            f"{PREDICTION_ROOT}"

        )



    if not PREDICTION_ROOT.is_dir():

        raise NotADirectoryError(

            "Đường dẫn dự đoán không phải thư mục:\n"

            f"{PREDICTION_ROOT}"

        )



    # --------------------------------------------------------

    # C. NORMALIZE THE DATASET LIST

    # --------------------------------------------------------



    # Remove:

    # - Empty names

    # - Leading and trailing whitespace

    # - Duplicate names

    #

    # Preserve the original input order.

    selected_dataset_names = list(

        dict.fromkeys(

            dataset_name.strip()

            for dataset_name in DATASET_FOLDER_NAMES

            if dataset_name.strip()

        )

    )



    if not selected_dataset_names:

        print(

            "⚠️ Chưa nhập tên dataset nào trong "

            "DATASET_FOLDER_NAMES."

        )

        return



    print("\n" + "=" * 100)

    print("DANH SÁCH DATASET SẼ ĐƯỢC REMAP")

    print("=" * 100)



    for index, dataset_name in enumerate(

        selected_dataset_names,

        start=1

    ):

        print(f"{index}. {dataset_name}")



    # --------------------------------------------------------

    # D. INITIALIZE STATISTICS

    # --------------------------------------------------------



    total_stats = {

        "requested_datasets": len(selected_dataset_names),



        # Datasets actually processed

        "datasets": 0,



        # Number of method directories checked

        "methods": 0,



        # File statistics

        "prediction_files": 0,

        "processed": 0,

        "changed": 0,

        "missing_ground_truth": 0,

        "size_mismatch": 0,

        "errors": 0,



        # Directory statistics

        "missing_prediction_folders": 0,

        "missing_input_folders": 0,

        "missing_gt_folders": 0,

    }



    # --------------------------------------------------------

    # E. PROCESS ONLY THE SELECTED DATASETS

    # --------------------------------------------------------



    for dataset_name in selected_dataset_names:



        # Example:

        # PREDICTION_ROOT/06935_b06_...

        dataset_folder = (

            PREDICTION_ROOT / dataset_name

        )



        print("\n" + "=" * 100)

        print(f"KIỂM TRA DATASET: {dataset_name}")

        print("=" * 100)



        # ----------------------------------------------------

        # 1. Check the prediction dataset directory

        # ----------------------------------------------------



        if not dataset_folder.is_dir():



            total_stats[

                "missing_prediction_folders"

            ] += 1



            print(

                "⚠️ Không tìm thấy dataset trong "

                "thư mục dự đoán:\n"

                f"   {dataset_folder}\n"

                "   Dataset này sẽ được bỏ qua."

            )



            continue



        # ----------------------------------------------------

        # 2. Check the input SEG subdirectory

        # ----------------------------------------------------



        input_root = (

            dataset_folder

            / INPUT_SEG_FOLDER_NAME

        )



        if not input_root.is_dir():



            total_stats[

                "missing_input_folders"

            ] += 1



            print(

                f"⚠️ Không tìm thấy thư mục "

                f"'{INPUT_SEG_FOLDER_NAME}':\n"

                f"   {input_root}\n"

                "   Dataset này sẽ được bỏ qua."

            )



            continue



        # ----------------------------------------------------

        # 3. Locate the ground-truth dataset with the same name

        # ----------------------------------------------------



        ground_truth_folder = (

            GT_ROOT / dataset_name

        )



        if not ground_truth_folder.is_dir():



            total_stats[

                "missing_gt_folders"

            ] += 1



            print(

                "⚠️ Không tìm thấy thư mục Ground Truth "

                "cùng tên:\n"

                f"   Dataset dự đoán: {dataset_folder}\n"

                f"   GT cần tìm     : {ground_truth_folder}\n"

                "   Dataset này sẽ được bỏ qua."

            )



            continue



        # ----------------------------------------------------

        # 4. Create the remapping output directory

        # ----------------------------------------------------



        output_root = (

            dataset_folder

            / OUTPUT_SEG_FOLDER_NAME

        )



        output_root.mkdir(

            parents=True,

            exist_ok=True

        )



        # Create method subdirectories:

        # seg_remap_one_to_one/QL

        # seg_remap_one_to_one/Ncut

        # seg_remap_one_to_one/QPE

        # seg_remap_one_to_one/IQPE

        # seg_remap_one_to_one/Quantum

        for method_name in METHOD_NAMES:



            (

                output_root / method_name

            ).mkdir(

                parents=True,

                exist_ok=True

            )



        total_stats["datasets"] += 1



        print(f"Ground Truth: {ground_truth_folder}")

        print(f"Prediction  : {input_root}")

        print(f"Output      : {output_root}")



        # ----------------------------------------------------

        # 5. Process the configured methods

        # ----------------------------------------------------



        for method_name in METHOD_NAMES:



            prediction_folder = (

                input_root

                / method_name

            )



            output_folder = (

                output_root

                / method_name

            )



            print("\n" + "-" * 100)

            print(f"PHƯƠNG PHÁP: {method_name}")

            print(f"Pred: {prediction_folder}")

            print(f"Out : {output_folder}")

            print("-" * 100)



            method_stats = remap_folder(

                ground_truth_folder=ground_truth_folder,

                prediction_folder=prediction_folder,

                output_folder=output_folder

            )



            total_stats["methods"] += 1



            # Accumulate statistics

            for key in (

                "prediction_files",

                "processed",

                "changed",

                "missing_ground_truth",

                "size_mismatch",

                "errors",

            ):

                total_stats[key] += method_stats[key]



    # ========================================================

    # F. FINAL STATISTICS

    # ========================================================



    print("\n" + "=" * 100)
    print("PROCESSING COMPLETED!")
    print("=" * 100)

    print(
        f"Number of datasets requested for processing: "
        f"{total_stats['requested_datasets']}"
    )

    print(
        f"Number of datasets processed: "
        f"{total_stats['datasets']}"
    )

    print(
        f"Number of datasets missing from the prediction directory: "
        f"{total_stats['missing_prediction_folders']}"
    )

    print(
        f"Number of datasets without "
        f"'{INPUT_SEG_FOLDER_NAME}': "
        f"{total_stats['missing_input_folders']}"
    )

    print(
        f"Number of datasets without ground truth: "
        f"{total_stats['missing_gt_folders']}"
    )

    print(
        f"Number of methods checked: "
        f"{total_stats['methods']}"
    )

    print(
        f"Total prediction files found: "
        f"{total_stats['prediction_files']}"
    )

    print(
        f"Total files successfully remapped: "
        f"{total_stats['processed']}"
    )

    print(
        f"Number of files with changed labels: "
        f"{total_stats['changed']}"
    )

    print(
        f"Number of prediction files without matching ground truth: "
        f"{total_stats['missing_ground_truth']}"
    )

    print(
        f"Number of files with dimension mismatches: "
        f"{total_stats['size_mismatch']}"
    )

    print(
        f"Number of files with processing errors: "
        f"{total_stats['errors']}"
    )

    print(
        "\nNote: Unmatched ground-truth files are automatically "
        "skipped and are not included in the statistics."
    )





if __name__ == "__main__":

    main()

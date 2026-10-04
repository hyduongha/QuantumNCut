# QuantumNCut

Code for patch-based Normalized Cuts (Ncut) with classical and quantum-assisted matrix–vector multiplication, including the scripts provided for the revised PLOS ONE manuscript **PONE-D-26-15908**.

The quantum-assisted experiments use simulation. The reported simulation results do not demonstrate a practical runtime speedup over classical patch-based Ncut.

## Code locations

The scripts associated with the revision are listed below. The repository also contains earlier experiments and notebooks.

| Location | Purpose |
| --- | --- |
| `segment_patches_quantum_simulator_lanczos.py` | Quantum-assisted patch segmentation using the matrix–vector function in `qmatmul.py` |
| `qmatmul.py` | Quantum matrix–vector routines |
| `requirements.txt` | Dependencies for the segmentation, preprocessing, and evaluation scripts |
| `plos_revision/preprocessing/` | Mask conversion, resizing, and image/ground-truth patch extraction |
| `plos_revision/classical/` | Classical image-level and patch-level Ncut scripts |
| `plos_revision/evaluation/` | Patch label alignment, image reconstruction, and Dice/IoU evaluation |

Keep `segment_patches_quantum_simulator_lanczos.py` and `qmatmul.py` in the same directory to preserve the existing import.

## Installation

Download the repository, or clone it:

```bash
git clone https://github.com/hyduongha/QuantumNCut.git
cd QuantumNCut
```

Create and activate a Python virtual environment. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The main packages are NumPy, SciPy, scikit-learn, scikit-image, Pillow, pandas, openpyxl, Qiskit, and Qiskit Aer. The supplied requirements specify Qiskit 1.4.4 and Qiskit Aer 0.16.0; the other entries are not a complete version record of the original experimental environment.

### Additional dependency in `qmatmul.py`

The current `qmatmul.py` imports `qoop` at module level. If these imports are retained, a compatible `qoop` package must be available in the active Python environment, even when only `qmatmul_qiskit()` is called. `qoop` is not installed by the supplied requirements file.

The `qoop` imports can instead be placed inside the functions that use them: `prepare_state()` requires both `WchainCNOT_xyz` and `QuantumStatePreparation`, and `qmatmul()` requires `WchainCNOT_xyz`. The module-level imports must then be removed or commented out. This makes `qoop` unnecessary for the `qmatmul_qiskit()` path, but it remains required for the other two functions.

Source: https://github.com/vutuanhai237/qoop

## Data and configuration

The study uses a subset of MaSS13K. Obtain the source data following the dataset authors' instructions:

https://github.com/xiechenxi99/MaSS13K

Use the image identifiers and settings reported in the revised manuscript. Do not assume that the repository contains the complete dataset or all experimental outputs.

The scripts contain local paths and configuration values that must be edited before running. Run the commands below from the repository root; relative data paths are interpreted from the current working directory.

Ground-truth and predicted segmentation files use the SEG run-length format. Each data row contains:

```text
label row start_column end_column
```

The start and end columns are inclusive. Label `0` is retained as a regular label.

## Workflow

### 1. Prepare images and ground truth

| Script in `plos_revision/preprocessing/` | Function |
| --- | --- |
| `convert_masks_to_seg.py` | Convert label masks to SEG files |
| `resize_image_by_factor.py` | Resize images for the downsampled image-level baseline |
| `resize_groundtruth.py` | Resize the corresponding ground truth |
| `split_images_and_groundtruth_into_patches.py` | Extract paired image and ground-truth patches |

Edit the input/output paths in the relevant scripts. Image and ground-truth dimensions and filenames must correspond. Preserve discrete ground-truth labels when resizing masks.

For patch extraction, set `patch_h` and `patch_w` to the intended configuration. Dimensions in the manuscript are expressed as height × width: use `(4, 8)` or `(8, 16)` accordingly. Do not assume the checked-in defaults match every reported experiment. Incomplete boundary patches are skipped by the extraction script.

Example commands:

```bash
python plos_revision/preprocessing/convert_masks_to_seg.py
python plos_revision/preprocessing/split_images_and_groundtruth_into_patches.py
```

For the downsampled baseline, configure and run the two resizing scripts separately.

### 2. Segment the images or patches

Edit the following settings in each segmentation script:

- `INPUT_DIR` and `OUTPUT_DIR`;
- `SIGMA_I_VALUES` and `SIGMA_X_VALUES`;
- `K_NEIGHBORS`.

Use the parameter values specified for each experiment. The image-level baseline script includes parameter sweeps; restrict these to the intended settings when reproducing a particular configuration. The neighbor count must not exceed the number of pixels in the input image or patch.

The scripts read the segment count from the final underscore-separated field of the filename, for example `example_00001_3.png` or `example_K3.jpg`. The patch-extraction script obtains this count from the corresponding ground-truth patch.

Run classical patch segmentation:

```bash
python plos_revision/classical/segment_patches_lanczos.py
```

Run quantum-assisted patch segmentation:

```bash
python segment_patches_quantum_simulator_lanczos.py
```

Run the classical image-level baseline on the configured images:

```bash
python plos_revision/classical/segment_images_lanczos.py
```

Use separate output directories for different methods and patch configurations. The segmentation scripts save visualization images and predicted SEG files. They process files directly inside `INPUT_DIR`; select an individual patch folder or adapt the input handling when data are organized into nested folders.

Both patch implementations construct a spatial nearest-neighbor graph. The multiplication step uses a dense-array representation that retains zero entries from graph construction. The quantum-assisted script calls `qmatmul_qiskit()` and computes the remaining vector subtraction classically.

### 3. Align patch labels and reconstruct images

Configure and run:

```bash
python plos_revision/evaluation/remap_seg_labels.py
python plos_revision/evaluation/merge_remap_patches.py
```

Set the ground-truth and prediction roots, dataset folder names, method names, and input/output subfolder names in each script. Set `PATCH_HEIGHT` and `PATCH_WIDTH` in the reconstruction script to match the extraction dimensions.

The workflow aligns predicted patch labels with their corresponding ground-truth labels and then places the aligned patches at their original spatial positions. The reconstruction filename contains the word `merge`, but this step assembles patches; it does not establish a ground-truth-independent region-merging procedure.

### 4. Evaluate Dice and IoU

Configure and run:

```bash
python plos_revision/evaluation/evaluate_patch_segmentation_iou_dice.py
python plos_revision/evaluation/evaluate_image_segmentation_iou_dice.py
```

Update `GT_ROOT`, `PREDICTION_ROOT`, `DATASET_FOLDER_NAMES`, `METHOD_NAMES`, and the prediction subfolder/file settings. Make these names consistent with the outputs of the preceding scripts: the checked-in subfolder names are not automatically synchronized across scripts.

Keep `USE_MAPPING = False` when evaluating predictions whose labels have already been aligned. Full-image evaluation uses the reconstructed aligned prediction. Metrics are averaged across ground-truth labels, including label `0`; patch summaries and reconstructed-image scores are separate measurements.

The evaluation scripts export Excel results. Change the output filenames if earlier results must be preserved.

## Interpretation and reproducibility

The evaluation is ground-truth-assisted: patch segment counts are derived from ground truth, and predicted patch labels are matched to ground truth before reconstruction and evaluation. The scores should be interpreted under this protocol, rather than as an evaluation of fully ground-truth-independent segmentation inference.

Record the exact code revision, Python/package versions, parameters, random seeds, simulator configuration, and data identifiers used for each run. Installing the requirements alone does not reproduce every setting of the reported experiments.

## License

See the repository's `LICENSE` file. Third-party libraries and datasets remain subject to their respective licenses.


# RT-DETR DICOM Preprocessing Experiments

This repository contains the **RT-DETR portion** of a chest X-ray preprocessing and object-detection study. It trains RT-DETR on six preprocessed VinDr-CXR dataset variants and includes a small dataset inspection utility.

> **Scope:** This repository is a partial code release. The YOLOv8, Faster R-CNN, and three-channel combined-preprocessing implementations/results discussed in the associated paper are not included here. The available code is not a complete reproduction of the paper.

## Contents

- `rtdetr_train_full.py` — downloads each prepared dataset with Rclone, prepares the training split/labels, trains RT-DETR, evaluates validation and test splits, and writes metrics and run artifacts.
- `data_info.py` — inspects a prepared dataset, matches CSV labels to image files, and writes a split/class summary.
- `annotations_all_merged_other_1000nf.csv` — annotation CSV used by the CSV-based training path.
- `requirements.txt` — Python dependencies for training.
- `rtdetr_preprocess_runner/` — saved experiment summaries and artifacts. Large model weights may be stored through Git LFS.

## Preprocessing variants

The training script expects six already-preprocessed image datasets, configured in `PREPROCESS_JOBS`:

| Variant | Configured dataset folder |
| --- | --- |
| CLAHE | `vinbigdata-CLAHE_512-png` |
| Histogram equalization | `vinbigdata-Histogram_EQ_512-png` |
| Adaptive percentile | `vinbigdata-adaptive-preprocessed-512-2` |
| Percentile clipping | `vinbigdata-percentile-png-512-2` |
| Raw min-max | `vinbigdata-raw_minmax-png-512` |
| VOI LUT | `vinbigdata-voi_lut-png-512` |

The datasets themselves are not generated from raw DICOM files by this repository. They must already be available in the configured Rclone remote or local workspace.

## Requirements

- Python 3.10 or a compatible environment for the pinned PyTorch packages
- NVIDIA GPU and a compatible CUDA setup for GPU training (CPU execution is supported but impractical for full training)
- [Rclone](https://rclone.org/) installed and available on `PATH`
- Access to the prepared dataset folders, either through Rclone or by staging them locally

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Data access and configuration

The default training configuration reads `rclone:` as its remote prefix and expects an `rclone.conf` file beside `rtdetr_train_full.py`. **Do not commit this file** because it may contain access credentials. Configure the Rclone remote locally, then set the base if its name/path differs:

```bash
export RCLONE_REMOTE_BASE="myremote:"
```

The default annotation CSV is `annotations_all_merged_other_1000nf.csv` next to the training script. You can select another CSV with `RTDETR_LABEL_CSV`.

The training script defaults to 50 epochs, image size 512, batch size 64, and a 70/10/20 train/validation/test split with seed 42. Adjust the constants or environment options in `rtdetr_train_full.py` for your hardware and dataset. A quick-check mode is available:

```bash
RTDETR_QUICK_CHECK=1 RTDETR_QUICK_CHECK_JOB_LIMIT=1 python rtdetr_train_full.py
```

Run all six configured experiments with:

```bash
python rtdetr_train_full.py
```

The script writes downloaded/extracted data, training runs, and summaries under `rtdetr_preprocess_runner/`. Review disk usage before starting a full run.

## Dataset inspection

`data_info.py` uses the first configured dataset (CLAHE) and the label CSV to produce `data_info_report.txt`:

```bash
python data_info.py
```

It uses the same Rclone remote settings and expects the prepared images to be available.

## Reported RT-DETR results

The checked-in experiment summary reports the following **test mAP@0.5** values:

| Preprocessing | Test mAP@0.5 |
| --- | ---: |
| Raw min-max | 34.81% |
| VOI LUT | 34.09% |
| Percentile clipping | 33.88% |
| CLAHE | 33.85% |
| Histogram equalization | 31.75% |
| Adaptive percentile | 20.51% |

These values describe the included RT-DETR runs and their recorded split/configuration. They should not be treated as a complete reproduction of the paper's cross-model or combined-channel results.

## Related paper

The work is related to the Springer chapter [DOI: 10.1007/978-981-92-5699-0_29](https://doi.org/10.1007/978-981-92-5699-0_29). This repository covers only the RT-DETR single-preprocessing experiments available in this code folder; consult the publication for the complete study and its reported methodology.

## Notes

- The repository does not include the full image dataset or raw DICOM collection.
- Rclone configuration, credentials, and any private dataset paths must be set up locally.
- Training and inference artifacts can be large; check Git LFS quotas and repository size before adding model weights.
- Dataset access and any redistribution must follow the dataset's terms and the applicable research/privacy requirements.

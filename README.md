# Chest X-Ray Preprocessing and Detection Experiments

This repository collects the code available in the local `Lung_2026 finals` project for chest X-ray preprocessing and object detection. It includes RT-DETR, YOLO, DINO-related experiments, an RT-DETR three-channel preprocessing experiment, analysis notebooks, and a Dockerized DICOM inference API.

> **Research scope:** The repository does not contain a Faster R-CNN training implementation. It is therefore a partial release of the study associated with [DOI: 10.1007/978-981-92-5699-0_29](https://doi.org/10.1007/978-981-92-5699-0_29), not a complete reproduction of every paper experiment.

## Repository map

| Path | Purpose |
| --- | --- |
| `rtdetr_train_full.py` | RT-DETR training across six single-preprocessing datasets |
| `data_info.py` | Dataset and annotation inspection utility |
| `experiments/yolo/` | YOLO training scripts and an additional YOLO training implementation |
| `experiments/rtdetr_combo/` | RT-DETR training/preprocessing for three-channel combinations |
| `experiments/rtdetr_minmax/` | Separate RT-DETR min-max training script |
| `experiments/dino/` | DINO training scripts; external MMDetection setup/configuration is required |
| `experiments/inference_compare.py` | Comparison/inference script for available model variants |
| `rtdetr_combo_api/` | FastAPI inference service, Dockerfile, Compose file, and browser UI |
| `notebooks/` | Training, preprocessing, visualization, and data inspection notebooks; saved outputs were cleared |
| `rtdetr_preprocess_runner/` | RT-DETR run summaries and selected training artifacts |

The local project also contained datasets, DICOM samples, generated prediction images, logs, checkpoints, and Rclone configuration files. Those are not included in this source upload. No credentials or DICOM files should be committed.

## RT-DETR single-preprocessing pipeline

The main training script expects six preprocessed VinDr-CXR datasets, configured in `PREPROCESS_JOBS`:

| Variant | Default Rclone subfolder |
| --- | --- |
| CLAHE | `vinbigdata-CLAHE_512-png` |
| Histogram equalization | `vinbigdata-Histogram_EQ_512-png` |
| Adaptive percentile | `vinbigdata-adaptive-preprocessed-512-2` |
| Percentile clipping | `vinbigdata-percentile-png-512-2` |
| Raw min-max | `vinbigdata-raw_minmax-png-512` |
| VOI LUT | `vinbigdata-voi_lut-png-512` |

The training script defaults to 50 epochs, image size 512, batch size 64, and a 70/10/20 split with seed 42. It prepares the runtime dataset, trains RT-DETR, evaluates validation/test splits, and writes a CSV summary. Exact settings and available environment overrides are defined in `rtdetr_train_full.py`.

The checked-in summary reports the following test mAP@0.5 results for these RT-DETR runs:

| Preprocessing | Test mAP@0.5 |
| --- | ---: |
| Raw min-max | 34.81% |
| VOI LUT | 34.09% |
| Percentile clipping | 33.88% |
| CLAHE | 33.85% |
| Histogram equalization | 31.75% |
| Adaptive percentile | 20.51% |

These are the recorded results for the included configuration and split; they do not establish equivalence to every result in the publication.

## Install and run the main RT-DETR experiments

Use Python 3.10 or a compatible environment. GPU training requires a compatible NVIDIA/CUDA setup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install [Rclone](https://rclone.org/) separately and configure a remote that can access the prepared image datasets. Keep `rclone.conf` local; it may contain credentials. The script defaults to the `rclone:` prefix and looks for `rclone.conf` next to `rtdetr_train_full.py`. Set a different remote prefix when needed:

```bash
export RCLONE_REMOTE_BASE="myremote:"
```

Run a small quick check:

```bash
RTDETR_QUICK_CHECK=1 RTDETR_QUICK_CHECK_JOB_LIMIT=1 python rtdetr_train_full.py
```

Run all configured variants:

```bash
python rtdetr_train_full.py
```

The datasets themselves are not generated from raw DICOM by the main script; prepared datasets must be accessible through Rclone or staged locally. Review the data paths and training settings in each experiment script before running it. Some scripts retain paths specific to the original research machines.

## Three-channel RT-DETR experiments

The scripts under `experiments/rtdetr_combo/` cover these channel combinations:

- Raw min-max + VOI LUT + percentile clipping
- Raw min-max + VOI LUT + CLAHE
- Raw min-max + percentile clipping + CLAHE
- VOI LUT + percentile clipping + CLAHE

The training code and inference API are separate pieces. The API currently builds its model input from raw min-max, VOI LUT, and CLAHE. Provide the corresponding trained checkpoint when launching the service.

## Docker DICOM inference API

The service under `rtdetr_combo_api/` accepts a DICOM upload, creates preprocessing previews, runs RT-DETR, and returns an annotated PNG. The Docker files and service source are included. The model checkpoint is not included; add the trained compatible model locally at `rtdetr_combo_api/weights/best.pt` before starting the service.

```bash
cd rtdetr_combo_api
mkdir -p weights
# Place the compatible checkpoint at weights/best.pt
docker compose up --build
```

Open `http://127.0.0.1:8000`. The API exposes `/health`, `/predict`, and `/predict-image`; see `rtdetr_combo_api/README.md` for request examples. Uploaded DICOMs and generated outputs are written under `rtdetr_combo_api/runs/`; do not publish that directory.

## Notebooks and remaining experiments

The notebooks are provided for inspection and reuse, with saved cell outputs removed. DINO scripts require the appropriate external DINO/MMDetection packages, configs, and dataset layout. Those dependencies and the Faster R-CNN implementation are not present in this repository.

## Large files and data handling

- Image datasets, raw DICOMs, generated prediction images, run logs, and local Rclone configurations are excluded.
- Training checkpoints may be large. The repository is configured for Git LFS on `.pt` files; install Git LFS before adding or pushing new checkpoints.
- Do not upload identifiable medical information or credentials. Follow VinDr-CXR and institutional data-use terms when accessing or redistributing data.

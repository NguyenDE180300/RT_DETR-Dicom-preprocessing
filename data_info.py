import os
from pathlib import Path
import gc
import json
import shutil as shutil_lib
import shutil
import subprocess
import zipfile
from collections import Counter
import sys

# Try to import required libraries
try:
    import numpy as np
    import pandas as pd
    import torch
    import yaml
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please install required libraries: pip install numpy pandas torch PyYAML")
    sys.exit(1)

# =========================
# STANDALONE CONFIG
# =========================
# This script is designed to be independent.
# It expects 'rclone.conf' and the label CSV to be in the same directory or environment paths.

SCRIPT_DIR = Path(__file__).parent.resolve()
JOB_NAME = "clahe"
REMOTE_SUBDIR = "vinbigdata-CLAHE_512-png"
REMOTE_BASE = os.environ.get("RCLONE_REMOTE_BASE", "rclone:")

# local workspace configuration
WORK_ROOT = SCRIPT_DIR / "rtdetr_analysis_workspace"
DOWNLOAD_ROOT = WORK_ROOT / "downloads"
EXTRACT_ROOT = WORK_ROOT / "extracted"

# Split settings (Mirrored from rtdetr_train_full.py)
SPLIT_SEED = 42
VAL_SIZE = 3000
CSV_SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

# Label and BBox settings
DEFAULT_LABEL_CSV_NAME = "annotations_all_merged_other_1000nf.csv"
LABEL_CSV = os.environ.get(
    "RTDETR_LABEL_CSV",
    str(SCRIPT_DIR / DEFAULT_LABEL_CSV_NAME) if (SCRIPT_DIR / DEFAULT_LABEL_CSV_NAME).exists() else ""
).strip()

CSV_BBOX_FORMAT = os.environ.get("RTDETR_CSV_BBOX_FORMAT", "auto").strip().lower()
CSV_BACKGROUND_CLASS = os.environ.get("RTDETR_CSV_BACKGROUND_CLASS", "No finding").strip()
CSV_IMAGE_ID_COL = os.environ.get("RTDETR_CSV_IMAGE_ID_COL", "image_id").strip()
CSV_CLASS_COL = os.environ.get("RTDETR_CSV_CLASS_COL", "class_name").strip()
CSV_SAMPLE_ID_COL = os.environ.get("RTDETR_CSV_SAMPLE_ID_COL", "sample_id").strip()
CSV_SCORE_COL = os.environ.get("RTDETR_CSV_SCORE_COL", "score").strip()
WBF_IOU_THRESHOLD = float(os.environ.get("RTDETR_WBF_IOU_THRESHOLD", "0.5"))
WBF_SKIP_BOX_THRESHOLD = float(os.environ.get("RTDETR_WBF_SKIP_BOX_THRESHOLD", "0.0"))

# Cleanup settings
KEEP_STAGE = os.environ.get("RTDETR_KEEP_STAGE", "0") == "1"

# =========================
# HELPERS
# =========================

def ensure_dirs():
    for d in [WORK_ROOT, DOWNLOAD_ROOT, EXTRACT_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def ensure_command_exists(command_name: str):
    if shutil_lib.which(command_name) is None:
        raise RuntimeError(f"Missing command '{command_name}'. Please install it and ensure it's in your PATH.")

def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

def rclone_copy(remote_path: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    rclone_conf = SCRIPT_DIR / "rclone.conf"
    if not rclone_conf.exists():
        # Try parent dir just in case
        rclone_conf = SCRIPT_DIR.parent / "rclone.conf"

    cmd = ["rclone", "copy", remote_path, str(local_path)]
    if rclone_conf.exists():
        cmd.extend(["--config", str(rclone_conf)])

    cmd.extend(["--transfers", "4", "--checkers", "4", "--stats", "10s"])
    run_cmd(cmd)

def unpack_archives_in_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    archive_suffixes = {".zip", ".tar", ".gz", ".tgz"}
    archives = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.suffix.lower() in archive_suffixes]
    if not archives:
        return False
    for arc in archives:
        print("Unpacking:", arc)
        if arc.suffix.lower() == ".zip":
            with zipfile.ZipFile(arc, "r") as zf:
                zf.extractall(dst_dir)
        else:
            shutil.unpack_archive(str(arc), str(dst_dir))
    return True

def find_dataset_root(search_root: Path):
    if (search_root / "data.yaml").exists(): return search_root
    if (search_root / "images").exists() and (search_root / "labels").exists(): return search_root
    for p in sorted(search_root.rglob("data.yaml")): return p.parent
    for p in sorted(search_root.rglob("*")):
        if p.is_dir() and (p / "images").exists() and (p / "labels").exists(): return p
    raise FileNotFoundError(f"Dataset root not found in {search_root}")

def ensure_local_job_stage(remote_path: str, local_download_dir: Path, local_extract_dir: Path):
    # Check if already extracted
    try:
        root = find_dataset_root(local_extract_dir)
        print(f"Using existing extracted data at {root}")
        return root
    except FileNotFoundError:
        pass

    # Check if already downloaded
    if not any(local_download_dir.iterdir()) if local_download_dir.exists() else True:
        print(f"Downloading from {remote_path}...")
        ensure_command_exists("rclone")
        rclone_copy(remote_path, local_download_dir)

    # Check if download contains the dataset root directly
    try:
        return find_dataset_root(local_download_dir)
    except FileNotFoundError:
        pass

    # Unpack if needed
    print(f"Unpacking to {local_extract_dir}...")
    unpack_archives_in_dir(local_download_dir, local_extract_dir)
    return find_dataset_root(local_extract_dir)

def list_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]

# =========================
# DATA ANALYSIS LOGIC
# =========================

def analyze_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["image_id"] = df[CSV_IMAGE_ID_COL].astype(str).str.strip()
    df["class_name"] = df[CSV_CLASS_COL].astype(str).str.strip()
    df["sample_id"] = df[CSV_SAMPLE_ID_COL].astype(str).str.strip() if CSV_SAMPLE_ID_COL in df.columns else df["image_id"]

    # Filter background
    bg_names = {CSV_BACKGROUND_CLASS.lower(), "no finding", "background", "negative"}
    df["is_background"] = df["class_name"].str.lower().isin(bg_names)

    ann_df = df[~df["is_background"]].copy()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        if col in ann_df.columns:
            ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce")

    ann_df = ann_df.dropna(subset=["sample_id", "class_name"])
    return df, ann_df

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def perform_wbf(ann_df):
    fused = []
    for _, group in ann_df.groupby(["sample_id", "class_name"]):
        if len(group) <= 1:
            fused.append(group)
            continue
        # Simplified WBF for analysis
        group = group.copy()
        group["score"] = pd.to_numeric(group.get(CSV_SCORE_COL, 1.0), errors="coerce").fillna(1.0)
        group = group.sort_values("score", ascending=False)
        clusters = []
        for _, row in group.iterrows():
            box = row[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
            match = None
            for i, c in enumerate(clusters):
                if compute_iou(box, c["box"]) >= WBF_IOU_THRESHOLD:
                    match = i; break
            if match is None:
                clusters.append({"box": box, "scores": [row["score"]], "boxes": [box]})
            else:
                c = clusters[match]
                c["scores"].append(row["score"])
                c["boxes"].append(box)
                c["box"] = np.average(c["boxes"], axis=0, weights=c["scores"])
        for c in clusters:
            row = group.iloc[0].copy()
            row[["x_min", "y_min", "x_max", "y_max"]] = c["box"]
            fused.append(pd.DataFrame([row]))
    return pd.concat(fused, ignore_index=True) if fused else ann_df.copy()

def split_data(ann_df, sample_ids):
    rng = np.random.default_rng(SPLIT_SEED)
    ratios = CSV_SPLIT_RATIOS

    # Multilabel Stratified Split (Simplified)
    img_classes = ann_df.groupby("sample_id")["class_name"].apply(list).to_dict()
    for sid in sample_ids:
        if sid not in img_classes: img_classes[sid] = [CSV_BACKGROUND_CLASS]

    total = len(sample_ids)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])

    shuffled = sorted(sample_ids)
    rng.shuffle(shuffled)

    splits = {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:])
    }
    return splits, img_classes

def main():
    ensure_dirs()
    print(f"--- Dataset Analysis Tool: {JOB_NAME} ---")

    # 1. Dataset Acquisition
    remote_path = f"{REMOTE_BASE}{REMOTE_SUBDIR}"
    try:
        source_root = ensure_local_job_stage(remote_path, DOWNLOAD_ROOT / JOB_NAME, EXTRACT_ROOT / JOB_NAME)
    except Exception as e:
        print(f"Error during data acquisition: {e}")
        print("Note: This script requires 'rclone' and 'rclone.conf' to download the dataset if not present.")
        return

    # 2. CSV Analysis
    if not LABEL_CSV:
        print("No LABEL_CSV found. Checking for data.yaml...")
        yaml_path = source_root / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
                print(f"Classes: {cfg['names']}")
            return
        else:
            print("No labels found to analyze.")
            return

    print(f"Analyzing labels from: {LABEL_CSV}")
    full_df, ann_df = analyze_csv(Path(LABEL_CSV))

    # Image matching
    image_files = list_image_files(source_root / "images")
    stems = {p.stem for p in image_files}
    matched_ids = [sid for sid in full_df["sample_id"].unique() if sid in stems]

    print(f"Total samples in CSV: {full_df['sample_id'].nunique()}")
    print(f"Samples matched with images: {len(matched_ids)}")

    if not matched_ids:
        print("Warning: No images matched the CSV sample IDs.")
        return

    # WBF and Splitting
    fused_df = perform_wbf(ann_df[ann_df["sample_id"].isin(matched_ids)])
    splits, img_classes = split_data(fused_df, matched_ids)

    # Report Generation
    report = []
    report.append("="*50)
    report.append(f"DATASET REPORT: {JOB_NAME}")
    report.append("="*50)
    report.append(f"Dataset Path: {source_root}")
    report.append(f"Total Matched Images: {len(matched_ids)}")

    report.append("\n[SPLIT DISTRIBUTION]")
    for split, ids in splits.items():
        anns = len(fused_df[fused_df["sample_id"].isin(ids)])
        report.append(f"  {split.capitalize()}: {len(ids)} images, {anns} annotations")

    report.append("\n[CLASS COUNTS (Total)]")
    class_counts = fused_df["class_name"].value_counts()
    for cls, count in class_counts.items():
        report.append(f"  {cls}: {count}")

    report.append("\n" + "="*50)

    output_text = "\n".join(report)
    print(output_text)

    with open(SCRIPT_DIR / "data_info_report.txt", "w") as f:
        f.write(output_text)
    print(f"Report saved to: {SCRIPT_DIR / 'data_info_report.txt'}")

    if not KEEP_STAGE:
        print("\nCleaning up analysis workspace...")
        # Optional: safe_rmtree(WORK_ROOT)
        print("(Cleanup skipped for safety; manually delete 'rtdetr_analysis_workspace' if needed)")

if __name__ == "__main__":
    main()

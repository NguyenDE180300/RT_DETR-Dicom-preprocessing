import os
import shutil
import subprocess
import zipfile
import cv2
from pathlib import Path

# =========================
# CONFIG
# =========================
REMOTE_BASE = "rclone:"  # Tên remote rclone của bạn
RCLONE_CONFIG = "rclone.conf" # File config rclone (phải nằm cùng thư mục)

PREPROCESS_JOBS = [
    {"name": "clahe", "remote_subdir": "vinbigdata-CLAHE_512-png"},
    {"name": "percentile", "remote_subdir": "vinbigdata-percentile-png-512-2"},
    {"name": "raw_minmax", "remote_subdir": "vinbigdata-raw_minmax-png-512"},
    {"name": "voi_lut", "remote_subdir": "vinbigdata-voi_lut-png-512"},
]

PREPROCESS_COMBOS = [
    ("raw_minmax", "voi_lut", "percentile"),
    ("raw_minmax", "voi_lut", "clahe"),
    ("raw_minmax", "percentile", "clahe"),
    ("voi_lut", "percentile", "clahe"),
]

WORK_ROOT = Path("./merge_workspace").resolve()
DOWNLOAD_ROOT = WORK_ROOT / "base_datasets"
MERGED_ROOT = WORK_ROOT / "merged_datasets"

for d in [WORK_ROOT, DOWNLOAD_ROOT, MERGED_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def run_cmd(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def rclone_download(remote_subdir, local_path):
    print(f"Downloading {remote_subdir}...")
    run_cmd([
        "rclone", "copy", f"{REMOTE_BASE}{remote_subdir}", str(local_path),
        "--config", RCLONE_CONFIG, "--transfers", "8"
    ])

def rclone_upload(local_path, remote_subdir):
    remote_path = f"{REMOTE_BASE}{remote_subdir}"
    print(f"Cleaning up old folder on Drive: {remote_subdir}...")
    # Xóa folder cũ trên Drive để tạo lại từ đầu
    subprocess.run(["rclone", "purge", remote_path, "--config", RCLONE_CONFIG], check=False)

    print(f"Uploading {local_path.name} to Drive...")
    run_cmd([
        "rclone", "copy", str(local_path), remote_path,
        "--config", RCLONE_CONFIG, "--transfers", "8"
    ])

def list_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def merge_and_save(p1, p2, p3, out_path):
    img1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(str(p3), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None or img3 is None:
        return False
    merged = cv2.merge([img1, img2, img3])
    cv2.imwrite(str(out_path), merged)
    return True

def unpack_zips(local_dir):
    zips = list(local_dir.glob("*.zip"))
    if not zips:
        return
    print(f"Unpacking {len(zips)} zips in {local_dir.name}...")
    for z in zips:
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        # Xóa file zip sau khi giải nén để tiết kiệm bộ nhớ
        print(f"Removing {z.name}")
        z.unlink()

# =========================
# MAIN
# =========================
def main():
    # 1. Download all base datasets
    jobs_map = {j["name"]: j for j in PREPROCESS_JOBS}
    for name, job in jobs_map.items():
        local_dir = DOWNLOAD_ROOT / name
        if not local_dir.exists() or not any(local_dir.iterdir()):
            rclone_download(job["remote_subdir"], local_dir)

        # Luôn check và unpack nếu có zip
        unpack_zips(local_dir)

    # 2. Merge each combo
    for combo in PREPROCESS_COMBOS:
        combo_name = f"combo_{'_'.join(combo)}"
        out_dir = MERGED_ROOT / combo_name

        # Luôn tạo mới (xóa folder cũ nếu có)
        if out_dir.exists():
            print(f"Removing old local folder: {combo_name}")
            shutil.rmtree(out_dir)

        print(f"\n--- Merging Combo: {combo_name} ---")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Base roots (giả sử cấu trúc là: images/train, labels/train...)
        base_roots = [DOWNLOAD_ROOT / name for name in combo]

        # Copy labels & metadata từ folder đầu tiên
        first_root = base_roots[0]
        for folder in ["labels", "annotations"]:
            if (first_root / folder).exists():
                shutil.copytree(first_root / folder, out_dir / folder, dirs_exist_ok=True)
        if (first_root / "data.yaml").exists():
            shutil.copy2(first_root / "data.yaml", out_dir / "data.yaml")

        # Merge images
        for split in ["train", "val", "test"]:
            img_src_dir = first_root / "images" / split
            if not img_src_dir.exists(): continue

            img_dst_dir = out_dir / "images" / split
            img_dst_dir.mkdir(parents=True, exist_ok=True)

            images = list_image_files(img_src_dir)
            print(f"Processing {split}: {len(images)} images")

            for p1 in images:
                rel = p1.relative_to(img_src_dir)
                p2 = base_roots[1] / "images" / split / rel
                p3 = base_roots[2] / "images" / split / rel
                target = img_dst_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)

                if not p2.exists() or not p3.exists():
                    shutil.copy2(p1, target)
                else:
                    merge_and_save(p1, p2, p3, target)

        # 3. Upload to Drive
        # Tên thư mục trên Drive sẽ là: vinbigdata-merged-raw_minmax-voi_lut-percentile
        drive_folder_name = f"vinbigdata-merged-{'-'.join(combo)}"
        rclone_upload(out_dir, drive_folder_name)

    print("\nHOÀN THÀNH! Tất cả các bộ combo đã được upload lên Drive.")

if __name__ == "__main__":
    main()

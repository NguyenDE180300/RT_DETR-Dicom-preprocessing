import os
import sys
import subprocess
from datetime import datetime

# ------------------------
# PATHS
# ------------------------
project_root = os.getcwd()
mmdet_path = os.path.join(project_root, "mmdetection")
config_file = os.path.join(project_root, "custom_train_dino.py")

# ------------------------
# LOG DIR
# ------------------------
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"dino_train_{time_str}.log")

# ------------------------
# ENV
# ------------------------
env = os.environ.copy()
env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

# ------------------------
# COMMAND
# ------------------------
cmd = [
    sys.executable,
    os.path.join(mmdet_path, "tools/train.py"),
    config_file
]

print("🚀 START TRAIN DINO")
print("LOG:", log_file)

# ------------------------
# RUN + SAVE LOG
# ------------------------
with open(log_file, "w") as f:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    for line in process.stdout:
        print(line, end="")
        f.write(line)

    process.wait()

print("✅ TRAIN DONE")

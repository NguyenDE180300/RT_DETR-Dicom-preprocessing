# RT-DETR DICOM Combo API

FastAPI service that accepts a DICOM chest X-ray, creates the raw min-max, VOI LUT, and CLAHE channels, merges them into a three-channel model input, and returns detections with an annotated PNG.

## Docker

The model checkpoint is not committed. Put a compatible trained checkpoint at `weights/best.pt`, then run:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000`. The `/health` endpoint reports whether the configured checkpoint is present.

## HTTP endpoints

- `GET /health` — service and checkpoint status
- `POST /predict?conf=0.25&iou=0.5` — JSON detections and URLs to intermediate/result images
- `POST /predict-image?conf=0.25&iou=0.5` — annotated PNG response

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict-image?conf=0.25&iou=0.5" \
  -F "file=@/path/to/chest-xray.dcm" \
  --output prediction.png
```

`docker-compose.yml` mounts `weights/` read-only and `runs/` for generated outputs. Uploaded DICOMs are retained in `runs/`; handle and delete them according to your data policies. Do not commit DICOMs or generated patient images.

For local development without Docker:

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

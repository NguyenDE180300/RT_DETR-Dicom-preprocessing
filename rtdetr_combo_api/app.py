from __future__ import annotations

import os
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import RTDETRComboDetector, draw_predictions, predictions_to_dicts
from preprocess import dicom_to_combo, write_preprocessed_pngs


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path(os.environ.get("RTDETR_WEIGHTS", BASE_DIR / "weights" / "best.pt"))
RUNS_DIR = Path(os.environ.get("RTDETR_RUNS_DIR", BASE_DIR / "runs"))
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(title="RT-DETR DICOM Combo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


@lru_cache(maxsize=1)
def get_detector() -> RTDETRComboDetector:
    return RTDETRComboDetector(WEIGHTS_PATH)


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "dicom"
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep)[:80] or "dicom"


def _run_prediction(file: UploadFile, conf: float, iou: float) -> dict:
    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(file.filename or "dicom")
    dicom_path = run_dir / f"{stem}.dcm"
    with dicom_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    preprocessed = dicom_to_combo(dicom_path)
    image_paths = write_preprocessed_pngs(preprocessed, run_dir, stem)
    detector = get_detector()
    predictions = detector.predict(
        image_paths["combo"],
        original_size=preprocessed.original_size,
        conf=conf,
        iou=iou,
    )
    annotated_path = draw_predictions(
        image_paths["combo"],
        predictions,
        run_dir / f"{stem}_prediction.png",
    )

    return {
        "run_id": run_id,
        "stem": stem,
        "preprocessed": preprocessed,
        "image_paths": image_paths,
        "annotated_path": annotated_path,
        "predictions": predictions,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "weights": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.exists(),
        "runs_dir": str(RUNS_DIR),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    try:
        result = _run_prediction(file, conf, iou)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = result["run_id"]
    preprocessed = result["preprocessed"]
    image_paths = result["image_paths"]
    annotated_path = result["annotated_path"]
    predictions = result["predictions"]

    def url_for(path: Path) -> str:
        return f"/runs/{run_id}/{path.name}"

    return {
        "run_id": run_id,
        "filename": file.filename,
        "conf": conf,
        "iou": iou,
        "original_size": {
            "width": preprocessed.original_size[0],
            "height": preprocessed.original_size[1],
        },
        "model_size": {
            "width": preprocessed.model_size[0],
            "height": preprocessed.model_size[1],
        },
        "images": {
            "raw_minmax": url_for(image_paths["raw_minmax"]),
            "voi_lut": url_for(image_paths["voi_lut"]),
            "clahe": url_for(image_paths["clahe"]),
            "combo": url_for(image_paths["combo"]),
            "prediction": url_for(annotated_path),
        },
        "detections": predictions_to_dicts(predictions),
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
) -> FileResponse:
    try:
        result = _run_prediction(file, conf, iou)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        result["annotated_path"],
        media_type="image/png",
        filename=f"{result['stem']}_prediction.png",
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


DEFAULT_IMAGE_SIZE = 512


@dataclass(frozen=True)
class PreprocessResult:
    raw_minmax: np.ndarray
    voi_lut: np.ndarray
    clahe: np.ndarray
    combo_bgr: np.ndarray
    original_size: tuple[int, int]
    model_size: tuple[int, int]


def _first_frame(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim > 3:
        return np.squeeze(arr)[0]
    return arr


def _to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    values = arr[finite]
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _maybe_invert(img: np.ndarray, photometric_interpretation: str) -> np.ndarray:
    if str(photometric_interpretation).upper() == "MONOCHROME1":
        return 255 - img
    return img


def _resize(img: np.ndarray, image_size: int) -> np.ndarray:
    if img.shape[:2] == (image_size, image_size):
        return img
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def dicom_to_combo(dicom_path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> PreprocessResult:
    ds = pydicom.dcmread(str(dicom_path), force=True)
    pixels = _first_frame(ds.pixel_array)
    original_h, original_w = pixels.shape[:2]

    modality = _first_frame(apply_modality_lut(pixels, ds))
    raw = _to_uint8_minmax(modality)

    try:
        voi_source = apply_voi_lut(modality, ds)
    except Exception:
        voi_source = modality
    voi = _to_uint8_minmax(_first_frame(voi_source))

    photometric = getattr(ds, "PhotometricInterpretation", "")
    raw = _maybe_invert(raw, photometric)
    voi = _maybe_invert(voi, photometric)

    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe_filter.apply(raw)

    raw = _resize(raw, image_size)
    voi = _resize(voi, image_size)
    clahe = _resize(clahe, image_size)

    # Match training: cv2.merge([raw_minmax, voi_lut, clahe]) was written as PNG
    # and later read by Ultralytics through OpenCV.
    combo_bgr = cv2.merge([raw, voi, clahe])

    return PreprocessResult(
        raw_minmax=raw,
        voi_lut=voi,
        clahe=clahe,
        combo_bgr=combo_bgr,
        original_size=(int(original_w), int(original_h)),
        model_size=(int(image_size), int(image_size)),
    )


def write_preprocessed_pngs(result: PreprocessResult, output_dir: str | Path, stem: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw_minmax": output_dir / f"{stem}_raw_minmax.png",
        "voi_lut": output_dir / f"{stem}_voi_lut.png",
        "clahe": output_dir / f"{stem}_clahe.png",
        "combo": output_dir / f"{stem}_raw_minmax__voi_lut__clahe.png",
    }

    cv2.imwrite(str(paths["raw_minmax"]), result.raw_minmax)
    cv2.imwrite(str(paths["voi_lut"]), result.voi_lut)
    cv2.imwrite(str(paths["clahe"]), result.clahe)
    cv2.imwrite(str(paths["combo"]), result.combo_bgr)
    return paths

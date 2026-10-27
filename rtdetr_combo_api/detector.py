from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR


CLASS_NAMES = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
}

PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 184, 255),
    (31, 112, 255),
    (127, 22, 255),
    (173, 0, 255),
    (255, 0, 255),
    (255, 0, 143),
]


@dataclass
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_xyxy_original: list[float]


class RTDETRComboDetector:
    def __init__(self, weights_path: str | Path, device: str | None = None) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Cannot find RT-DETR weight: {self.weights_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(str(self.weights_path))

    def predict(
        self,
        image_path: str | Path,
        original_size: tuple[int, int],
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> list[Prediction]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=512,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
        model_h, model_w = result.orig_shape
        original_w, original_h = original_size
        scale_x = original_w / float(model_w)
        scale_y = original_h / float(model_h)

        predictions: list[Prediction] = []
        for box, score, label in zip(xyxy, scores, labels):
            class_id = int(label)
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                Prediction(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, str(class_id)),
                    confidence=float(score),
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xyxy_original=[
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    ],
                )
            )
        return predictions


def draw_predictions(image_path: str | Path, predictions: list[Prediction], output_path: str | Path) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image for drawing: {image_path}")

    for pred in predictions:
        x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox_xyxy]
        color = (128, 128, 128) if pred.class_name == "Other lesion" else PALETTE[pred.class_id % len(PALETTE)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{pred.class_name} {pred.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_y1 = max(0, y1 - th - baseline - 6)
        cv2.rectangle(image, (x1, label_y1), (x1 + tw + 6, label_y1 + th + baseline + 6), color, -1)
        cv2.putText(image, text, (x1 + 3, label_y1 + th + 3), font, font_scale, (255, 255, 255), thickness)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def predictions_to_dicts(predictions: list[Prediction]) -> list[dict]:
    return [
        {
            "class_id": p.class_id,
            "class_name": p.class_name,
            "confidence": round(p.confidence, 6),
            "bbox_xyxy": [round(v, 3) for v in p.bbox_xyxy],
            "bbox_xyxy_original": [round(v, 3) for v in p.bbox_xyxy_original],
        }
        for p in predictions
    ]

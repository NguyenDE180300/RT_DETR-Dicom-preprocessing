import os
from pathlib import Path
import torch
import cv2
from ultralytics import RTDETR
import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path(r"c:\Users\Admin\Documents\Python Project\Lung_2026 finals")
PREPROCESSING_IMAGES_DIR = BASE_DIR / "preprocessing_images"
LUNG_DIR = BASE_DIR / "Lung"
ANNOTATIONS_CSV = LUNG_DIR / "annotations_all_merged_other_1000nf.csv"
LUNG_RESULTS_DIR = LUNG_DIR / "rtdetr_preprocess_runner" / "results"
COMBO_RESULTS_DIR = BASE_DIR / "rtdetr_combo_train_workspace" / "results"
OUTPUT_DIR = BASE_DIR / "inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Methods mapping: suffix -> model path
METHODS = {
    "adaptive": LUNG_RESULTS_DIR / "adaptive_preprocessed" / "best.pt",
    "clahe": LUNG_RESULTS_DIR / "clahe" / "best.pt",
    "hist_eq": LUNG_RESULTS_DIR / "histogram_eq" / "best.pt",
    "percentile": LUNG_RESULTS_DIR / "percentile" / "best.pt",
    "raw_minmax": LUNG_RESULTS_DIR / "raw_minmax" / "best.pt",
    "voi_lut": LUNG_RESULTS_DIR / "LUT" / "best.pt",
    "raw_plus_percentile_plus_clahe": COMBO_RESULTS_DIR / "raw_minmax__percentile__clahe" / "best.pt",
    "raw_plus_voi_plus_clahe": COMBO_RESULTS_DIR / "raw_minmax__voi_lut__clahe" / "best.pt",
    "raw_plus_voi_plus_percentile": COMBO_RESULTS_DIR / "raw_minmax__voi_lut__percentile" / "best.pt",
    "voi_plus_percentile_plus_clahe": COMBO_RESULTS_DIR / "voi_lut__percentile__clahe" / "best.pt",
}

# WBF Configuration
WBF_IOU_THRESHOLD = 0.5

def compute_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union

def weighted_fuse(boxes, scores, labels, iou_thr=0.5):
    if len(boxes) == 0:
        return [], [], []

    # Sort by score
    indices = np.argsort(scores)[::-1]
    boxes = np.array(boxes)[indices]
    scores = np.array(scores)[indices]
    labels = np.array(labels)[indices]

    clusters = []
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        label = labels[i]

        best_cluster = None
        best_iou = 0.0

        for cluster_idx, cluster in enumerate(clusters):
            if cluster['label'] != label:
                continue
            iou = compute_iou_xyxy(box, cluster['box'])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_cluster = cluster_idx

        if best_cluster is None:
            clusters.append({
                'boxes': [box],
                'scores': [score],
                'label': label,
                'box': box.copy()
            })
        else:
            cluster = clusters[best_cluster]
            cluster['boxes'].append(box)
            cluster['scores'].append(score)
            weights = np.array(cluster['scores'])
            weights = np.maximum(weights, 1e-6)
            cluster['box'] = np.average(np.array(cluster['boxes']), axis=0, weights=weights)

    fused_boxes = []
    fused_scores = []
    fused_labels = []
    for cluster in clusters:
        fused_boxes.append(cluster['box'])
        fused_scores.append(np.mean(cluster['scores']))
        fused_labels.append(cluster['label'])

    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)

def get_ground_truth(csv_path, image_id):
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    gt_boxes = df[df['image_id'] == image_id]
    results = []
    for _, row in gt_boxes.iterrows():
        if pd.notna(row['x_min']):
            results.append({
                'label': row['class_name'],
                'bbox': [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
            })
    return results

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    if len(boxes) == 0:
        return boxes, scores, labels

    keep = []
    # Sort by score descending
    idxs = np.argsort(scores)[::-1]

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        remaining_idxs = idxs[1:]
        # Calculate IoU of current box with all remaining boxes
        current_box = boxes[i]
        ious = np.array([compute_iou(current_box, boxes[j]) for j in remaining_idxs])

        # Identify boxes to suppress: Same label AND IoU > threshold
        same_label = (labels[remaining_idxs] == labels[i])
        suppress = (ious > iou_threshold) & same_label

        # Keep indices that are NOT suppressed
        idxs = remaining_idxs[~suppress]

    return boxes[keep], scores[keep], labels[keep]

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Discover all unique image IDs
    all_files = list(PREPROCESSING_IMAGES_DIR.glob("*.png"))
    image_ids = sorted(list(set(f.name.split('_')[0] for f in all_files)))
    print(f"Found {len(image_ids)} unique image IDs.")

    # Define a simple color palette for classes
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 184, 255), (31, 112, 255), (127, 22, 255), (173, 0, 255),
        (255, 0, 255), (255, 0, 143)
    ]

    for image_id in image_ids:
        print(f"\n========== Processing Image ID: {image_id} ==========")

        img_output_dir = OUTPUT_DIR / image_id
        img_output_dir.mkdir(parents=True, exist_ok=True)

        for method_name, model_path in METHODS.items():
            img_name = f"{image_id}_{method_name}.png"
            img_path = PREPROCESSING_IMAGES_DIR / img_name

            if not img_path.exists() or not model_path.exists():
                continue

            print(f"--- Inferencing {method_name} ---")
            model = RTDETR(model_path)
            results = model.predict(source=str(img_path), device=device, conf=0.3, verbose=False)
            res = results[0]

            # Load background image for drawing
            draw_img = cv2.imread(str(img_path))
            names = res.names

            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                labels = res.boxes.cls.cpu().numpy().astype(int)

                # NEW: Apply NMS to clean up overlapping boxes within this model's output
                boxes, scores, labels = apply_nms(boxes, scores, labels, iou_threshold=0.5)

                # Identify 'Other lesion' index for layering
                other_lesion_idx = -1
                for idx, name in names.items():
                    if name == "Other lesion":
                        other_lesion_idx = idx
                        break

                # Sort to put 'Other lesion' first (drawn behind)
                if other_lesion_idx != -1:
                    sort_idx = np.argsort([1 if l == other_lesion_idx else 0 for l in labels])
                    boxes, scores, labels = boxes[sort_idx], scores[sort_idx], labels[sort_idx]

                # Draw boxes with large font
                for box, score, label in zip(boxes, scores, labels):
                    name = names[int(label)]
                    color = (128, 128, 128) if name == "Other lesion" else palette[int(label) % len(palette)]

                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(draw_img, p1, p2, color, 3)

                    label_txt = f"{name} {score:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale, font_thickness = 2.5, 4
                    (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, font_thickness)
                    cv2.rectangle(draw_img, (p1[0], p1[1] - th - 15), (p1[0] + tw, p1[1]), color, -1)
                    cv2.putText(draw_img, label_txt, (p1[0], p1[1] - 10), font, font_scale, (255, 255, 255), font_thickness)

            # Summary legend
            counts = {}
            if len(res.boxes) > 0:
                for l in res.boxes.cls:
                    name = names[int(l)]
                    counts[name] = counts.get(name, 0) + 1

            sorted_items = sorted(counts.items(), key=lambda x: x[0] == "Other lesion")
            summary_text = f"Method: {method_name} | Found: " + (", ".join([f"{k}({v})" for k, v in sorted_items]) if sorted_items else "Nothing")

            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
            (text_w, text_h), _ = cv2.getTextSize(summary_text, font, font_scale, thickness)
            cv2.rectangle(draw_img, (10, 10), (30 + text_w, 30 + text_h + 20), (0, 0, 0), -1)
            cv2.putText(draw_img, summary_text, (25, 25 + text_h + 5), font, font_scale, (255, 255, 255), thickness)

            out_path = img_output_dir / f"pred_{image_id}_{method_name}.png"
            cv2.imwrite(str(out_path), draw_img)
            print(f"Saved prediction to {out_path}")

if __name__ == "__main__":
    run_inference()

#!/usr/bin/env python3
"""
Instance segmentation: 7 encoding visualizations + pairwise prompts for VLLM ranking.

Run:  conda activate cv
      python instseg_visualize_pairwise.py
(Requires opencv-python in the active environment.)

Step 1) Visualize each annotation (including GT) in 7 ways:
  1. Each instance different 50% color, white stroke, label in middle
  2. Same as 1 but 100% opacity
  3. Same color per class, white stroke, label
  4. Same as 3 but no label (prompt text describes color->class)
  5. Only stroke, same color per class, with label
  6. Same as 5 but no label (prompt text describes color->class)
  7. No image; prediction as JSON-like text in prompt

Step 2) Pairwise: exhaust all pairs with same (image_id, coi, encoding).
"""
from __future__ import annotations

import json
import itertools
import sys
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    import numpy as np
except ImportError as e:
    print("Need opencv and numpy. Run: conda activate cv", file=sys.stderr)
    raise SystemExit(1) from e
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
IMAGES_JSON = ROOT / "images.json"
ANNOTATIONS_JSON = ROOT / "annotations.json"
FINAL_IMAGES_DIR = ROOT / "final_images"
MEDIA_ROOT = ROOT / "media" / "instance_seg"
OUTPUT_JSON = ROOT / "instance_seg_pairwise_encoded.json"

# Encoding names and keys
ENCODINGS = [
    ("enc1_50pct_per_instance", "1"),   # 50% per-instance color, white stroke, label
    ("enc2_100pct_per_instance", "2"),  # 100% per-instance, white stroke, label
    ("enc3_same_color_per_class", "3"), # same color per class, white stroke, label
    ("enc4_same_color_per_class_no_label", "4"), # same as 3, no label (describe in prompt)
    ("enc5_stroke_only_per_class", "5"),        # stroke only, same color per class, label
    ("enc6_stroke_only_per_class_no_label", "6"), # stroke only, no label (describe in prompt)
    ("enc7_json_text", "7"),            # no image, JSON text in prompt
]

# BGR colors for instances (per-instance distinct)
INSTANCE_COLORS_BGR = [
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 128, 0),   # orange
    (128, 0, 255),   # violet
    (0, 165, 255),   # orange (cv2)
    (203, 192, 255), # pink
    (0, 128, 255),   # dark orange
]
# Per-class colors (for encodings 3–6): stable order by coi
CLASS_COLORS_BGR = [
    (0, 0, 255),     # red - 1st class
    (0, 255, 0),     # green - 2nd
    (255, 0, 0),     # blue - 3rd
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 128, 0),   # orange
    (128, 0, 255),   # violet
]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def put_text_white_black_stroke(img, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=1):
    """Draw text: white fill with a thin black stroke (outline)."""
    x, y = org
    # Thin black outline: 1-pixel offsets in 8 directions
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        cv2.putText(img, text, (x + dx, y + dy), font_face, font_scale, BLACK, thickness, cv2.LINE_AA)
    # White fill on top
    cv2.putText(img, text, (x, y), font_face, font_scale, WHITE, thickness, cv2.LINE_AA)


def polygon_to_pts(polygon):
    """Convert [[x,y],...] to np array (N,1,2) for cv2."""
    return np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))


def polygon_centroid(polygon):
    """Return (x, y) centroid for label placement."""
    pts = np.array(polygon, dtype=np.float32)
    m = cv2.moments(pts)
    if m["m00"] == 0:
        return (int(pts[:, 0].mean()), int(pts[:, 1].mean()))
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (int(cx), int(cy))


def get_class_to_color(coi: list) -> dict:
    """Stable mapping: class name -> BGR (by order in coi)."""
    return {c: CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)] for i, c in enumerate(coi)}


def prediction_labels_numbered(predictions: list) -> list:
    """Return list of labels as 'class 1', 'class 2', ... per class (same order as predictions)."""
    class_count = defaultdict(int)
    out = []
    for pred in predictions:
        class_count[pred["label"]] += 1
        out.append(f"{pred['label']} {class_count[pred['label']]}")
    return out


def draw_predictions_enc1(img_bgr, predictions, alpha=0.5):
    """Each instance different 50% color, white stroke, label in middle ('class 1', 'class 2', ...). Labels always 100% opacity."""
    labels = prediction_labels_numbered(predictions)
    overlay = img_bgr.copy()
    for i, pred in enumerate(predictions):
        pts = polygon_to_pts(pred["polygon"])
        color = INSTANCE_COLORS_BGR[i % len(INSTANCE_COLORS_BGR)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, WHITE, 2)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    for pred, label_text in zip(predictions, labels):
        cx, cy = polygon_centroid(pred["polygon"])
        put_text_white_black_stroke(out, label_text, (cx - 20, cy))
    return out


def draw_predictions_enc2(img_bgr, predictions):
    """Same as enc1 but 100% opacity. Labels: 'class 1', 'class 2', ..."""
    labels = prediction_labels_numbered(predictions)
    out = img_bgr.copy()
    for i, pred in enumerate(predictions):
        pts = polygon_to_pts(pred["polygon"])
        color = INSTANCE_COLORS_BGR[i % len(INSTANCE_COLORS_BGR)]
        cv2.fillPoly(out, [pts], color)
        cv2.polylines(out, [pts], True, WHITE, 2)
        cx, cy = polygon_centroid(pred["polygon"])
        put_text_white_black_stroke(out, labels[i], (cx - 20, cy))
    return out


def draw_predictions_enc3(img_bgr, predictions, coi: list):
    """Same color per class, white stroke, label as 'class 1', 'class 2', ... per class."""
    labels = prediction_labels_numbered(predictions)
    class_color = get_class_to_color(coi)
    out = img_bgr.copy()
    for pred, label_text in zip(predictions, labels):
        pts = polygon_to_pts(pred["polygon"])
        color = class_color.get(pred["label"], (128, 128, 128))
        cv2.fillPoly(out, [pts], color)
        cv2.polylines(out, [pts], True, WHITE, 2)
        cx, cy = polygon_centroid(pred["polygon"])
        put_text_white_black_stroke(out, label_text, (cx - 20, cy))
    return out


def draw_predictions_enc4(img_bgr, predictions, coi: list):
    """Same color per class, white stroke, no label (caller adds color->class in prompt)."""
    class_color = get_class_to_color(coi)
    out = img_bgr.copy()
    for pred in predictions:
        pts = polygon_to_pts(pred["polygon"])
        color = class_color.get(pred["label"], (128, 128, 128))
        cv2.fillPoly(out, [pts], color)
        cv2.polylines(out, [pts], True, WHITE, 2)
    return out


def draw_predictions_enc5(img_bgr, predictions, coi: list):
    """Only stroke, same color per class, with label ('class 1', 'class 2', ...)."""
    labels = prediction_labels_numbered(predictions)
    class_color = get_class_to_color(coi)
    out = img_bgr.copy()
    for pred, label_text in zip(predictions, labels):
        pts = polygon_to_pts(pred["polygon"])
        color = class_color.get(pred["label"], (128, 128, 128))
        cv2.polylines(out, [pts], True, color, 2)
        cx, cy = polygon_centroid(pred["polygon"])
        put_text_white_black_stroke(out, label_text, (cx - 20, cy))
    return out


def draw_predictions_enc6(img_bgr, predictions, coi: list):
    """Only stroke, same color per class, no label."""
    class_color = get_class_to_color(coi)
    out = img_bgr.copy()
    for pred in predictions:
        pts = polygon_to_pts(pred["polygon"])
        color = class_color.get(pred["label"], (128, 128, 128))
        cv2.polylines(out, [pts], True, color, 2)
    return out


def _round_polygon(polygon: list, ndigits: int = 2) -> list:
    """Round polygon coordinates to ndigits decimal places."""
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in polygon]


def predictions_to_json_text(predictions: list) -> str:
    """Format predictions as compact JSON-like text for encoding 7. Coordinates rounded to 2 decimal places."""
    parts = []
    for i, p in enumerate(predictions):
        label = p["label"]
        poly = _round_polygon(p["polygon"], 2)
        parts.append(json.dumps({"instance_id": i, "label": label, "polygon": poly}, separators=(",", ":")))
    return "\n".join(parts)[:4000]


def color_class_prompt_text(coi: list) -> str:
    """Text describing which color corresponds to which class (for enc 4 and 6)."""
    class_color = get_class_to_color(coi)
    color_names = ["red", "green", "blue", "magenta", "yellow", "orange", "violet"]
    lines = []
    for i, c in enumerate(coi):
        name = color_names[i % len(color_names)]
        lines.append(f"'{c}' is {name}")
    return "Legend: " + "; ".join(lines) + "."


def render_encoding(enc_key: str, img_bgr, ann: dict, out_path: Path) -> bool:
    """Render one annotation with given encoding; save to out_path. Returns True on success."""
    preds = ann.get("predictions", [])
    coi = ann.get("coi", []) or list({p["label"] for p in preds})
    if enc_key == "1":
        out = draw_predictions_enc1(img_bgr, preds, alpha=0.5)
    elif enc_key == "2":
        out = draw_predictions_enc2(img_bgr, preds)
    elif enc_key == "3":
        out = draw_predictions_enc3(img_bgr, preds, coi)
    elif enc_key == "4":
        out = draw_predictions_enc4(img_bgr, preds, coi)
    elif enc_key == "5":
        out = draw_predictions_enc5(img_bgr, preds, coi)
    elif enc_key == "6":
        out = draw_predictions_enc6(img_bgr, preds, coi)
    else:
        return False  # enc 7: no image
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
    return True


def main():
    with open(IMAGES_JSON) as f:
        images = json.load(f)
    with open(ANNOTATIONS_JSON) as f:
        annotations = json.load(f)

    image_by_id = {img["id"]: img for img in images}
    # Resolve base image path: file_path is e.g. "instance_seg/xxx.png"
    def get_image_path(image_id: str) -> Path | None:
        img = image_by_id.get(image_id)
        if not img:
            return None
        return FINAL_IMAGES_DIR / img["file_path"]

    # Create encoding output dirs (1-6 only; 7 has no image)
    for enc_dir, _ in ENCODINGS[:-1]:
        (MEDIA_ROOT / enc_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: For each annotation, render 7 encodings (1-6 to disk; 7 is text only)
    # We'll store: (ann_id, encoding_key) -> rel_path for enc 1-6, and json_text for 7
    ann_encoding_assets = {}  # (ann["id"], enc_key) -> {"path": rel_path or None, "json_text": str or None}

    for ann in tqdm(annotations, desc="Rendering encodings"):
        image_id = ann["image_id"]
        img_path = get_image_path(image_id)
        if not img_path or not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        preds = ann.get("predictions", [])
        if not preds:
            continue
        coi = ann.get("coi", []) or list({p["label"] for p in preds})
        coi_key = tuple(sorted(coi))
        ann_id = ann["id"]

        for enc_dir, enc_key in ENCODINGS:
            if enc_key == "7":
                ann_encoding_assets[(ann_id, enc_key)] = {"path": None, "json_text": predictions_to_json_text(preds)}
                continue
            # Filename: unique per annotation (one vis per ann per encoding)
            safe_name = f"{image_id}_{ann_id}_{enc_key}.jpg"
            out_path = MEDIA_ROOT / enc_dir / safe_name
            ok = render_encoding(enc_key, img_bgr, ann, out_path)
            if ok:
                rel = f"media/instance_seg/{enc_dir}/{safe_name}"
                ann_encoding_assets[(ann_id, enc_key)] = {"path": rel, "json_text": None}
            else:
                ann_encoding_assets[(ann_id, enc_key)] = {"path": None, "json_text": None}

    # Step 2: Group by (image_id, coi, encoding)
    def coi_key(ann):
        c = ann.get("coi") or []
        return tuple(sorted(c)) if c else ()

    groups = defaultdict(list)  # (image_id, coi_key, enc_key) -> [ann, ...]
    for ann in annotations:
        if not ann.get("predictions"):
            continue
        image_id = ann["image_id"]
        ck = coi_key(ann)
        for _, enc_key in ENCODINGS:
            key = (image_id, ck, enc_key)
            groups[key].append(ann)

    # Build pairwise entries: same (image_id, coi, encoding) -> all pairs (A, B)
    PROMPT_PREFIX = "This is an instance segmentation task. Given the original image <image>, which prediction result is better?\n"
    PROMPT_SUFFIX = "\nPlease answer with A or B directly."
    prompt_image_part = "A. <image>\nB. <image>"

    entries = []
    global_id = 0
    for (image_id, ck, enc_key), group_anns in tqdm(groups.items(), desc="Pairwise", total=len(groups)):
        if len(group_anns) < 2:
            continue
        img = image_by_id.get(image_id)
        orig_media = img["file_path"] if img else None  # e.g. "instance_seg/xxx.png"

        coi_list = list(ck)
        legend_text = color_class_prompt_text(coi_list) if enc_key in ("4", "6") else ""

        for ann_a, ann_b in itertools.combinations(group_anns, 2):
            if ann_a["id"] == ann_b["id"]:
                continue
            asset_a = ann_encoding_assets.get((ann_a["id"], enc_key), {})
            asset_b = ann_encoding_assets.get((ann_b["id"], enc_key), {})

            if enc_key == "7":
                json_a = asset_a.get("json_text") or "[]"
                json_b = asset_b.get("json_text") or "[]"
                prompt_body = f"A.\n{json_a}\n\nB.\n{json_b}"
                media_list = [orig_media] if orig_media else []
            else:
                path_a = asset_a.get("path")
                path_b = asset_b.get("path")
                if not path_a or not path_b:
                    continue
                prompt_body = prompt_image_part
                if legend_text:
                    prompt_body = legend_text + "\n\n" + prompt_body
                media_list = [orig_media, path_a, path_b] if orig_media else [path_a, path_b]

            entry = {
                "id": str(global_id),
                "media": media_list,
                "prompt": PROMPT_PREFIX + prompt_body + PROMPT_SUFFIX,
                "choices": ["A", "B"],
                "metadata": {
                    "task": "instance_segmentation",
                    "image_id": image_id,
                    "coi": coi_list,
                    "encoding": enc_key,
                    "ann_id_a": ann_a["id"],
                    "ann_id_b": ann_b["id"],
                    "model_name_a": ann_a.get("model_name", ""),
                    "model_name_b": ann_b.get("model_name", ""),
                    "error_type_a": ann_a.get("error_type"),
                    "error_type_b": ann_b.get("error_type"),
                    "final_score_a": ann_a.get("final_score"),
                    "final_score_b": ann_b.get("final_score"),
                },
            }
            # Answer: A is better if score_a > score_b
            score_a = ann_a.get("final_score")
            score_b = ann_b.get("final_score")
            if score_a is not None and score_b is not None:
                entry["answer"] = "A" if score_a >= score_b else "B"
            else:
                entry["answer"] = None
            entries.append(entry)
            global_id += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} pairwise entries to {OUTPUT_JSON}")
    print(f"Media under {MEDIA_ROOT}")


if __name__ == "__main__":
    main()

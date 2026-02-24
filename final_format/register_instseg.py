#!/usr/bin/env python3
"""
Register instance_segmentation task from error_annotations into the format of
sample_json_edited.json: one images list (base images from BASE_IMAGES_50) and
one annotations list (one annotation per error_annotation JSON).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
ROOT = Path(__file__).resolve().parent
INSTSEG = ROOT / "202509-vllm-tool-judge5-inst-seg-labelme" / "instseg_data"
ERROR_ANNOTATIONS = INSTSEG / "error_annotations"
IMAGES_DIR = INSTSEG / "images"
BASE_IMAGES_LIST = INSTSEG / "BASE_IMAGES_50.txt"
FINAL_IMAGES_DIR = ROOT / "final_images" / "instance_seg"
OUTPUT_JSON = ROOT / "instance_seg_registered.json"


def load_base_images() -> list[str]:
    """Load base image IDs (no extension) from BASE_IMAGES_50.txt."""
    lines = BASE_IMAGES_LIST.read_text().strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def data_source_from_base_id(base_id: str) -> str:
    if base_id.startswith("cityscapes_"):
        return "Cityscapes"
    if base_id.startswith("coco_"):
        return "COCO"
    if base_id.startswith("lvis_"):
        return "LVIS"
    return ""


def image_extension(base_id: str) -> str:
    if base_id.startswith("cityscapes_"):
        return ".png"
    return ".jpg"


def get_groundtruth_class_and_dims(base_id: str) -> tuple[list[str], int | None, int | None]:
    """Get distinct labels and image size from neighbor _gt.json in instseg_data/images."""
    gt_path = IMAGES_DIR / f"{base_id}_gt.json"
    if not gt_path.exists():
        # fallback: any json with same base (e.g. base_id.json)
        gt_path = IMAGES_DIR / f"{base_id}.json"
    if not gt_path.exists():
        return [], None, None
    data = json.loads(gt_path.read_text())
    labels = list({s["label"] for s in data.get("shapes", [])})
    h = data.get("imageHeight")
    w = data.get("imageWidth")
    return labels, h, w


def build_images(base_ids: list[str]) -> list[dict]:
    """Build images list: copy base images to final_images/instance_seg and build image records."""
    FINAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    for i, base_id in enumerate(base_ids):
        ext = image_extension(base_id)
        src_name = base_id + ext
        src = IMAGES_DIR / src_name
        if not src.exists():
            # try alternate
            alt = IMAGES_DIR / (base_id + (".jpg" if ext == ".png" else ".png"))
            if alt.exists():
                src, src_name = alt, alt.name
            else:
                images.append({
                    "id": base_id,
                    "file_path": f"instance_seg/{base_id}{ext}",
                    "data_source": data_source_from_base_id(base_id),
                    "height": None,
                    "width": None,
                    "scene": "",
                    "is_crowd": False,
                    "is_longtail": False,
                    "groundtruth_class": [],
                    "groundtruth_class_id": [],
                })
                continue
        dst = FINAL_IMAGES_DIR / src_name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
        gt_classes, height, width = get_groundtruth_class_and_dims(base_id)
        images.append({
            "id": base_id,
            "file_path": f"instance_seg/{src_name}",
            "data_source": data_source_from_base_id(base_id),
            "height": height,
            "width": width,
            "scene": "",
            "is_crowd": False,
            "is_longtail": False,
            "groundtruth_class": gt_classes,
            "groundtruth_class_id": [],  # leave blank
        })
    return images


def parse_error_json_filename(name: str) -> tuple[str | None, list[str], float | None, bool]:
    """
    Parse filename to (base_id, coi, final_score, is_gt).
    Examples:
      cityscapes_..._leftImg8bit_coi-car_iou029.json -> (base, ["car"], 0.29, False)
      cityscapes_..._leftImg8bit_coi-car-motorcycle-person_iou031.json -> (base, ["car","motorcycle","person"], 0.31, False)
      lvis_000000416451_gt.json -> (lvis_000000416451, [], None, True)
    """
    if not name.endswith(".json"):
        return None, [], None, False
    stem = name[:-5]  # remove .json
    if stem.endswith("_gt"):
        base_id = stem[:-3]  # remove _gt
        return base_id, [], None, True
    # ..._coi-A-B_iou031
    m = re.match(r"^(.+)_coi-(.+)_iou(\d+)$", stem)
    if not m:
        return None, [], None, False
    base_id, coi_str, iou_str = m.groups()
    coi = [c for c in coi_str.split("-") if c]
    score = int(iou_str) / 100.0 if iou_str else None
    return base_id, coi, score, False


def instance_type_from_shapes(shapes: list[dict], coi: list[str]) -> str:
    """
    Classify: 1C1I, 1CnI, nC1I, nCnI by 1 or >1 COI classes and 1 or >1 max instance per class.
    """
    if not shapes:
        return "1C1I"
    counts_per_label = defaultdict(int)
    for s in shapes:
        counts_per_label[s["label"]] += 1
    labels_in_file = list(counts_per_label.keys())
    n_classes = len(coi) if coi else len(labels_in_file)
    if n_classes == 0:
        n_classes = len(labels_in_file)
    max_instances = max(counts_per_label.values()) if counts_per_label else 1
    n_c = "1" if n_classes == 1 else "n"
    n_i = "1" if max_instances == 1 else "n"
    return f"{n_c}C{n_i}I"


def shapes_to_predictions(shapes: list[dict]) -> list[dict]:
    """Convert labelme shapes to predictions with polygon key."""
    return [
        {"label": s["label"], "polygon": s["points"]}
        for s in shapes
    ]


def build_annotations(base_ids: set[str]) -> list[dict]:
    """Scan error_annotations (max depth 1 = error_type folders), one annotation per JSON."""
    annotations = []
    seen_bases = set()
    error_type_dirs = [d for d in ERROR_ANNOTATIONS.iterdir() if d.is_dir()]
    for error_type_dir in sorted(error_type_dirs):
        error_type_name = error_type_dir.name
        for jpath in sorted(error_type_dir.glob("*.json")):
            name = jpath.name
            base_id, coi_from_name, final_score, is_gt = parse_error_json_filename(name)
            if base_id not in base_ids:
                continue
            data = json.loads(jpath.read_text())
            shapes = data.get("shapes", [])
            if is_gt:
                model_name = "gt"
                error_type = "gt"
                coi = list({s["label"] for s in shapes})
                final_score_val = 1.0  # or None; use 1.0 for gt
            else:
                model_name = "synthetic"
                error_type = error_type_name
                coi = coi_from_name
                final_score_val = final_score
            predictions = shapes_to_predictions(shapes)
            instance_type = instance_type_from_shapes(shapes, coi)
            ann = {
                "id": f"is_{len(annotations)}",
                "task": "instance_segmentation",
                "image_id": base_id,
                "coi": coi,
                "instance_type": instance_type,
                "model_name": model_name,
                "error_type": error_type,
                "final_score": final_score_val,
                "other_scores": {"mean_iou": final_score_val},
                "predictions_type": "polygon",
                "predictions": predictions,
            }
            annotations.append(ann)
    return annotations


def main() -> None:
    base_ids = load_base_images()
    base_id_set = set(base_ids)
    print(f"Loaded {len(base_ids)} base images from {BASE_IMAGES_LIST.name}")

    images = build_images(base_ids)
    print(f"Built {len(images)} image records, copied to {FINAL_IMAGES_DIR}")

    annotations = build_annotations(base_id_set)
    print(f"Built {len(annotations)} annotations from {ERROR_ANNOTATIONS}")

    out = {
        "images": images,
        "annotations": annotations,
    }
    OUTPUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

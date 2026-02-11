#!/usr/bin/env python3
"""
根据 error_annotations 下的 5 种 error type 数据，生成 pairwise comparison 的 metadata。

Pairwise 范围：同一 base image、同一 error type、同一 COI 下，不同 IoU 版本两两比较。
（GT 视为 IoU=1.0）
A/B 顺序随机；answer 为较好的一方。

media 路径：指向可视化图的约定路径（需运行 render_pairwise_vis.py 生成实际图片）。

用法:
  python generate_pairwise_metadata.py [--output pairwise_metadata.json] [--vis-dir vis]
"""

import argparse
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "instseg_data"
ERROR_DIR = BASE_DIR / "error_annotations"
SEED = 42

ERROR_TYPES = [
    "boundary_inaccuracy",
    "under_segmentation",
    "over_segmentation",
    "missed_instance",
    "label_confusion",
]

ERROR_TYPE_DISPLAY = {
    "boundary_inaccuracy": "boundary-inaccuracy",
    "under_segmentation": "under-segmentation",
    "over_segmentation": "over-segmentation",
    "missed_instance": "missed-instance",
    "label_confusion": "label-confusion",
}


def infer_scene(base_name: str) -> str:
    if base_name.startswith("cityscapes_"):
        return "outdoor-city"
    return ""


def infer_longtail(base_name: str) -> bool:
    return base_name.startswith("lvis_")


def vis_path(vis_dir: str, error_type: str, base_name: str, coi_str: str, iou: float, ext: str) -> str:
    """约定：error 版本为 xxx_iou075.jpg，GT 为 xxx_gt.jpg"""
    name = f"{base_name}_coi-{coi_str}"
    if iou >= 0.999:
        return f"{vis_dir}/{error_type}/{name}_gt{ext}"
    iou_int = int(round(iou * 100))
    return f"{vis_dir}/{error_type}/{name}_iou{iou_int:03d}{ext}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="pairwise_metadata.json")
    parser.add_argument("--vis-dir", "-v", default="vis", help="可视化图输出根目录")
    parser.add_argument("--ext", default=".jpg")
    args = parser.parse_args()

    vis_dir = args.vis_dir.strip("/")
    ext = args.ext if args.ext.startswith(".") else "." + args.ext
    random.seed(SEED)

    pattern = re.compile(r"(.+)_coi-(.+)_iou(\d+)\.json")
    groups = {}  # (base, error_type, coi_str) -> {iou: json_path, ...}

    for error_type in ERROR_TYPES:
        err_dir = ERROR_DIR / error_type
        if not err_dir.exists():
            continue
        for f in sorted(err_dir.glob("*_coi-*_iou*.json")):
            match = pattern.match(f.name)
            if not match:
                continue
            base_name, coi_str, iou_int = match.group(1), match.group(2), int(match.group(3))
            iou = iou_int / 100.0
            key = (base_name, error_type, coi_str)
            if key not in groups:
                groups[key] = {}
            groups[key][iou] = str(f)

    # 读取每个 group 的 COI 信息（从第一个 json 取）
    group_info = {}
    for key in groups:
        base_name, error_type, coi_str = key
        iou_vals = sorted(groups[key].keys())
        first_path = groups[key][iou_vals[0]]
        try:
            with open(first_path, "r") as fp:
                data = json.load(fp)
        except Exception:
            group_info[key] = None
            continue
        meta = data.get("_error_metadata") or {}
        coi_list = meta.get("coi") or [c.replace("_", " ") for c in coi_str.split("-")]
        task = ", ".join(coi_list)
        category = coi_list[0] if len(coi_list) == 1 else task
        group_info[key] = {
            "task": task,
            "category": category,
            "coi_list": coi_list,
        }

    # 对每个 group 生成两两比较
    out_list = []
    id_count = 0

    for (base_name, error_type, coi_str), iou_to_path in groups.items():
        info = group_info.get((base_name, error_type, coi_str))
        if not info:
            continue
        task = info["task"]
        category = info["category"]
        image_key = f"{base_name}_coi-{coi_str}"
        err_display = ERROR_TYPE_DISPLAY.get(error_type, error_type)

        # 加入 GT (1.0)
        ious = sorted(set(iou_to_path.keys()) | {1.0})
        if len(ious) < 2:
            continue

        for i in range(len(ious)):
            for j in range(i + 1, len(ious)):
                iou_lo, iou_hi = ious[i], ious[j]
                worse_iou, better_iou = iou_lo, iou_hi

                # 随机 A/B
                if random.random() < 0.5:
                    path_a = vis_path(vis_dir, error_type, base_name, coi_str, worse_iou, ext)
                    path_b = vis_path(vis_dir, error_type, base_name, coi_str, better_iou, ext)
                    answer = "B"
                    score_a, score_b = worse_iou, better_iou
                else:
                    path_a = vis_path(vis_dir, error_type, base_name, coi_str, better_iou, ext)
                    path_b = vis_path(vis_dir, error_type, base_name, coi_str, worse_iou, ext)
                    answer = "A"
                    score_a, score_b = better_iou, worse_iou

                id_count += 1
                entry = {
                    "prompt": (
                        f"Which of the images is a better instance segmentation result of {task}? "
                        f"The segmentation masks are shown on the images.\n"
                        f"A. <image>\nB. <image>\nPlease answer with A or B directly."
                    ),
                    "media": [path_a, path_b],
                    "choices": ["A", "B"],
                    "answer": answer,
                    "id": str(id_count),
                    "metadata": {
                        "image_key": image_key,
                        "base_image": base_name,
                        "category": category,
                        "task": task,
                        "error_type": err_display,
                        "iou_a": score_a,
                        "iou_b": score_b,
                        "score_good": max(score_a, score_b),
                        "score_bad": min(score_a, score_b),
                        "score_difference": round(abs(score_a - score_b), 4),
                    },
                }
                scene = infer_scene(base_name)
                if scene:
                    entry["metadata"]["image_type_scene"] = scene
                entry["metadata"]["image_type_longtail"] = infer_longtail(base_name)
                out_list.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out_list)} pairwise entries to {out_path}")


if __name__ == "__main__":
    main()

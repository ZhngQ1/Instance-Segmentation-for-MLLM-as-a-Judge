#!/usr/bin/env python3
"""
根据 error_annotations 下的 JSON 和图片，生成 pairwise 所需的可视化图（在原图上绘制 polygon）。

输出路径约定：vis/{error_type}/{base}_coi-{coi}_iou{iou:03d}.jpg 或 _gt.jpg

用法:
  python render_pairwise_vis.py [--vis-dir vis] [--dry-run]
  
不指定 --dry-run 时会实际生成图片。图片体积较大，可按需运行或分享脚本供他人生成。
"""

import argparse
import json
import re
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    print("请安装: pip install numpy pillow")
    raise

BASE_DIR = Path(__file__).resolve().parent / "instseg_data"
ERROR_DIR = BASE_DIR / "error_annotations"

ERROR_TYPES = [
    "boundary_inaccuracy",
    "under_segmentation",
    "over_segmentation",
    "missed_instance",
    "label_confusion",
]


def draw_polygons_on_image(img_path: Path, shapes: list, out_path: Path, alpha: float = 0.4) -> bool:
    """在原图上绘制 polygons 并保存"""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        return False
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = [
        (255, 0, 0, int(255 * alpha)),
        (0, 255, 0, int(255 * alpha)),
        (0, 0, 255, int(255 * alpha)),
        (255, 255, 0, int(255 * alpha)),
    ]
    for i, s in enumerate(shapes):
        pts = s.get("points", [])
        if len(pts) < 3:
            continue
        flat = [p for pt in pts for p in pt]
        c = colors[i % len(colors)]
        draw.polygon(flat, outline=c[:3] + (255,), fill=c)
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-dir", "-v", default="vis")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不生成")
    args = parser.parse_args()

    vis_dir = Path(args.vis_dir)
    pattern = re.compile(r"(.+)_coi-(.+)_iou(\d+)\.json")
    pattern_gt = re.compile(r"(.+)_coi-(.+)_gt\.json")

    to_render = []  # (img_path, json_path, out_path)

    for error_type in ERROR_TYPES:
        err_dir = ERROR_DIR / error_type
        if not err_dir.exists():
            continue
        for f in sorted(err_dir.glob("*.json")):
            if f.name.endswith("_gt.json"):
                m = pattern_gt.match(f.name)
                if not m:
                    continue
                base_name, coi_str = m.group(1), m.group(2)
                out_name = f"{base_name}_coi-{coi_str}_gt.jpg"
                out_path = vis_dir / error_type / out_name
            else:
                m = pattern.match(f.name)
                if not m:
                    continue
                base_name, coi_str, iou_int = m.group(1), m.group(2), m.group(3)
                out_name = f"{base_name}_coi-{coi_str}_iou{iou_int}.jpg"
                out_path = vis_dir / error_type / out_name

            # 找图片
            img_candidates = list(err_dir.glob(f"{base_name}.*"))
            img_candidates = [p for p in img_candidates if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if not img_candidates:
                continue
            img_path = img_candidates[0]
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
            except Exception:
                continue
            shapes = data.get("shapes", [])
            if not shapes:
                continue
            to_render.append((img_path, f, out_path, shapes))

    print(f"待渲染 {len(to_render)} 张图")
    if args.dry_run:
        return

    done, fail = 0, 0
    for img_path, json_path, out_path, shapes in to_render:
        if draw_polygons_on_image(img_path, shapes, out_path):
            done += 1
        else:
            fail += 1
            print(f"  failed: {out_path}")
    print(f"完成: {done}, 失败: {fail}")


if __name__ == "__main__":
    main()

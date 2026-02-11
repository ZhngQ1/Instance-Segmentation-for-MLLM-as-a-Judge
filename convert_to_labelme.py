#!/usr/bin/env python
"""
convert_to_labelme.py

Convert COCO, LVIS, and Cityscapes annotations to labelme format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    from labelme import __version__ as labelme_version
except ImportError:
    labelme_version = "5.0.0"

# 路径配置
BASE_DIR = Path("/Users/charleszhng/Desktop/inst seg/instseg_data")
IMG_DIR = BASE_DIR / "images"
ANN_DIR = BASE_DIR / "annotations"
OUTPUT_DIR = BASE_DIR / "labelme_annotations"

# Cityscapes instance类别（只保留这些作为instance）
CITYSCAPES_INSTANCE_LABELS = {
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
}


def flat_to_points(flat_coords: List[float]) -> List[List[float]]:
    """
    将COCO/LVIS的flat坐标列表转换为点列表
    [x1,y1,x2,y2,...] -> [[x1,y1], [x2,y2], ...]
    """
    points = []
    for i in range(0, len(flat_coords), 2):
        points.append([flat_coords[i], flat_coords[i+1]])
    return points


def convert_coco_to_labelme(ann_path: Path, img_dir: Path) -> Dict:
    """转换COCO格式标注到labelme格式"""
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    # 获取图片信息
    img_info = data['images'][0]
    img_height = img_info['height']
    img_width = img_info['width']
    
    # 获取原始文件名并找到对应的图片
    original_filename = img_info['file_name']
    # 图片在images目录中的命名是 coco_000000017714.jpg
    base_name = ann_path.stem  # coco_000000017714
    
    # 查找对应的图片文件
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = img_dir / f"{base_name}{ext}"
        if candidate.exists():
            img_path = candidate.name
            break
    
    if img_path is None:
        img_path = f"{base_name}.jpg"  # 默认假设jpg
    
    # 创建category_id到name的映射
    cat_id_to_name = {}
    for cat in data.get('categories', []):
        cat_id_to_name[cat['id']] = cat['name']
    
    # 转换annotations
    shapes = []
    for ann in data['annotations']:
        category_name = cat_id_to_name.get(ann['category_id'], f"category_{ann['category_id']}")
        
        # COCO的segmentation可能是多个polygon的列表
        segmentation = ann.get('segmentation', [])
        
        # 跳过RLE格式的segmentation
        if isinstance(segmentation, dict):
            continue
            
        for seg in segmentation:
            if len(seg) < 6:  # 至少需要3个点
                continue
            
            points = flat_to_points(seg)
            
            shape = {
                "label": category_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": ""
            }
            shapes.append(shape)
    
    # 创建labelme格式
    labelme_data = {
        "version": labelme_version,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    return labelme_data


def convert_lvis_to_labelme(ann_path: Path, img_dir: Path) -> Dict:
    """转换LVIS格式标注到labelme格式（与COCO类似）"""
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    # 获取图片信息
    img_info = data['images'][0]
    img_height = img_info['height']
    img_width = img_info['width']
    
    # 图片在images目录中的命名
    base_name = ann_path.stem  # lvis_000000000632
    
    # 查找对应的图片文件
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = img_dir / f"{base_name}{ext}"
        if candidate.exists():
            img_path = candidate.name
            break
    
    if img_path is None:
        img_path = f"{base_name}.jpg"
    
    # 创建category_id到name的映射
    cat_id_to_name = {}
    for cat in data.get('categories', []):
        cat_id_to_name[cat['id']] = cat['name']
    
    # 转换annotations
    shapes = []
    for ann in data['annotations']:
        category_name = cat_id_to_name.get(ann['category_id'], f"category_{ann['category_id']}")
        
        segmentation = ann.get('segmentation', [])
        
        # 跳过RLE格式
        if isinstance(segmentation, dict):
            continue
        
        for seg in segmentation:
            if len(seg) < 6:
                continue
            
            points = flat_to_points(seg)
            
            shape = {
                "label": category_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": ""
            }
            shapes.append(shape)
    
    labelme_data = {
        "version": labelme_version,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    return labelme_data


def convert_cityscapes_to_labelme(ann_path: Path, img_dir: Path) -> Dict:
    """转换Cityscapes格式标注到labelme格式"""
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    img_height = data['imgHeight']
    img_width = data['imgWidth']
    
    # 图片命名：cityscapes_aachen_aachen_000130_000019_leftImg8bit.png
    base_name = ann_path.stem  # cityscapes_aachen_aachen_000130_000019
    
    # 查找对应的图片文件
    img_path = None
    for suffix in ['_leftImg8bit.png', '.png', '.jpg']:
        candidate = img_dir / f"{base_name}{suffix}"
        if candidate.exists():
            img_path = candidate.name
            break
    
    if img_path is None:
        img_path = f"{base_name}_leftImg8bit.png"
    
    # 转换objects
    shapes = []
    for obj in data.get('objects', []):
        label = obj.get('label', 'unknown')
        
        # 只保留instance segmentation的类别
        if label not in CITYSCAPES_INSTANCE_LABELS:
            continue
        
        polygon = obj.get('polygon', [])
        if len(polygon) < 3:
            continue
        
        # Cityscapes的polygon已经是 [[x,y], [x,y], ...] 格式
        points = [[float(p[0]), float(p[1])] for p in polygon]
        
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
            "description": ""
        }
        shapes.append(shape)
    
    labelme_data = {
        "version": labelme_version,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    return labelme_data


def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 获取所有annotation文件
    ann_files = list(ANN_DIR.glob("*.json"))
    
    print(f"Found {len(ann_files)} annotation files")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    stats = {"coco": 0, "lvis": 0, "cityscapes": 0, "errors": 0}
    
    for ann_path in sorted(ann_files):
        try:
            filename = ann_path.name
            
            if filename.startswith("coco_"):
                labelme_data = convert_coco_to_labelme(ann_path, IMG_DIR)
                stats["coco"] += 1
            elif filename.startswith("lvis_"):
                labelme_data = convert_lvis_to_labelme(ann_path, IMG_DIR)
                stats["lvis"] += 1
            elif filename.startswith("cityscapes_"):
                labelme_data = convert_cityscapes_to_labelme(ann_path, IMG_DIR)
                stats["cityscapes"] += 1
            else:
                print(f"  Unknown format: {filename}")
                stats["errors"] += 1
                continue
            
            # 保存labelme格式的JSON
            output_path = OUTPUT_DIR / filename
            with open(output_path, 'w') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ {filename} -> {len(labelme_data['shapes'])} shapes")
            
        except Exception as e:
            print(f"✗ Error processing {ann_path.name}: {e}")
            stats["errors"] += 1
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 50)
    print("Conversion complete!")
    print(f"  COCO:       {stats['coco']} files")
    print(f"  LVIS:       {stats['lvis']} files")
    print(f"  Cityscapes: {stats['cityscapes']} files")
    print(f"  Errors:     {stats['errors']}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

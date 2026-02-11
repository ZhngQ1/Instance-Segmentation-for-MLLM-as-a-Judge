#!/usr/bin/env python3
"""
analyze_images.py - 分析100张图片的class和instance分布

分析原始annotations文件夹中的数据（COCO/LVIS/Cityscapes格式）
"""

import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/Users/charleszhng/Desktop/inst seg/instseg_data")
ANNOTATIONS_DIR = BASE_DIR / "annotations"  # 原始标注
LABELME_DIR = BASE_DIR / "gt_backup"        # labelme格式（处理后）

# Cityscapes instance类别
CITYSCAPES_INSTANCE_LABELS = {
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
}

def analyze_coco_lvis(json_path):
    """分析COCO/LVIS格式的标注"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取category映射
    cat_id_to_name = {}
    for cat in data.get('categories', []):
        cat_id_to_name[cat['id']] = cat['name']
    
    # 统计每个class的instance数量
    class_counts = Counter()
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")
        class_counts[cat_name] += 1
    
    # 获取图片尺寸
    img_info = data.get('images', [{}])[0]
    width = img_info.get('width', 0)
    height = img_info.get('height', 0)
    
    return class_counts, width, height

def analyze_cityscapes(json_path, instance_only=True):
    """分析Cityscapes格式的标注"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    width = data.get('imgWidth', 0)
    height = data.get('imgHeight', 0)
    
    class_counts = Counter()
    all_class_counts = Counter()  # 包含所有类别
    
    for obj in data.get('objects', []):
        label = obj.get('label', 'unknown')
        all_class_counts[label] += 1
        
        if not instance_only or label in CITYSCAPES_INSTANCE_LABELS:
            class_counts[label] += 1
    
    return class_counts, all_class_counts, width, height

def categorize_classes(class_counts):
    """将classes分为1-instance和multi-instance两类"""
    single_instance_classes = []
    multi_instance_classes = []
    
    for cls, count in class_counts.items():
        if count == 1:
            single_instance_classes.append(cls)
        else:
            multi_instance_classes.append((cls, count))
    
    return single_instance_classes, multi_instance_classes

def main():
    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    
    print(f"分析原始annotations文件夹中的 {len(json_files)} 个文件...\n")
    print("=" * 80)
    
    good_images = []
    only_multi = []
    only_single = []
    
    all_stats = []
    
    for json_path in json_files:
        filename = json_path.name
        
        try:
            if filename.startswith("cityscapes_"):
                # Cityscapes格式
                instance_counts, all_counts, width, height = analyze_cityscapes(json_path, instance_only=True)
                class_counts = instance_counts
                extra_info = {
                    'all_classes_count': len(all_counts),
                    'all_instances_count': sum(all_counts.values()),
                    'non_instance_classes': [c for c in all_counts if c not in CITYSCAPES_INSTANCE_LABELS]
                }
            else:
                # COCO/LVIS格式
                class_counts, width, height = analyze_coco_lvis(json_path)
                extra_info = {}
            
            single_classes, multi_classes = categorize_classes(class_counts)
            
            stats = {
                'name': json_path.stem,
                'source': 'cityscapes' if filename.startswith('cityscapes_') else ('lvis' if filename.startswith('lvis_') else 'coco'),
                'total_classes': len(class_counts),
                'total_instances': sum(class_counts.values()),
                'single_instance_classes': single_classes,
                'multi_instance_classes': multi_classes,
                'class_counts': dict(class_counts),
                'image_size': f"{width}x{height}",
                **extra_info
            }
            all_stats.append(stats)
            
            if single_classes and multi_classes:
                good_images.append(stats)
            elif multi_classes:
                only_multi.append(stats)
            else:
                only_single.append(stats)
                
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")
    
    # 输出结果
    print("\n" + "=" * 80)
    print("【符合条件的图片】同时有1-instance和multi-instance classes")
    print("（注：Cityscapes只统计instance类别）")
    print("=" * 80)
    print(f"共 {len(good_images)} 张\n")
    
    for img in good_images[:15]:
        print(f"\n📷 {img['name']} [{img['source']}]")
        print(f"   图片尺寸: {img['image_size']}")
        print(f"   总classes: {img['total_classes']}, 总instances: {img['total_instances']}")
        print(f"   1-instance classes ({len(img['single_instance_classes'])}): {img['single_instance_classes'][:5]}{'...' if len(img['single_instance_classes']) > 5 else ''}")
        print(f"   multi-instance classes ({len(img['multi_instance_classes'])}): {img['multi_instance_classes'][:5]}{'...' if len(img['multi_instance_classes']) > 5 else ''}")
        if 'all_classes_count' in img:
            print(f"   [Cityscapes原始: {img['all_classes_count']}类, {img['all_instances_count']}个objects]")
    
    if len(good_images) > 15:
        print(f"\n... 还有 {len(good_images) - 15} 张图片")
    
    print("\n" + "=" * 80)
    print("【只有multi-instance classes的图片】")
    print("=" * 80)
    print(f"共 {len(only_multi)} 张")
    for img in only_multi[:5]:
        print(f"  - {img['name']} [{img['source']}]: {img['multi_instance_classes']}")
    
    print("\n" + "=" * 80)
    print("【只有1-instance classes的图片】")
    print("=" * 80)
    print(f"共 {len(only_single)} 张")
    for img in only_single[:5]:
        print(f"  - {img['name']} [{img['source']}]: {img['single_instance_classes'][:5]}")
    
    # 按数据源统计
    print("\n" + "=" * 80)
    print("【按数据源统计】")
    print("=" * 80)
    
    for source in ['coco', 'lvis', 'cityscapes']:
        source_good = [img for img in good_images if img['source'] == source]
        source_multi = [img for img in only_multi if img['source'] == source]
        source_single = [img for img in only_single if img['source'] == source]
        total = len(source_good) + len(source_multi) + len(source_single)
        print(f"\n  {source.upper()} (共{total}张):")
        print(f"    符合条件: {len(source_good)} 张")
        print(f"    只有multi-inst: {len(source_multi)} 张")
        print(f"    只有1-inst: {len(source_single)} 张")
    
    # 保存详细结果
    output_path = BASE_DIR / "image_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            'good_images': good_images,
            'only_multi': only_multi,
            'only_single': only_single,
            'summary': {
                'total': len(json_files),
                'good': len(good_images),
                'only_multi': len(only_multi),
                'only_single': len(only_single)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    # 推荐图片
    print("\n" + "=" * 80)
    print("【推荐用于data curation的图片】")
    print("筛选条件：>=2个1-inst class, >=1个multi-inst class, 5-50个instances")
    print("=" * 80)
    
    recommended = [
        img for img in good_images 
        if len(img['single_instance_classes']) >= 2 
        and len(img['multi_instance_classes']) >= 1
        and img['total_instances'] >= 5
        and img['total_instances'] <= 50
    ]
    
    print(f"共 {len(recommended)} 张\n")
    
    for i, img in enumerate(recommended[:15], 1):
        print(f"{i:2d}. {img['name']} [{img['source']}]")
        print(f"    instances: {img['total_instances']}, classes: {img['total_classes']}")
        print(f"    1-inst: {img['single_instance_classes']}")
        print(f"    multi-inst: {img['multi_instance_classes']}")
        print()

if __name__ == "__main__":
    main()

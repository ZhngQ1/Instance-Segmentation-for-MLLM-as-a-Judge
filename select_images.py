"""
从COCO, LVIS, Cityscapes中筛选100张图片用于instance segmentation data curation
- 50张COCO标注
- 25张LVIS标注（不与COCO重复）
- 25张Cityscapes
每张图片至少有3个instance
"""

import json
import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# 设置随机种子以保证可复现
random.seed(42)

# 路径配置
BASE_DIR = Path("/Users/charleszhng/Desktop/inst seg")
COCO_IMG_DIR = BASE_DIR / "coco/val2017"
COCO_ANN_FILE = BASE_DIR / "coco/annotations/instances_val2017.json"
LVIS_ANN_FILE = BASE_DIR / "lvis/lvis_v1_val.json"
CITYSCAPES_IMG_DIR = BASE_DIR / "cityscape/leftImg8bit_trainvaltest/leftImg8bit"
CITYSCAPES_GT_DIR = BASE_DIR / "cityscape/gtFine_trainvaltest/gtFine"

# 输出目录
OUTPUT_IMG_DIR = BASE_DIR / "selected_images"
OUTPUT_ANN_DIR = BASE_DIR / "selected_annotations"

# Cityscapes instance类别
CITYSCAPES_INSTANCE_LABELS = {
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
}

MIN_INSTANCES = 3

def load_coco_annotations():
    """加载COCO标注并按image_id分组"""
    print("Loading COCO annotations...")
    with open(COCO_ANN_FILE, 'r') as f:
        coco_data = json.load(f)
    
    # 创建image_id到image信息的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 创建category_id到category信息的映射
    cat_id_to_info = {cat['id']: cat for cat in coco_data['categories']}
    
    # 按image_id分组annotations
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    return img_id_to_info, cat_id_to_info, img_to_anns

def load_lvis_annotations():
    """加载LVIS标注并按image_id分组"""
    print("Loading LVIS annotations...")
    with open(LVIS_ANN_FILE, 'r') as f:
        lvis_data = json.load(f)
    
    # 创建image_id到image信息的映射
    img_id_to_info = {img['id']: img for img in lvis_data['images']}
    
    # 创建category_id到category信息的映射
    cat_id_to_info = {cat['id']: cat for cat in lvis_data['categories']}
    
    # 按image_id分组annotations
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    return img_id_to_info, cat_id_to_info, img_to_anns

def get_cityscapes_files():
    """获取Cityscapes的文件列表并统计instance数量"""
    print("Scanning Cityscapes files...")
    cityscapes_files = []
    
    for split in ['train', 'val']:
        gt_split_dir = CITYSCAPES_GT_DIR / split
        img_split_dir = CITYSCAPES_IMG_DIR / split
        
        if not gt_split_dir.exists():
            continue
            
        for city_dir in gt_split_dir.iterdir():
            if not city_dir.is_dir():
                continue
            
            for json_file in city_dir.glob("*_gtFine_polygons.json"):
                # 统计instance数量
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                instance_count = sum(
                    1 for obj in data.get('objects', [])
                    if obj.get('label') in CITYSCAPES_INSTANCE_LABELS
                )
                
                if instance_count >= MIN_INSTANCES:
                    # 构建对应的图片路径
                    base_name = json_file.stem.replace('_gtFine_polygons', '')
                    img_file = img_split_dir / city_dir.name / f"{base_name}_leftImg8bit.png"
                    
                    if img_file.exists():
                        cityscapes_files.append({
                            'json_path': json_file,
                            'img_path': img_file,
                            'instance_count': instance_count,
                            'city': city_dir.name
                        })
    
    return cityscapes_files

def create_per_image_json(img_info, annotations, categories, source='coco'):
    """为单张图片创建独立的JSON标注文件"""
    # 获取这张图片用到的category ids
    used_cat_ids = set(ann['category_id'] for ann in annotations)
    used_categories = [categories[cat_id] for cat_id in used_cat_ids if cat_id in categories]
    
    return {
        'info': {
            'description': f'{source.upper()} annotation for single image',
            'source': source
        },
        'images': [img_info],
        'annotations': annotations,
        'categories': used_categories
    }

def main():
    # 清空并创建输出目录
    if OUTPUT_IMG_DIR.exists():
        shutil.rmtree(OUTPUT_IMG_DIR)
    if OUTPUT_ANN_DIR.exists():
        shutil.rmtree(OUTPUT_ANN_DIR)
    OUTPUT_IMG_DIR.mkdir(exist_ok=True)
    OUTPUT_ANN_DIR.mkdir(exist_ok=True)
    
    # 获取COCO val2017中存在的所有图片
    existing_coco_imgs = set(f.name for f in COCO_IMG_DIR.glob("*.jpg"))
    print(f"Existing images in COCO val2017: {len(existing_coco_imgs)}")
    
    # 加载COCO数据
    coco_img_info, coco_cat_info, coco_img_to_anns = load_coco_annotations()
    
    # 加载LVIS数据
    lvis_img_info, lvis_cat_info, lvis_img_to_anns = load_lvis_annotations()
    
    # 筛选COCO中至少有MIN_INSTANCES个instance的图片
    coco_valid_imgs = [
        img_id for img_id, anns in coco_img_to_anns.items()
        if len(anns) >= MIN_INSTANCES
    ]
    print(f"COCO images with >= {MIN_INSTANCES} instances: {len(coco_valid_imgs)}")
    
    # 筛选LVIS中至少有MIN_INSTANCES个instance的图片，且图片必须存在于val2017
    lvis_valid_imgs = []
    for img_id, anns in lvis_img_to_anns.items():
        if len(anns) >= MIN_INSTANCES:
            img_info = lvis_img_info.get(img_id)
            if img_info:
                file_name = img_info.get('file_name') or img_info.get('coco_url', '').split('/')[-1]
                if file_name in existing_coco_imgs:
                    lvis_valid_imgs.append(img_id)
    print(f"LVIS images with >= {MIN_INSTANCES} instances (in val2017): {len(lvis_valid_imgs)}")
    
    # 获取Cityscapes文件
    cityscapes_files = get_cityscapes_files()
    print(f"Cityscapes images with >= {MIN_INSTANCES} instances: {len(cityscapes_files)}")
    
    # 随机选择50张COCO图片
    random.shuffle(coco_valid_imgs)
    selected_coco_ids = coco_valid_imgs[:50]
    print(f"\nSelected {len(selected_coco_ids)} COCO images")
    
    # 从LVIS中选择25张不与COCO重复的图片
    lvis_unique_imgs = [
        img_id for img_id in lvis_valid_imgs
        if img_id not in selected_coco_ids
    ]
    random.shuffle(lvis_unique_imgs)
    selected_lvis_ids = lvis_unique_imgs[:25]
    print(f"Selected {len(selected_lvis_ids)} LVIS images (excluding COCO)")
    
    # 随机选择25张Cityscapes图片
    random.shuffle(cityscapes_files)
    selected_cityscapes = cityscapes_files[:25]
    print(f"Selected {len(selected_cityscapes)} Cityscapes images")
    
    # 复制文件并创建标注
    print("\n--- Processing COCO images ---")
    for i, img_id in enumerate(selected_coco_ids):
        img_info = coco_img_info[img_id]
        file_name = img_info['file_name']
        
        # 复制图片
        src_img = COCO_IMG_DIR / file_name
        dst_img = OUTPUT_IMG_DIR / f"coco_{file_name}"
        shutil.copy2(src_img, dst_img)
        
        # 创建per-image JSON
        annotations = coco_img_to_anns[img_id]
        json_data = create_per_image_json(img_info, annotations, coco_cat_info, 'coco')
        
        json_name = f"coco_{Path(file_name).stem}.json"
        with open(OUTPUT_ANN_DIR / json_name, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/50 COCO images")
    
    print("\n--- Processing LVIS images ---")
    for i, img_id in enumerate(selected_lvis_ids):
        img_info = lvis_img_info[img_id]
        # LVIS使用coco_url字段，文件名格式可能不同
        file_name = img_info.get('file_name') or img_info.get('coco_url', '').split('/')[-1]
        
        # LVIS图片实际上在COCO目录中
        src_img = COCO_IMG_DIR / file_name
        if not src_img.exists():
            print(f"  Warning: Image not found: {src_img}")
            continue
            
        dst_img = OUTPUT_IMG_DIR / f"lvis_{file_name}"
        shutil.copy2(src_img, dst_img)
        
        # 创建per-image JSON
        annotations = lvis_img_to_anns[img_id]
        json_data = create_per_image_json(img_info, annotations, lvis_cat_info, 'lvis')
        
        json_name = f"lvis_{Path(file_name).stem}.json"
        with open(OUTPUT_ANN_DIR / json_name, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/25 LVIS images")
    
    print("\n--- Processing Cityscapes images ---")
    for i, cs_file in enumerate(selected_cityscapes):
        # 复制图片
        src_img = cs_file['img_path']
        img_name = f"cityscapes_{cs_file['city']}_{src_img.name}"
        dst_img = OUTPUT_IMG_DIR / img_name
        shutil.copy2(src_img, dst_img)
        
        # 复制JSON（保持原格式）
        json_name = f"cityscapes_{cs_file['city']}_{cs_file['json_path'].stem.replace('_gtFine_polygons', '')}.json"
        with open(cs_file['json_path'], 'r') as f:
            json_data = json.load(f)
        
        # 添加source标识
        json_data['source'] = 'cityscapes'
        
        with open(OUTPUT_ANN_DIR / json_name, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/25 Cityscapes images")
    
    print("\n" + "="*50)
    print("Selection complete!")
    print(f"Images saved to: {OUTPUT_IMG_DIR}")
    print(f"Annotations saved to: {OUTPUT_ANN_DIR}")
    
    # 统计
    total_imgs = len(list(OUTPUT_IMG_DIR.glob("*")))
    total_jsons = len(list(OUTPUT_ANN_DIR.glob("*.json")))
    print(f"\nTotal images: {total_imgs}")
    print(f"Total JSON files: {total_jsons}")

if __name__ == "__main__":
    main()

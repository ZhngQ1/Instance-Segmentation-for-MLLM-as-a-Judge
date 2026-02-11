#!/usr/bin/env python3
"""
为labelme的IoU功能创建 _gt.json 文件

labelme需要 xxx_gt.json 文件来计算IoU：
- 打开 image.jpg 时，会自动查找 image_gt.json
- 如果找到，会显示当前polygon与GT的IoU

这个脚本：
1. 把 gt_backup/ 中的GT文件复制到 images/ 文件夹，命名为 xxx_gt.json
"""

import shutil
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent / "instseg_data"
GT_BACKUP_DIR = BASE_DIR / "gt_backup"
IMAGES_DIR = BASE_DIR / "images"


def main():
    if not GT_BACKUP_DIR.exists():
        print(f"❌ GT backup目录不存在: {GT_BACKUP_DIR}")
        return
    
    gt_files = list(GT_BACKUP_DIR.glob("*.json"))
    print(f"找到 {len(gt_files)} 个GT文件")
    
    created_count = 0
    for gt_file in gt_files:
        # 原文件名: coco_000000017714.json
        # 目标文件名: coco_000000017714_gt.json
        base_name = gt_file.stem  # coco_000000017714
        gt_target = IMAGES_DIR / f"{base_name}_gt.json"
        
        if gt_target.exists():
            print(f"  跳过 (已存在): {gt_target.name}")
            continue
        
        shutil.copy2(gt_file, gt_target)
        print(f"  ✓ 创建: {gt_target.name}")
        created_count += 1
    
    print(f"\n完成！共创建 {created_count} 个 _gt.json 文件")
    print(f"现在用labelme打开 images/ 文件夹时，应该能看到IoU统计了")


if __name__ == "__main__":
    main()

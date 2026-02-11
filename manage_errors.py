#!/usr/bin/env python3
"""
manage_errors.py - 管理error type和COI的JSON标注文件

用法:
    python manage_errors.py prepare <image_name> <error_type> --coi <class1,class2,...>
    python manage_errors.py save <image_name> <error_type> <iou> --coi <class1,class2,...>
    python manage_errors.py restore <image_name>
    python manage_errors.py status [image_name]
    python manage_errors.py list
    python manage_errors.py classes <image_name>    # 查看图片中有哪些classes
    python manage_errors.py fix_gt                  # 为所有error data创建COI-specific GT文件
    python manage_errors.py restore_all             # 恢复所有图片的xxx.json和xxx_gt.json
    python manage_errors.py check_images            # 检查images下JSON是否与gt_backup一致
    python manage_errors.py derive <image> <error_type> --from-coi "A|B" --to-coi "A" [--from-iou 0.5]

Error types (手动操作):
    boundary_inaccuracy, under_segmentation, over_segmentation

COI (Class of Interest):
    指定要操作的classes，用逗号分隔，如: banana,cup,knife
    prepare时会只保留COI相关的polygons
    save时会记录COI到文件名和JSON中

示例:
    # 1. 查看图片有哪些classes
    python manage_errors.py classes coco_000000017714
    
    # 2. 准备编辑，指定COI
    python manage_errors.py prepare coco_000000017714 boundary_inaccuracy --coi banana,cup,knife
    
    # 3. 在labelme中编辑...
    
    # 4. 保存，指定IoU（从labelme状态栏读取）
    python manage_errors.py save coco_000000017714 boundary_inaccuracy 0.75 --coi banana,cup,knife
"""

import sys
import shutil
import json
import re
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/charleszhng/Desktop/inst seg/instseg_data")
IMAGES_DIR = BASE_DIR / "images"
GT_BACKUP_DIR = BASE_DIR / "gt_backup"
ERROR_DIR = BASE_DIR / "error_annotations"

# 手动操作的error types
MANUAL_ERROR_TYPES = [
    "boundary_inaccuracy",
    "under_segmentation", 
    "over_segmentation",
]

# 可自动化的error types (后续实现)
AUTO_ERROR_TYPES = [
    "missed_instance",
    "label_confusion",
]

ALL_ERROR_TYPES = MANUAL_ERROR_TYPES + AUTO_ERROR_TYPES


def find_image_file(image_name):
    """找到对应的图片文件"""
    for ext in ['.jpg', '.png', '.jpeg']:
        for f in IMAGES_DIR.glob(f"*{ext}"):
            if '_gt' in f.stem:
                continue
            if f.stem == image_name or image_name in f.stem or f.stem in image_name:
                return f
    return None


def parse_coi(coi_str):
    """解析COI字符串为列表
    
    支持两种分隔符：
    - 逗号分隔：banana,cup,knife
    - 竖线分隔（用于包含空格的class名）：dining table|banana|cup
    """
    if not coi_str:
        return []
    # 如果包含竖线，用竖线分隔；否则用逗号
    if '|' in coi_str:
        return [c.strip() for c in coi_str.split('|') if c.strip()]
    return [c.strip() for c in coi_str.split(',') if c.strip()]


def coi_to_str(coi_list):
    """COI列表转为文件名友好的字符串
    
    空格会被替换为下划线
    """
    # 替换空格为下划线，然后用-连接
    safe_names = [c.replace(' ', '_') for c in sorted(coi_list)]
    return '-'.join(safe_names)


def get_error_path(base_name, error_type, coi_list, iou):
    """获取error annotation的路径"""
    coi_str = coi_to_str(coi_list)
    iou_str = f"{float(iou):.2f}".replace('.', '')
    return ERROR_DIR / error_type / f"{base_name}_coi-{coi_str}_iou{iou_str}.json"


def coi_from_filename(coi_str):
    """从文件名中的coi字符串解析为class名列表"""
    return [c.replace('_', ' ') for c in coi_str.split('-')]


def list_existing_versions(base_name, error_type):
    """列出某个图片某个error type的所有已存在版本"""
    error_dir = ERROR_DIR / error_type
    if not error_dir.exists():
        return []
    
    versions = []
    # 匹配新格式: _coi-xxx_iou075.json
    pattern = re.compile(rf"{re.escape(base_name)}_coi-(.+)_iou(\d+)\.json")
    
    for f in error_dir.glob(f"{base_name}_*.json"):
        if f.name.endswith("_gt.json"):
            continue
        match = pattern.match(f.name)
        if match:
            coi_str = match.group(1)
            iou = int(match.group(2)) / 100.0
            coi_list = coi_str.split('-')
            versions.append((coi_list, iou, f))
    
    return sorted(versions, key=lambda x: (len(x[0]), x[1]), reverse=True)


def get_image_classes(image_name):
    """获取图片中所有的classes及其instance数量"""
    img_file = find_image_file(image_name)
    if not img_file:
        return None
    
    base_name = img_file.stem
    gt_json = GT_BACKUP_DIR / f"{base_name}.json"
    
    if not gt_json.exists():
        return None
    
    with open(gt_json, 'r') as f:
        data = json.load(f)
    
    class_counts = {}
    for shape in data.get('shapes', []):
        label = shape.get('label', 'unknown')
        class_counts[label] = class_counts.get(label, 0) + 1
    
    return class_counts


def get_image_batch(n_single, n_multi):
    """根据single/multi instance class数量判断图片所属批次"""
    if n_single >= 2 and n_multi >= 2:
        return 1, "第一批（最佳）", ">=2 single + >=2 multi"
    elif n_single >= 2 and n_multi >= 1:
        return 2, "第二批", ">=2 single + >=1 multi"
    elif n_single >= 1 and n_multi >= 2:
        return 3, "第三批", ">=1 single + >=2 multi"
    elif n_single >= 1 and n_multi >= 1:
        return 4, "第四批", ">=1 single + >=1 multi"
    else:
        return 5, "第五批", "其他"


def show_classes(image_name):
    """显示图片中的classes并根据批次给出处理建议"""
    class_counts = get_image_classes(image_name)
    if class_counts is None:
        print(f"❌ 找不到图片或GT: {image_name}")
        return
    
    print(f"📷 图片: {image_name}")
    print(f"\nClasses ({len(class_counts)} 个):")
    
    # 按instance数量排序
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    single_inst = []
    multi_inst = []
    
    for label, count in sorted_classes:
        if count == 1:
            single_inst.append(label)
            print(f"  {label}: {count} (single)")
        else:
            multi_inst.append((label, count))
            print(f"  {label}: {count} (multi)")
    
    # 判断批次
    batch_num, batch_name, batch_condition = get_image_batch(len(single_inst), len(multi_inst))
    
    print(f"\n{'='*60}")
    print(f"📊 批次: {batch_name} ({batch_condition})")
    print(f"{'='*60}")
    
    # 检查是否有带空格的class名
    has_space = any(' ' in c for c in class_counts.keys())
    sep = '|' if has_space else ','
    
    # 根据批次给出不同的建议
    if batch_num == 1:
        # 第一批：完整处理4种question types
        print(f"\n✅ 推荐: 完整处理4种Question Types")
        print(f"   (boundary_inaccuracy, under_segmentation, over_segmentation)")
        print(f"\n推荐的COI组合:")
        
        # 1c1i
        print(f"  1c1i:     --coi \"{single_inst[0]}\"")
        # 1c-multi
        print(f"  1c-multi: --coi \"{multi_inst[0][0]}\"")
        # Nc1i
        nc1i_classes = single_inst[:2]
        print(f"  Nc1i:     --coi \"{sep.join(nc1i_classes)}\"")
        # Nc-multi
        nc_multi_classes = [multi_inst[0][0], single_inst[0]]
        print(f"  Nc-multi: --coi \"{sep.join(nc_multi_classes)}\"")
        
        print(f"\n⚠️  注意: under_segmentation 不适用于 1c1i 和 Nc1i")
    
    elif batch_num == 2:
        # 第二批：简化处理 1c1i, Nc1i
        print(f"\n✅ 推荐: 简化处理 - 专注 1c1i 和 Nc1i")
        print(f"   (充分利用多个single-instance classes)")
        print(f"\n推荐的COI组合:")
        
        # 1c1i
        print(f"  1c1i:     --coi \"{single_inst[0]}\"")
        # Nc1i
        nc1i_classes = single_inst[:2]
        print(f"  Nc1i:     --coi \"{sep.join(nc1i_classes)}\"")
        
        print(f"\n⚠️  注意: under_segmentation 不适用于这两种question types")
        print(f"         只需要做 boundary_inaccuracy 和 over_segmentation")
        
        # 可选补充
        if multi_inst:
            print(f"\n📌 可选补充 (如果有时间):")
            print(f"  1c-multi: --coi \"{multi_inst[0][0]}\"")
    
    elif batch_num == 3:
        # 第三批：简化处理 1c-multi, Nc-multi
        print(f"\n✅ 推荐: 简化处理 - 专注 1c-multi 和 Nc-multi")
        print(f"   (充分利用多个multi-instance classes)")
        print(f"\n推荐的COI组合:")
        
        # 1c-multi
        print(f"  1c-multi: --coi \"{multi_inst[0][0]}\"")
        # Nc-multi
        nc_multi_classes = [c[0] for c in multi_inst[:2]]
        print(f"  Nc-multi: --coi \"{sep.join(nc_multi_classes)}\"")
        
        print(f"\n✓ 这两种都可以做全部3种error types")
        
        # 可选补充
        if single_inst:
            print(f"\n📌 可选补充 (如果有时间):")
            print(f"  1c1i:     --coi \"{single_inst[0]}\"")
    
    elif batch_num == 4:
        # 第四批：基本处理
        print(f"\n✅ 推荐: 基本处理")
        print(f"\n推荐的COI组合:")
        
        # 1c1i
        if single_inst:
            print(f"  1c1i:     --coi \"{single_inst[0]}\"")
        # 1c-multi
        if multi_inst:
            print(f"  1c-multi: --coi \"{multi_inst[0][0]}\"")
        
        print(f"\n⚠️  Nc1i 和 Nc-multi 难以满足，可跳过")
    
    else:
        # 第五批：其他
        print(f"\n⚠️  此图片不太适合完整的diversity要求")
        print(f"   single classes: {len(single_inst)}, multi classes: {len(multi_inst)}")
        
        if single_inst:
            print(f"\n可用的COI:")
            print(f"  1c1i: --coi \"{single_inst[0]}\"")
        if multi_inst:
            print(f"  1c-multi: --coi \"{multi_inst[0][0]}\"")
    
    # IoU建议
    print(f"\n{'='*60}")
    print(f"📏 IoU建议:")
    if batch_num <= 2:
        print(f"   简化版: 0.5, 0.75 (2个levels)")
    else:
        print(f"   简化版: 0.5, 0.75 (2个levels)")
    print(f"   完整版: 0.25, 0.5, 0.75 (3个levels)")


def filter_shapes_by_coi(shapes, coi_list):
    """只保留COI相关的shapes"""
    if not coi_list:
        return shapes
    coi_set = set(coi_list)
    return [s for s in shapes if s.get('label', '') in coi_set]


def calc_iou_guidance(error_type, n_polygons, target_ious=[0.75, 0.50, 0.25]):
    """计算达到目标IoU需要的操作
    
    Over-segmentation: IoU = n_original / n_final
        - 拆分polygons，n_final > n_original
        - n_final = n_original / IoU
        
    Under-segmentation: IoU = n_final / n_original  
        - 合并polygons，n_final < n_original
        - n_final = n_original * IoU
    """
    guidance = []
    n = n_polygons
    
    if error_type == "over_segmentation":
        for iou in target_ious:
            n_final = n / iou
            n_add = n_final - n
            # 需要拆分的数量（每个polygon拆成2个会增加1个）
            guidance.append({
                'iou': iou,
                'n_final': int(round(n_final)),
                'change': f"+{int(round(n_add))} (拆分)",
                'desc': f"{n}→{int(round(n_final))} polygons"
            })
    
    elif error_type == "under_segmentation":
        if n <= 1:
            return None  # under-seg需要多个polygons
        for iou in target_ious:
            n_final = n * iou
            if n_final < 1:
                n_final = 1
            n_remove = n - n_final
            guidance.append({
                'iou': iou,
                'n_final': max(1, int(round(n_final))),
                'change': f"-{int(round(n_remove))} (合并)",
                'desc': f"{n}→{max(1, int(round(n_final)))} polygons"
            })
    
    return guidance


def prepare(image_name, error_type, coi_list):
    """准备编辑某个error type，只保留COI相关的polygons"""
    if error_type not in MANUAL_ERROR_TYPES:
        print(f"❌ 无效的error type: {error_type}")
        print(f"   可用类型 (手动): {', '.join(MANUAL_ERROR_TYPES)}")
        return False
    
    if not coi_list:
        print("❌ 必须指定COI: --coi class1,class2,...")
        return False
    
    img_file = find_image_file(image_name)
    if not img_file:
        print(f"❌ 找不到图片: {image_name}")
        return False
    
    base_name = img_file.stem
    target_json = IMAGES_DIR / f"{base_name}.json"
    gt_json = GT_BACKUP_DIR / f"{base_name}.json"
    
    if not gt_json.exists():
        print(f"❌ 找不到GT备份: {gt_json}")
        return False
    
    # 读取GT
    with open(gt_json, 'r') as f:
        data = json.load(f)
    
    # 检查COI是否有效
    all_labels = set(s.get('label', '') for s in data.get('shapes', []))
    invalid_coi = [c for c in coi_list if c not in all_labels]
    if invalid_coi:
        print(f"⚠️  以下COI在图片中不存在: {invalid_coi}")
        print(f"   图片中的classes: {sorted(all_labels)}")
        return False
    
    # 只保留COI相关的shapes
    original_count = len(data.get('shapes', []))
    data['shapes'] = filter_shapes_by_coi(data.get('shapes', []), coi_list)
    filtered_count = len(data['shapes'])
    
    # 保存到工作目录
    with open(target_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 同时更新_gt.json，也只保留COI相关的shapes
    # 这样labelme计算的IoU就是只针对COI的
    gt_target = IMAGES_DIR / f"{base_name}_gt.json"
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    gt_data['shapes'] = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
    with open(gt_target, 'w') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 已更新GT文件只包含COI: {gt_target.name}")
    
    # 列出已存在的版本
    existing_versions = list_existing_versions(base_name, error_type)
    if existing_versions:
        print(f"📋 已存在的 {error_type} 版本:")
        for coi, iou, path in existing_versions:
            print(f"   COI={coi}, IoU={iou:.2f}: {path.name}")
        print()
    
    print(f"✓ 已准备编辑 {error_type}")
    print(f"  COI: {coi_list}")
    print(f"  保留的polygons: {filtered_count}/{original_count}")
    print(f"  目标: {target_json}")
    
    # 显示IoU操作指南（仅对over_seg和under_seg）
    if error_type in ["over_segmentation", "under_segmentation"]:
        guidance = calc_iou_guidance(error_type, filtered_count)
        if guidance:
            print(f"\n{'='*50}")
            print(f"📏 IoU 操作指南 ({error_type})")
            print(f"   当前: {filtered_count} polygons")
            print(f"{'='*50}")
            for g in guidance:
                print(f"   IoU {g['iou']:.2f}: {g['desc']} ({g['change']})")
            
            if error_type == "over_segmentation":
                print(f"\n💡 提示: 选择polygon后用Edit→拆分，或删除后重画多个")
            else:
                print(f"\n💡 提示: 删除部分polygons，或合并相邻的polygons")
        elif error_type == "under_segmentation":
            print(f"\n⚠️  under_segmentation 需要 >1 个polygons，当前只有 {filtered_count} 个")
    
    print(f"\n现在可以用labelme编辑:")
    print(f'  labelme "{img_file}"')
    print(f"\n编辑完成后，从状态栏读取IoU值，然后运行:")
    # 使用竖线分隔如果有空格
    has_space = any(' ' in c for c in coi_list)
    sep = '|' if has_space else ','
    coi_arg = sep.join(coi_list)
    print(f'  python manage_errors.py save "{base_name}" {error_type} <iou> --coi "{coi_arg}"')
    return True


def derive(image_name, error_type, from_coi_list, to_coi_list, from_iou=None):
    """从已有 COI 的 error 结果衍生到子 COI，作为编辑起点
    
    例如：已有 car+person 的 over_seg，可衍生为 car 单独或 person 单独，
    直接保留该类已有的 polygon（含 over-seg），无需重新拆分。
    """
    if error_type not in MANUAL_ERROR_TYPES:
        print(f"❌ 无效的 error type: {error_type}")
        return False
    
    if not from_coi_list or not to_coi_list:
        print("❌ 必须指定 --from-coi 和 --to-coi")
        return False
    
    to_set = set(to_coi_list)
    from_set = set(from_coi_list)
    if not to_set.issubset(from_set):
        print(f"❌ to-coi 必须是 from-coi 的子集")
        print(f"   from-coi: {from_coi_list}")
        print(f"   to-coi: {to_coi_list}")
        return False
    
    img_file = find_image_file(image_name)
    if not img_file:
        print(f"❌ 找不到图片: {image_name}")
        return False
    
    base_name = img_file.stem
    error_dir = ERROR_DIR / error_type
    if not error_dir.exists():
        print(f"❌ 找不到目录: {error_dir}")
        return False
    
    # 查找匹配 from_coi 的 error 文件
    from_coi_str = coi_to_str(from_coi_list)
    pattern = re.compile(rf"{re.escape(base_name)}_coi-(.+)_iou(\d+)\.json")
    
    candidates = []
    for f in error_dir.glob(f"{base_name}_*.json"):
        if f.name.endswith("_gt.json"):
            continue
        match = pattern.match(f.name)
        if match:
            # 文件名里 COI 用 - 连接，空格在文件名里是下划线；匹配时归一化（空格与下划线等价）
            def _norm(s):
                return s.replace(' ', '_')
            file_coi = match.group(1).split('-')
            if set(_norm(c) for c in file_coi) == set(_norm(c) for c in from_coi_list):
                iou = int(match.group(2)) / 100.0
                candidates.append((iou, f))
    
    if not candidates:
        print(f"❌ 未找到 {error_type} 下 COI={from_coi_list} 的 error 文件")
        return False
    
    # 选择源文件
    candidates.sort(key=lambda x: x[0], reverse=True)
    if from_iou is not None:
        best = None
        for iou, path in candidates:
            if abs(iou - from_iou) < 0.01:
                best = (iou, path)
                break
        if best is None:
            print(f"⚠️  未找到 IoU≈{from_iou} 的版本，可选: {[c[0] for c in candidates]}")
            source_iou, source_path = candidates[0]
        else:
            source_iou, source_path = best
    else:
        source_iou, source_path = candidates[0]
        if len(candidates) > 1:
            print(f"📋 找到 {len(candidates)} 个版本，使用 IoU={source_iou:.2f}: {source_path.name}")
    
    # 读取并过滤
    with open(source_path, 'r') as f:
        data = json.load(f)
    
    original_shapes = data.get('shapes', [])
    filtered_shapes = filter_shapes_by_coi(original_shapes, to_coi_list)
    
    if not filtered_shapes:
        print(f"❌ 过滤后无 shapes，请检查 to-coi: {to_coi_list}")
        return False
    
    # 写入 images/
    data['shapes'] = filtered_shapes
    data['imagePath'] = img_file.name
    
    target_json = IMAGES_DIR / f"{base_name}.json"
    with open(target_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 更新 xxx_gt.json
    gt_backup = GT_BACKUP_DIR / f"{base_name}.json"
    with open(gt_backup, 'r') as f:
        gt_data = json.load(f)
    gt_data['shapes'] = filter_shapes_by_coi(gt_data.get('shapes', []), to_coi_list)
    gt_data['imagePath'] = img_file.name
    
    gt_target = IMAGES_DIR / f"{base_name}_gt.json"
    with open(gt_target, 'w') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已从 {source_path.name} 衍生到 COI={to_coi_list}")
    print(f"  保留 {len(filtered_shapes)} 个 shapes (原 {len(original_shapes)} 个)")
    print(f"  工作文件: {target_json}")
    print(f"\n可用 labelme 打开继续编辑，完成后保存:")
    has_space = any(' ' in c for c in to_coi_list)
    sep = '|' if has_space else ','
    coi_arg = sep.join(to_coi_list)
    print(f'  labelme "{img_file}"')
    print(f'  python manage_errors.py save "{base_name}" {error_type} <iou> --coi "{coi_arg}"')
    return True


def save(image_name, error_type, iou, coi_list):
    """保存当前编辑到error目录"""
    if error_type not in MANUAL_ERROR_TYPES:
        print(f"❌ 无效的error type: {error_type}")
        return False
    
    if not coi_list:
        print("❌ 必须指定COI: --coi class1,class2,...")
        return False
    
    # 验证IoU值
    try:
        iou_val = float(iou)
        if not 0 <= iou_val <= 1:
            print(f"❌ IoU值必须在0-1之间: {iou}")
            return False
    except ValueError:
        print(f"❌ 无效的IoU值: {iou}")
        return False
    
    img_file = find_image_file(image_name)
    if not img_file:
        print(f"❌ 找不到图片: {image_name}")
        return False
    
    base_name = img_file.stem
    source_json = IMAGES_DIR / f"{base_name}.json"
    target_json = get_error_path(base_name, error_type, coi_list, iou_val)
    
    if not source_json.exists():
        print(f"❌ 找不到JSON文件: {source_json}")
        return False
    
    # 确保目标目录存在
    target_json.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取JSON并添加元信息
    with open(source_json, 'r') as f:
        json_data = json.load(f)
    
    # 添加error metadata
    json_data['_error_metadata'] = {
        'error_type': error_type,
        'coi': coi_list,
        'iou': iou_val,
        'base_image': base_name,
        'created_at': datetime.now().isoformat(),
    }
    json_data['imagePath'] = img_file.name
    
    # 保存JSON
    with open(target_json, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 复制图片到同一目录
    target_img = target_json.parent / img_file.name
    if not target_img.exists():
        shutil.copy2(img_file, target_img)
        print(f"✓ 已复制图片: {img_file.name}")
    
    # 为每个COI创建对应的GT文件（用于labelme准确显示IoU）
    # GT文件名格式: {base_name}_coi-{coi_str}_gt.json
    coi_str = coi_to_str(coi_list)
    target_gt = target_json.parent / f"{base_name}_coi-{coi_str}_gt.json"
    
    # 从原始GT备份创建COI-specific的GT文件
    gt_backup = GT_BACKUP_DIR / f"{base_name}.json"
    if gt_backup.exists():
        with open(gt_backup, 'r') as f:
            gt_data = json.load(f)
        # 过滤只保留COI相关的shapes
        gt_data['shapes'] = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
        gt_data['imagePath'] = img_file.name
        with open(target_gt, 'w') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 已创建COI-specific GT: {target_gt.name}")
    
    # 更新error JSON的imagePath，使其能找到对应的GT
    # labelme会自动查找 {filename}_gt.json，所以需要让filename匹配
    # 但由于我们的error文件名包含iou，无法直接匹配，所以保持原imagePath
    
    print(f"✓ 已保存 {error_type} 版本")
    print(f"  COI: {coi_list}")
    print(f"  IoU: {iou_val:.2f}")
    print(f"  文件: {target_json.name}")
    
    # 显示该error type的所有版本
    all_versions = list_existing_versions(base_name, error_type)
    if len(all_versions) > 1:
        print(f"\n📋 {error_type} 的所有版本:")
        for coi, v_iou, v_path in all_versions:
            marker = "→" if v_path == target_json else " "
            print(f"  {marker} COI={coi}, IoU={v_iou:.2f}: {v_path.name}")
    
    print(f"\n可以直接用labelme查看:")
    print(f'  labelme "{target_json.parent}"')
    return True


def restore(image_name):
    """恢复为原始GT（同时恢复 xxx.json 和 xxx_gt.json）"""
    img_file = find_image_file(image_name)
    if not img_file:
        print(f"❌ 找不到图片: {image_name}")
        return False
    
    base_name = img_file.stem
    gt_json = GT_BACKUP_DIR / f"{base_name}.json"
    target_json = IMAGES_DIR / f"{base_name}.json"
    target_gt = IMAGES_DIR / f"{base_name}_gt.json"
    
    if not gt_json.exists():
        print(f"❌ 找不到GT备份: {gt_json}")
        return False
    
    # 读取备份并确保 imagePath 正确
    with open(gt_json, 'r') as f:
        data = json.load(f)
    img_name = img_file.name
    data['imagePath'] = img_name
    
    # 恢复 xxx.json
    with open(target_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 恢复 xxx_gt.json（labelme 用于 IoU 的参考）
    with open(target_gt, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已恢复为原始GT（xxx.json 和 xxx_gt.json）")
    print(f"  来源: {gt_json}")
    return True


def restore_all():
    """将所有 images 下的 xxx.json 和 xxx_gt.json 从 gt_backup 恢复"""
    print("🔧 恢复所有图片的 JSON 文件...")
    print("=" * 60)
    
    fixed = 0
    skipped = 0
    errors = 0
    
    for gt_file in sorted(GT_BACKUP_DIR.glob("*.json")):
        base_name = gt_file.stem
        target_json = IMAGES_DIR / f"{base_name}.json"
        target_gt = IMAGES_DIR / f"{base_name}_gt.json"
        
        # 找到对应的图片文件以获取正确的 imagePath
        img_file = find_image_file(base_name)
        if not img_file:
            skipped += 1
            continue
        
        with open(gt_file, 'r') as f:
            data = json.load(f)
        data['imagePath'] = img_file.name
        
        # 写入 xxx.json 和 xxx_gt.json
        with open(target_json, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(target_gt, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        fixed += 1
    
    print(f"✓ 已恢复 {fixed} 张图片的 xxx.json 和 xxx_gt.json")
    if skipped:
        print(f"  跳过 {skipped} 张（无对应图片）")
    return fixed


def check_images():
    """检查 images 文件夹中所有图片的 JSON 是否与 gt_backup 一致"""
    print("=" * 70)
    print("Images 文件夹 JSON 检查")
    print("=" * 70)
    print("\n文件说明:")
    print("  - xxx.json:      当前工作文件，在 labelme 中编辑")
    print("  - xxx_gt.json:   labelme 用于 IoU 计算的参考 GT")
    print("  - gt_backup:     原始完整 GT（数据集的原始标注）")
    print()
    
    ok_count = 0
    issues = []
    
    for gt_file in sorted(GT_BACKUP_DIR.glob("*.json")):
        base_name = gt_file.stem
        json_path = IMAGES_DIR / f"{base_name}.json"
        gt_path = IMAGES_DIR / f"{base_name}_gt.json"
        
        with open(gt_file, 'r') as f:
            backup_data = json.load(f)
        n_backup = len(backup_data.get('shapes', []))
        
        n_json = None
        n_gt = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                n_json = len(json.load(f).get('shapes', []))
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                n_gt = len(json.load(f).get('shapes', []))
        
        ok = (n_json == n_backup and n_gt == n_backup)
        if ok:
            ok_count += 1
            status = "✓"
        else:
            status = "⚠"
            issues.append((base_name, n_backup, n_json, n_gt))
        
        json_s = f"{n_json}/{n_backup}" if n_json is not None else "N/A"
        gt_s = f"{n_gt}/{n_backup}" if n_gt is not None else "N/A"
        print(f"{status} {base_name}: gt_backup={n_backup} | xxx.json={json_s} | xxx_gt.json={gt_s}")
    
    print("\n" + "=" * 70)
    print(f"总计: {ok_count}/{ok_count + len(issues)} 张图片正确")
    if issues:
        print(f"\n有问题的图片 ({len(issues)} 张):")
        for base, n_b, n_j, n_g in issues:
            j_msg = f"缺少{n_b-n_j}" if n_j is not None and n_j < n_b else "正确"
            g_msg = f"缺少{n_b-n_g}" if n_g is not None and n_g < n_b else "正确"
            if n_j is None:
                j_msg = "不存在"
            if n_g is None:
                g_msg = "不存在"
            print(f"  - {base}: xxx.json {j_msg}, xxx_gt.json {g_msg}")
        print("\n运行以下命令修复: python manage_errors.py restore_all")
    return ok_count, len(issues)


def status(image_name=None):
    """查看状态"""
    if image_name:
        img_file = find_image_file(image_name)
        if not img_file:
            print(f"❌ 找不到图片: {image_name}")
            return
        
        base_name = img_file.stem
        print(f"📷 图片: {img_file.name}")
        
        # 显示classes
        class_counts = get_image_classes(image_name)
        if class_counts:
            print(f"   Classes: {list(class_counts.keys())}")
        
        print(f"\n已创建的error versions:")
        
        found = False
        for error_type in MANUAL_ERROR_TYPES:
            versions = list_existing_versions(base_name, error_type)
            if versions:
                found = True
                print(f"\n  {error_type}:")
                for coi, iou, path in versions:
                    print(f"    ✓ COI={coi}, IoU={iou:.2f}")
                    print(f"      {path.name}")
        
        if not found:
            print("  (无)")
    else:
        print("=" * 70)
        print("Error Type 统计 (手动操作)")
        print("=" * 70)
        for error_type in MANUAL_ERROR_TYPES:
            error_dir = ERROR_DIR / error_type
            if error_dir.exists():
                files = [f for f in error_dir.glob("*_coi-*_iou*.json")]
                if files:
                    print(f"\n  {error_type}: {len(files)} 个文件")
                    # 统计不同的base images
                    base_images = set()
                    for f in files:
                        match = re.match(r'(.+)_coi-', f.name)
                        if match:
                            base_images.add(match.group(1))
                    print(f"    涉及 {len(base_images)} 张base images")
                else:
                    print(f"\n  {error_type}: 0 个文件")
            else:
                print(f"\n  {error_type}: 0 个文件")


def list_images():
    """列出所有图片"""
    images = sorted([f.stem for f in IMAGES_DIR.glob("*.json") if '_gt' not in f.stem])
    print(f"共 {len(images)} 张图片:")
    for i, name in enumerate(images, 1):
        class_counts = get_image_classes(name)
        if class_counts:
            n_classes = len(class_counts)
            n_instances = sum(class_counts.values())
            print(f"  {i:3d}. {name} ({n_classes} classes, {n_instances} instances)")
        else:
            print(f"  {i:3d}. {name}")


def fix_gt_files():
    """为所有已有的error data创建COI-specific GT文件"""
    print("🔧 修复GT文件...")
    print("=" * 60)
    
    # 新格式pattern: xxx_coi-yyy_iouZZZ.json
    pattern = re.compile(r'(.+)_coi-(.+)_iou(\d+)\.json')
    
    created = 0
    skipped = 0
    errors = 0
    
    for error_type in MANUAL_ERROR_TYPES:
        error_dir = ERROR_DIR / error_type
        if not error_dir.exists():
            continue
        
        print(f"\n📁 {error_type}:")
        
        # 收集所有唯一的 (base_name, coi_str) 组合
        coi_combos = {}
        for f in error_dir.glob("*_coi-*_iou*.json"):
            match = pattern.match(f.name)
            if match:
                base_name = match.group(1)
                coi_str = match.group(2)
                key = (base_name, coi_str)
                if key not in coi_combos:
                    coi_combos[key] = []
                coi_combos[key].append(f)
        
        for (base_name, coi_str), files in sorted(coi_combos.items()):
            gt_filename = f"{base_name}_coi-{coi_str}_gt.json"
            gt_path = error_dir / gt_filename
            
            if gt_path.exists():
                skipped += 1
                continue
            
            # 从备份创建GT文件
            gt_backup = GT_BACKUP_DIR / f"{base_name}.json"
            if not gt_backup.exists():
                print(f"  ❌ {gt_filename}: 找不到备份 {gt_backup.name}")
                errors += 1
                continue
            
            # 解析COI（下划线恢复为空格）
            coi_list = [c.replace('_', ' ') for c in coi_str.split('-')]
            
            # 读取备份并过滤
            with open(gt_backup, 'r') as f:
                gt_data = json.load(f)
            
            gt_data['shapes'] = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
            
            # 找到对应的图片文件名
            img_file = find_image_file(base_name)
            if img_file:
                gt_data['imagePath'] = img_file.name
            
            # 保存
            with open(gt_path, 'w') as f:
                json.dump(gt_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ {gt_filename} (COI: {coi_list}, {len(gt_data['shapes'])} shapes)")
            created += 1
    
    print(f"\n{'='*60}")
    print(f"完成: 创建 {created} 个GT文件, 跳过 {skipped} 个已存在, {errors} 个错误")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    # 解析 --coi, --from-coi, --to-coi, --from-iou 参数
    coi_list = []
    from_coi_list = []
    to_coi_list = []
    from_iou = None
    args = sys.argv[2:]
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == '--coi' and i + 1 < len(args):
            coi_list = parse_coi(args[i + 1])
            i += 2
        elif args[i] == '--from-coi' and i + 1 < len(args):
            from_coi_list = parse_coi(args[i + 1])
            i += 2
        elif args[i] == '--to-coi' and i + 1 < len(args):
            to_coi_list = parse_coi(args[i + 1])
            i += 2
        elif args[i] == '--from-iou' and i + 1 < len(args):
            try:
                from_iou = float(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1
    
    if cmd == "prepare":
        if len(filtered_args) >= 2:
            prepare(filtered_args[0], filtered_args[1], coi_list)
        else:
            print("用法: python manage_errors.py prepare <image_name> <error_type> --coi <classes>")
    
    elif cmd == "save":
        if len(filtered_args) >= 3:
            save(filtered_args[0], filtered_args[1], filtered_args[2], coi_list)
        else:
            print("用法: python manage_errors.py save <image_name> <error_type> <iou> --coi <classes>")
    
    elif cmd == "restore" and len(filtered_args) >= 1:
        restore(filtered_args[0])
    
    elif cmd == "status":
        status(filtered_args[0] if filtered_args else None)
    
    elif cmd == "list":
        list_images()
    
    elif cmd == "classes" and len(filtered_args) >= 1:
        show_classes(filtered_args[0])
    
    elif cmd == "fix_gt":
        fix_gt_files()
    
    elif cmd == "restore_all":
        restore_all()
    
    elif cmd == "check_images":
        check_images()
    
    elif cmd == "derive":
        if len(filtered_args) >= 2 and from_coi_list and to_coi_list:
            derive(filtered_args[0], filtered_args[1], from_coi_list, to_coi_list, from_iou)
        else:
            print("用法: python manage_errors.py derive <image_name> <error_type> --from-coi \"A|B\" --to-coi \"A\" [--from-iou 0.5]")
            print("  从已有 COI 的 error 结果衍生到子 COI，作为编辑起点")
    
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

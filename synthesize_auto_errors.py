#!/usr/bin/env python3
"""
自动合成 missed_instance 和 label_confusion 两种 error type 的数据。

- missed_instance: 从 GT 中按 COI 随机删除部分 instance，IoU = n_kept / n_gt，生成 0.25/0.5/0.75 三档。
- label_confusion: 按 COI 随机将部分 instance 的 label 改为图中已有的其他类别，混淆比例 0.25/0.5/0.75。

对 50 张 base image（见 instseg_data/BASE_IMAGES_50.txt）尽可能覆盖 4 种 question type：
  1c1i, 1c-multi, Nc1i, Nc-multi（Nc-multi 取 n>=2 个 class，至少 2 个）。

用法:
  python synthesize_auto_errors.py missed_instance   # 只生成 missed_instance
  python synthesize_auto_errors.py label_confusion   # 只生成 label_confusion
  python synthesize_auto_errors.py all              # 两种都生成
  python synthesize_auto_errors.py list_bases      # 列出 50 张 base 并检查
"""

import json
import random
import shutil
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent / "instseg_data"
GT_BACKUP_DIR = BASE_DIR / "gt_backup"
ERROR_DIR = BASE_DIR / "error_annotations"
IMAGES_DIR = BASE_DIR / "images"
BASE_LIST_FILE = BASE_DIR / "BASE_IMAGES_50.txt"

IOU_LEVELS = [0.25, 0.50, 0.75]
SEED = 42


def get_base_images():
    """从 BASE_IMAGES_50.txt 或已有 error 数据推断 50 张 base image"""
    if BASE_LIST_FILE.exists():
        bases = []
        with open(BASE_LIST_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bases.append(line)
        return bases
    # fallback: 从已有 3 种 error 目录收集
    import re
    seen = set()
    for et in ["boundary_inaccuracy", "under_segmentation", "over_segmentation"]:
        d = ERROR_DIR / et
        if d.exists():
            for f in d.glob("*_coi-*_iou*.json"):
                m = re.match(r'(.+)_coi-', f.name)
                if m:
                    seen.add(m.group(1))
    return sorted(seen)


def load_gt(base_name):
    """加载 gt_backup 中的 JSON"""
    path = GT_BACKUP_DIR / f"{base_name}.json"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def filter_shapes_by_coi(shapes, coi_list):
    if not coi_list:
        return shapes
    coi_set = set(coi_list)
    return [s for s in shapes if s.get('label', '') in coi_set]


def coi_to_str(coi_list):
    safe = [c.replace(' ', '_') for c in sorted(coi_list)]
    return '-'.join(safe)


def get_error_path(base_name, error_type, coi_list, iou):
    coi_str = coi_to_str(coi_list)
    iou_str = f"{float(iou):.2f}".replace('.', '')
    return ERROR_DIR / error_type / f"{base_name}_coi-{coi_str}_iou{iou_str}.json"


def find_image_file(base_name):
    for ext in ['.jpg', '.png', '.jpeg']:
        for f in IMAGES_DIR.glob(f"*{ext}"):
            if '_gt' in f.stem:
                continue
            if f.stem == base_name or base_name in f.stem or f.stem in base_name:
                return f
    return None


def get_coi_combinations_for_image(gt_data):
    """
    根据 GT 的 class 分布，返回 4 种 question type 的 COI 组合。
    返回 list of (qt_name, coi_list)，coi_list 可能为 None 表示该 QT 不适用。
    """
    shapes = gt_data.get('shapes', [])
    class_counts = {}
    for s in shapes:
        label = s.get('label', 'unknown')
        class_counts[label] = class_counts.get(label, 0) + 1

    single_inst = [c for c, n in class_counts.items() if n == 1]
    multi_inst = [(c, n) for c, n in class_counts.items() if n > 1]
    all_classes = list(class_counts.keys())

    out = []

    # 1c1i
    if single_inst:
        out.append(('1c1i', [single_inst[0]]))
    else:
        out.append(('1c1i', None))

    # 1c-multi
    if multi_inst:
        out.append(('1c-multi', [multi_inst[0][0]]))
    else:
        out.append(('1c-multi', None))

    # Nc1i: >=2 single-instance classes
    if len(single_inst) >= 2:
        out.append(('Nc1i', single_inst[:2]))
    else:
        out.append(('Nc1i', None))

    # Nc-multi: n>=2 classes，至少 2 个 class
    if len(all_classes) >= 2 and multi_inst:
        # 优先：1 multi + 1 single；否则 2 multi
        if single_inst:
            out.append(('Nc-multi', [multi_inst[0][0], single_inst[0]]))
        else:
            out.append(('Nc-multi', [multi_inst[0][0], multi_inst[1][0]] if len(multi_inst) >= 2 else None))
    elif len(all_classes) >= 2:
        if len(single_inst) >= 2:
            out.append(('Nc-multi', single_inst[:2]))
        else:
            out.append(('Nc-multi', None))
    else:
        out.append(('Nc-multi', None))

    return out


def synthesize_missed_instance(base_name, gt_data, coi_list, iou, img_name, seed_offset=0):
    """
    生成 missed_instance：从 COI 过滤后的 shapes 中随机保留 n_keep = round(n * iou) 个。
    """
    coi_shapes = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
    n = len(coi_shapes)
    if n == 0:
        return False
    n_keep = max(1, round(n * iou))  # 至少留 1 个
    rng = random.Random(SEED + hash(base_name) % 100000 + seed_offset + int(iou * 100))
    indices = list(range(n))
    rng.shuffle(indices)
    keep_indices = set(indices[:n_keep])
    kept_shapes = [coi_shapes[i] for i in sorted(keep_indices)]

    out_data = {k: v for k, v in gt_data.items() if k != 'shapes'}
    out_data['shapes'] = kept_shapes
    out_data['imagePath'] = img_name
    out_data['_error_metadata'] = {
        'error_type': 'missed_instance',
        'coi': coi_list,
        'iou': iou,
        'base_image': base_name,
        'created_at': datetime.now().isoformat(),
        'n_original': n,
        'n_kept': n_keep,
    }

    path = get_error_path(base_name, 'missed_instance', coi_list, iou)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    return True


def synthesize_label_confusion(base_name, gt_data, coi_list, iou, img_name, seed_offset=0):
    """
    生成 label_confusion：从 COI 过滤后的 shapes 中随机选 (1-iou)*n 个，将其 label 改为图中已有的其他类别（随机）。
    """
    coi_shapes = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
    n = len(coi_shapes)
    if n == 0:
        return False
    all_labels = list(set(s.get('label', '') for s in gt_data.get('shapes', [])))
    if len(all_labels) < 2:
        return False  # 至少要有 2 个类别才能混淆

    n_confuse = min(n - 1, max(0, round(n * (1 - iou))))  # 至少保留 1 个正确
    if n_confuse == 0:
        # 不混淆，直接复制
        out_shapes = [dict(s) for s in coi_shapes]
    else:
        rng = random.Random(SEED + hash(base_name) % 100000 + seed_offset + int(iou * 100))
        indices = list(range(n))
        rng.shuffle(indices)
        confuse_indices = set(indices[:n_confuse])
        out_shapes = []
        for i, s in enumerate(coi_shapes):
            shape = dict(s)
            if i in confuse_indices:
                current = shape.get('label', '')
                others = [l for l in all_labels if l != current]
                if others:
                    shape['label'] = rng.choice(others)
            out_shapes.append(shape)

    out_data = {k: v for k, v in gt_data.items() if k != 'shapes'}
    out_data['shapes'] = out_shapes
    out_data['imagePath'] = img_name
    out_data['_error_metadata'] = {
        'error_type': 'label_confusion',
        'coi': coi_list,
        'iou': iou,
        'base_image': base_name,
        'created_at': datetime.now().isoformat(),
        'n_confused': n_confuse,
        'n_total': n,
    }

    path = get_error_path(base_name, 'label_confusion', coi_list, iou)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    return True


def write_gt_and_copy_image(base_name, error_type, coi_list, gt_data, img_file):
    """写入 COI-specific _gt.json 并复制图片到 error 目录"""
    coi_str = coi_to_str(coi_list)
    error_dir = ERROR_DIR / error_type
    gt_path = error_dir / f"{base_name}_coi-{coi_str}_gt.json"
    gt_data = dict(gt_data)
    gt_data['shapes'] = filter_shapes_by_coi(gt_data.get('shapes', []), coi_list)
    gt_data['imagePath'] = img_file.name
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    img_dst = error_dir / img_file.name
    if not img_dst.exists():
        shutil.copy2(img_file, img_dst)


def run_missed_instance(dry_run=False):
    random.seed(SEED)
    bases = get_base_images()
    print(f"missed_instance: 共 {len(bases)} 张 base image")
    total = 0
    written_gt = set()  # (base_name, tuple(coi_list)) 已写入 GT/图片
    for base_name in bases:
        gt_data = load_gt(base_name)
        if not gt_data:
            print(f"  跳过（无 GT）: {base_name}")
            continue
        img_file = find_image_file(base_name)
        if not img_file:
            print(f"  跳过（无图片）: {base_name}")
            continue
        combos = get_coi_combinations_for_image(gt_data)
        for qt_name, coi_list in combos:
            if coi_list is None:
                continue
            key = (base_name, tuple(sorted(coi_list)))
            for iou in IOU_LEVELS:
                if dry_run:
                    total += 1
                    continue
                ok = synthesize_missed_instance(base_name, gt_data, coi_list, iou, img_file.name, seed_offset=hash(qt_name) % 100000)
                if ok:
                    total += 1
                    if key not in written_gt:
                        write_gt_and_copy_image(base_name, 'missed_instance', coi_list, gt_data, img_file)
                        written_gt.add(key)
    if dry_run:
        print(f"  将生成约 {total} 个文件")
    else:
        print(f"  已生成 {total} 个 missed_instance 文件")


def run_label_confusion(dry_run=False):
    random.seed(SEED)
    bases = get_base_images()
    print(f"label_confusion: 共 {len(bases)} 张 base image")
    total = 0
    written_gt = set()
    for base_name in bases:
        gt_data = load_gt(base_name)
        if not gt_data:
            continue
        img_file = find_image_file(base_name)
        if not img_file:
            continue
        combos = get_coi_combinations_for_image(gt_data)
        for qt_name, coi_list in combos:
            if coi_list is None:
                continue
            key = (base_name, tuple(sorted(coi_list)))
            for iou in IOU_LEVELS:
                if dry_run:
                    total += 1
                    continue
                ok = synthesize_label_confusion(base_name, gt_data, coi_list, iou, img_file.name, seed_offset=hash(qt_name) % 100000)
                if ok:
                    total += 1
                    if key not in written_gt:
                        write_gt_and_copy_image(base_name, 'label_confusion', coi_list, gt_data, img_file)
                        written_gt.add(key)
    if dry_run:
        print(f"  将生成约 {total} 个文件")
    else:
        print(f"  已生成 {total} 个 label_confusion 文件")


def list_bases():
    bases = get_base_images()
    print(f"Base images 数量: {len(bases)}\n")
    for i, b in enumerate(bases, 1):
        gt = load_gt(b)
        if gt:
            combos = get_coi_combinations_for_image(gt)
            qts = [qt for qt, coi in combos if coi is not None]
            print(f"{i:2d}. {b}  -> QT: {', '.join(qts)}")
        else:
            print(f"{i:2d}. {b}  (无GT)")
    return bases


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].strip().lower()
    dry = '--dry-run' in sys.argv

    if cmd == 'list_bases':
        list_bases()
    elif cmd == 'missed_instance':
        run_missed_instance(dry_run=dry)
    elif cmd == 'label_confusion':
        run_label_confusion(dry_run=dry)
    elif cmd == 'all':
        run_missed_instance(dry_run=dry)
        run_label_confusion(dry_run=dry)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

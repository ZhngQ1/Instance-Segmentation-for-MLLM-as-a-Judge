# Instance Segmentation Error Curation

本仓库用于对 instance segmentation 任务进行 **data curation**：从 COCO、Cityscapes、LVIS 中选取 base images，人工/自动合成多种 **error type** 的标注数据，并生成 **pairwise comparison** 的 metadata，用于模型评估。

---

## 目录结构

```
inst seg/
├── README.md                    # 本说明
├── .gitignore
│
├── select_images.py            # 从原始数据集中筛选 base images（≥3 instances）
├── convert_to_labelme.py       # 将原始标注转为 Labelme JSON 格式
├── create_labelme_format_gt.py  # GT 转 Labelme 格式（若与 convert 分离）
├── setup_gt_files.py           # 将 GT 复制到 images/ 并命名为 *_gt.json
├── analyze_images.py           # 分析 class/instance 分布，输出 image_analysis.json
│
├── manage_errors.py            # 核心：prepare / save / restore / derive / fix_gt / check_images
├── synthesize_auto_errors.py   # 自动合成 missed_instance、label_confusion
│
├── generate_pairwise_metadata.py  # 生成 pairwise 比较的 metadata（同 base/error_type/COI 不同 IoU 两两比较）
├── render_pairwise_vis.py      # 根据 JSON 在原图上绘制 polygon，生成 vis/ 下的可视化图
├── pairwise_metadata.json      # 生成的 pairwise 条目（可重新生成）
│
└── instseg_data/               # 数据目录
    ├── BASE_IMAGES_50.txt      # 50 张 base image 的列表
    ├── annotations/            # 原始格式标注（每张 base 对应一个 JSON）
    ├── gt_backup/              # Labelme 格式的完整 GT（只读备份）
    ├── images/                 # 工作目录：当前编辑用 JSON + *_gt.json，以及 base 图片
    ├── labelme_annotations/    # convert_to_labelme 的中间输出
    ├── error_annotations/      # 按 error type 存放的合成数据
    │   ├── boundary_inaccuracy/
    │   ├── under_segmentation/
    │   ├── over_segmentation/
    │   ├── missed_instance/
    │   └── label_confusion/
    └── image_analysis.json     # analyze_images.py 的分析结果（可重新生成）
```

---

## 数据与脚本说明

### 脚本

| 脚本 | 作用 |
|------|------|
| `select_images.py` | 从 COCO/Cityscapes/LVIS 中按条件（如 ≥3 instances）筛选图片与标注，写入 `instseg_data/annotations` 等。 |
| `convert_to_labelme.py` | 将 `annotations/` 的原始格式转为 Labelme JSON，输出到 `labelme_annotations/`。 |
| `setup_gt_files.py` | 将 GT 复制到 `images/` 并命名为 `*_gt.json`，供 Labelme 做 IoU 对比。 |
| `analyze_images.py` | 统计每张图的 class/instance 分布，写 `image_analysis.json`，用于选图与 COI。 |
| **`manage_errors.py`** | 管理 error 标注：`prepare`（按 COI 准备编辑）、`save`（保存到 error_annotations）、`restore`/`restore_all`、`derive`（从多 COI 衍生到单 COI）、`fix_gt`、`check_images`、`classes`（查看 COI 建议）等。 |
| `synthesize_auto_errors.py` | 对 50 张 base 自动合成 **missed_instance**、**label_confusion**（按 COI，IoU 0.25/0.5/0.75）。 |
| `generate_pairwise_metadata.py` | 在同一 base + error_type + COI 下，对不同 IoU（含 GT）两两生成比较条目，A/B 随机，输出 `pairwise_metadata.json`。 |
| `render_pairwise_vis.py` | 按 JSON 在原图上画 polygon，输出到 `vis/`，供 pairwise 的 media 使用；可不跑，仅分享脚本。 |

### Error types

- **手动**：`boundary_inaccuracy`、`under_segmentation`、`over_segmentation`（在 Labelme 里编辑后 `manage_errors.py save`）。
- **自动**：`missed_instance`（随机删 instance）、`label_confusion`（随机改 label）。

### 数据目录

| 目录/文件 | 说明 |
|-----------|------|
| `instseg_data/annotations/` | 原始数据集格式的标注（每张 base 一个 JSON）。 |
| `instseg_data/gt_backup/` | Labelme 格式的完整 GT，不在此目录内修改。 |
| `instseg_data/images/` | 当前工作用图与 JSON；`*_gt.json` 为 Labelme 的 IoU 参考。 |
| `instseg_data/error_annotations/{error_type}/` | 各 error type 的 JSON、对应 `*_gt.json` 及复制的图片。 |
| `instseg_data/BASE_IMAGES_50.txt` | 用于自动合成与 pairwise 的 50 张 base 列表。 |
| `pairwise_metadata.json` | Pairwise 比较的 metadata；可由 `generate_pairwise_metadata.py` 重新生成。 |

---

## 使用流程简述

1. **准备数据**：若有原始数据，用 `select_images.py`、`convert_to_labelme.py`、`setup_gt_files.py` 得到 `images/` 与 `gt_backup/`。
2. **手动做 error**：`manage_errors.py classes <image>` 看 COI → `prepare` → 在 Labelme 中编辑 → `save`。
3. **自动做 error**：`python synthesize_auto_errors.py all` 生成 missed_instance、label_confusion。
4. **Pairwise**：`python generate_pairwise_metadata.py` 生成 metadata；需要图时再运行 `render_pairwise_vis.py`。

---

## 依赖与约定

- **Labelme**：用于编辑 polygon 与查看 IoU（需能加载 `*_gt.json`）。
- **Python 3**：脚本依赖 `pathlib`、`json` 等标准库；`render_pairwise_vis.py` 需 `numpy`、`pillow`。
- **路径**：`manage_errors.py` 与 `synthesize_auto_errors.py` 内 `BASE_DIR` 为 `instseg_data`，若仓库名或路径改动需相应修改。

---

## 关于 GitHub

- 已删除 **raw_data/**（原始大体积数据）及 **.DS_Store**。
- `pairwise_metadata.json` 较大时可加入 `.gitignore`，由 `generate_pairwise_metadata.py` 在本地或 CI 中重新生成。
- 可视化图目录 **vis/** 已在 `.gitignore` 中注释列出，需要时可取消注释以忽略生成图片。

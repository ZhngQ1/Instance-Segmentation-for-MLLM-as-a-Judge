# Instance segmentation encoding summary

**Script:** `instseg_visualize_pairwise.py`  
**Inputs:** `images.json`, `annotations.json`, base images under `final_images/instance_seg/`  
**Outputs:** `instance_seg_pairwise_encoded.json`, visualization images under `media/instance_seg/<enc_dir>/`

## Encodings (7)

| Enc | Dir | Description |
|-----|-----|-------------|
| 1 | `enc1_50pct_per_instance` | Each instance different 50% color, white stroke, label (white + thin black stroke, 100% opacity). |
| 2 | `enc2_100pct_per_instance` | Same as 1 but 100% fill opacity. |
| 3 | `enc3_same_color_per_class` | Same color per class, white stroke, label. |
| 4 | `enc4_same_color_per_class_no_label` | Same as 3, no label; prompt includes legend (which color = which class). |
| 5 | `enc5_stroke_only_per_class` | Stroke only, same color per class, with label. |
| 6 | `enc6_stroke_only_per_class_no_label` | Same as 5, no label; prompt includes legend. |
| 7 | `enc7_json_text` | No image; prediction as JSON-like text in the prompt. |

## Pairwise

- Group annotations by **(image_id, coi, encoding)**.
- For each group with ≥2 annotations, exhaust all pairs (A, B).
- Each pair is one ranking item: original image + two visualizations (or two text blocks for enc 7), prompt “Which prediction is better? A or B.”, metadata (ann_id_a/b, model_name, error_type, final_score, etc.).

## Run

```bash
conda activate cv
python instseg_visualize_pairwise.py
```

Requires: `opencv-python`, `numpy`, `tqdm`.

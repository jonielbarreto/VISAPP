# VISAPP — YOLO vs. SAHI vs. Temporal-Consistency Detection Benchmark

Benchmarking pipeline comparing three object-detection strategies on the same YOLO model:
native full-frame YOLO inference, **SAHI** (Sliced Aided Hyper Inference) for small-object
detection on high-resolution imagery, and a temporal-consistency filter on top of SAHI that
confirms detections across consecutive frames before accepting them. Built around a maritime/
aerial surveillance detection dataset (MMOS).

---

## Repository structure

```text
VISAPP/
├── pipeline.py          # Entry point: wires together training/evaluation for each strategy
├── yolo_functions.py    # YOLOPipeline: train / validate / visualize native YOLO detections
├── sahi_strategy.py      # SahiStrategy: sliced inference wrapper around a YOLO model via SAHI
├── temporal_manager.py   # TemporalConsistencyManager: multi-frame detection confirmation filter
├── sahi_functions.py     # SahiDetectionPipeline: SAHI-only and SAHI+temporal-consistency benchmarking
├── yolo_coco_eval.py      # COCOEvaluator: COCO-style mAP/precision/recall metric computation
├── utils.py                # Dataset prep helpers (COCO slicing for SAHI, YOLO<->COCO conversion)
└── requirements.txt
```

### Detection strategies

* **YOLO (native)** — `yolo_functions.YOLOPipeline`: standard full-frame training/inference/validation with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
* **SAHI only** — `sahi_strategy.SahiStrategy` + `sahi_functions.SahiDetectionPipeline` (`mode="sahi_only"`): slices each frame into overlapping tiles for detection, then merges results — improves recall on small/distant objects in high-resolution footage.
* **SAHI + temporal consistency** — same as above with `mode="temporal_consistency"`: `temporal_manager.TemporalConsistencyManager` requires a detection to persist across `min_confirmation_frames` consecutive frames (within `edge_tolerance_px`) before it is confirmed, suppressing one-off false positives.

All three strategies are evaluated with the same COCO-style metrics (`yolo_coco_eval.COCOEvaluator`): mAP, and precision/recall/F1 at a given confidence/IoU operating point.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

`pipeline.py` shows how the pieces fit together. Point it at your own dataset and trained model:

```python
from yolo_functions import YOLOPipeline
from sahi_functions import SahiDetectionPipeline
from yolo_coco_eval import COCOEvaluator

# Native YOLO: train (optional) + validate + COCO metrics
yolo_pipeline(data_yaml, model_path, images_dir, labels_dir, to_train=False, img_size=640)

# SAHI-only or SAHI + temporal-consistency benchmarking
sahi_pipeline(data_yaml, model_path, images_dir, labels_dir, to_train=False)
```

Each dataset split is expected in standard YOLO format (`images/`, `labels/`, `data.yaml`).
`utils.py` provides helpers to slice a COCO-format dataset for SAHI training
(`sahi.slicing.slice_coco`) and to convert between COCO and YOLO label formats.

> **Note:** the dataset, trained weights, and experiment logs/outputs used for the paper's
> results are not included in this repository.

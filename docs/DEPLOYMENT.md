# Deployment Guide — DC2PSA-YOLO26-Seg

Step-by-step instructions for reproducing the experiments, running inference, and deploying the model.

---

## 1. Requirements

### Hardware

| Component | Training | Inference Only |
|---|---|---|
| GPU | NVIDIA GPU ≥ 16 GB VRAM (Tesla T4, RTX 3090, A100) | Optional (CPU supported) |
| RAM | ≥ 16 GB | ≥ 8 GB |
| Storage | ≥ 10 GB (dataset + checkpoints) | ≥ 1 GB |

> **Recommended**: Use [Google Colab](https://colab.research.google.com/) with a T4 GPU (free tier) for training.

### Software

- Python 3.10–3.13
- pip ≥ 23.0
- Git
- CUDA 12.x (for GPU training)

---

## 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/Saifal-abbas/DC2PSA-YOLO26-Real-Time-Airport-Runway-Crack-Detection-and-Instance-Segmentation.git
cd DC2PSA-YOLO26-Real-Time-Airport-Runway-Crack-Detection-and-Instance-Segmentation

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

### Download CrackAirport

1. Download from [Mendeley Data](https://data.mendeley.com/) (CrackAirport dataset)
2. Extract and organize:

```
dataset_seg/
├── data.yaml
├── train/
│   ├── images/     # 1,599 images (512×512 px)
│   └── labels/     # YOLO polygon segmentation labels
├── val/
│   ├── images/     # 452 images
│   └── labels/
└── test/
    ├── images/     # 227 images
    └── labels/
```

### data.yaml Format

```yaml
path: /absolute/path/to/dataset_seg
train: train/images
val: val/images
test: test/images
nc: 1
names: ['Crack']
```

### Label Format

Each `.txt` label file uses YOLO polygon segmentation format:
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```
All coordinates are normalized to [0, 1]. Class ID is `0` (Crack).

---

## 4. Training

### Option A: Google Colab (Recommended)

1. Upload `notebooks/DC2PSA_YOLO26_Seg_Training.ipynb` to Google Colab
2. Mount Google Drive and upload the dataset
3. Select **T4 GPU** runtime
4. Run all cells sequentially

The notebook trains all 16 experiments (9 detection + 7 segmentation) with:
- Identical hyperparameters (see `config.json`)
- Seed 42 for full deterministic reproducibility
- Automatic metric collection and figure generation

### Option B: Local Training

```bash
# Ensure GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Launch notebook
jupyter notebook notebooks/DC2PSA_YOLO26_Seg_Training.ipynb
```

### Training Configuration

All experiments use the configuration in `config.json`:

| Parameter | Value |
|---|---|
| Image size | 640 × 640 |
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | SGD (momentum=0.937) |
| Learning rate | Cosine annealing (1e-3 → 0.01) |
| Seed | 42 |

---

## 5. Inference

### Single Image

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.35,
    save=True
)
```

### Batch Inference

```python
results = model.predict(
    source="path/to/image_folder/",
    imgsz=640,
    conf=0.35,
    save=True,
    save_txt=True,
    save_conf=True
)
```

### Accessing Masks for PCI Quantification

```python
import numpy as np

for result in results:
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data):
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            area_px = binary_mask.sum()
            area_mm2 = area_px * 2.25  # GSD² = 1.5² mm²
            print(f"Crack {i}: {area_px} px² = {area_mm2:.1f} mm²")
```

---

## 6. Model Export

### ONNX Export

```python
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

### TensorRT Export (for NVIDIA edge devices)

```python
model.export(format="engine", imgsz=640, half=True)
```

### NCNN Export (for Raspberry Pi / ARM)

```python
model.export(format="ncnn", imgsz=640)
```

---

## 7. Evaluation

### Validation

```python
model = YOLO("best.pt")
metrics = model.val(
    data="path/to/data.yaml",
    imgsz=640,
    split="test",
    save_json=True
)

print(f"Box mAP@0.5: {metrics.box.map50:.4f}")
print(f"Mask mAP@0.5: {metrics.seg.map50:.4f}")
```

---

## 8. Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: ultralytics` | Re-run `pip install -r requirements.txt` |
| CUDA not detected | Set `device="cpu"` — runs slower but works |
| Out of GPU memory | Reduce `batch` to 8 or 4 in config |
| Labels not found | Check `data.yaml` paths match your directory structure |
| Permission denied on weights | Ensure write access to the output directory |

---

## 9. Reproducibility Checklist

- [x] Fixed random seed (42) across all frameworks
- [x] Deterministic mode enabled (`deterministic=True`)
- [x] Identical hyperparameters for all 16 experiments
- [x] Same dataset splits (70/20/10 stratified)
- [x] Same hardware (Tesla T4 16 GB via Google Colab)
- [x] All metrics logged to CSV for verification
- [x] 600 DPI figures generated programmatically

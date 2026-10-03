# DC2PSA-YOLO26-Seg — Complete Results & Analysis

> **Project**: DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for Real-Time Airport Runway Crack Detection and Instance Segmentation
>
> **Baseline Paper**: Abbas, S.; Shawon, M.T.I.; Qamar, S.; Adeel, M. *Evaluating YOLO26s for Multi-Class Pavement Crack Detection.* **Sensors 2026**, 26(16), 5113.
>
> **Experiment Date**: September 6–19, 2026 | **Device**: Tesla T4 (15.6 GB) via Google Colab | **Seed**: 42
>
> **Status**: **16/16 Experiments Fully Completed (9 Detection + 7 Segmentation)**

---

## 1. Research Trajectory: From Road to Runway

| Dimension | Baseline Paper (*Sensors* 2026) | This Study (DC2PSA-YOLO26-Seg) | Performance Impact |
|---|---|---|---|
| **Domain** | Municipal road highways (RDD2022) | Airport runways/taxiways (CrackAirport) | First airport runway benchmark |
| **Task** | 2-D bounding-box detection only | Dual-task: detection **+** instance segmentation | Enables FAA PCI crack width/area quantification |
| **Attention** | Standard C2PSA (rectilinear 3×3) | **DC2PSA** — Dynamic Snake Convolution (DSConv) | Adapts receptive field along curvilinear crack paths |
| **Metrics** | Box mAP@0.5 | Box mAP + Mask mAP + FAA PCI geometry | Sub-pixel contour boundary evaluation |
| **Classes** | 4 (longitudinal, transverse, pothole, alligator) | 1 (structural crack — binary) | False-alarm suppression via 1,053 negative images |
| **Dataset** | 6,972 images (4 countries) | 2,262 images (Tennessee, USA) | 100 ft AGL drone nadir imagery (GSD = 1.5 mm/px) |
| **Best Detection mAP@0.5** | 89.0% (YOLO26s, road) | **42.7%** (DC2PSA-YOLO26s, airport) | **46.3 pp domain gap quantified** |
| **Best Dual-Task Box mAP@0.5** | — | **45.1%** (DC2PSA-YOLO26s-seg) | **+2.2 pp over YOLO26s** |
| **Best Mask mAP@0.5** | — | **29.8%** (DC2PSA-YOLO26s-seg) | **+3.0 pp over YOLO26s** |

---

## 2. Detection Benchmark — All 9 Models

| Rank | Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 Score | Epochs |
|---|---|---|---|---|---|---|---|
| 🥇 **1** | **DC2PSA-YOLO26s ★** | **42.71%** | **20.40%** | **61.48%** | **42.01%** | **49.92%** | 100 |
| 🥈 2 | YOLO12n | 41.09% | 19.78% | 57.78% | 40.04% | 47.30% | 100 |
| 🥉 3 | YOLO12s | 40.32% | 19.82% | 59.29% | 38.86% | 46.95% | 100 |
| 4 | YOLO26s | 40.06% | 19.35% | 59.22% | 39.45% | 47.35% | 100 |
| 5 | YOLO11s | 39.71% | 19.84% | 51.98% | 39.64% | 44.98% | 100 |
| 6 | YOLO11n | 37.37% | 17.57% | 51.60% | 38.26% | 43.94% | 100 |
| 7 | YOLOv8s | 37.11% | 16.90% | 57.03% | 35.90% | 44.06% | 100 |
| 8 | YOLOv8n | 36.48% | 15.33% | 56.90% | 35.31% | 43.57% | 100 |
| 9 | YOLO26n | 31.01% | 14.48% | 47.22% | 31.56% | 37.83% | 100 |

**Key findings:**
- DC2PSA-YOLO26s surpasses YOLO12n by +1.62 pp and YOLO26s by +2.65 pp in mAP@0.5
- Only model exceeding 61% Precision and 42% Recall, achieving F1 = 0.4992

---

## 3. Segmentation Benchmark — All 7 Models

| Rank | Model | Mask mAP@0.5 | Mask mAP@0.5:0.95 | Mask Precision | Mask Recall | Mask F1 | Box mAP@0.5 |
|---|---|---|---|---|---|---|---|
| 🥇 **1** | **DC2PSA-YOLO26s ★** | **29.83%** | **6.90%** | **50.97%** | **32.94%** | **40.02%** | **45.08%** |
| 🥈 2 | YOLO26s | 26.80% | 5.65% | 48.00% | 29.60% | 36.62% | 42.90% |
| 🥉 3 | YOLO11s | 26.12% | 5.88% | 47.80% | 28.40% | 35.63% | 42.85% |
| 4 | YOLOv8s | 25.61% | 5.40% | 53.75% | 27.61% | 36.48% | 42.78% |
| 5 | YOLOv8n | 25.57% | 5.46% | 51.48% | 27.42% | 35.78% | 41.40% |
| 6 | YOLO11n | 23.12% | 5.24% | 43.88% | 27.22% | 33.60% | 41.53% |
| 7 | YOLO26n | 21.27% | 4.19% | 38.08% | 26.44% | 31.21% | 33.25% |

**DC2PSA validation highlights:**
1. **Mask mAP@0.5**: +3.03 pp over YOLO26s (+11.3% relative gain)
2. **Strict Mask mAP@0.5:0.95**: +1.25 pp (+22.1% relative) — crisper crack edge alignment
3. **Mask Recall**: +3.34 pp — recovers more fractured crack branches
4. **Dual-Task Box mAP@0.5**: 45.08% — new high watermark for airport pavement distress

---

## 4. Dual-Task Synergy Analysis

| Model | Det-Only mAP@0.5 | Seg Box mAP@0.5 | Mask mAP@0.5 | Box Gain | Relative Synergy |
|---|---|---|---|---|---|
| YOLOv8n | 36.48% | 41.40% | 25.57% | +4.92 pp | +13.49% |
| YOLOv8s | 37.11% | 42.78% | 25.61% | +5.67 pp | +15.28% |
| YOLO11n | 37.37% | 41.53% | 23.12% | +4.16 pp | +11.13% |
| YOLO11s | 39.71% | 42.85% | 26.12% | +3.13 pp | +7.89% |
| YOLO26n | 31.01% | 33.25% | 21.27% | +2.23 pp | +7.21% |
| YOLO26s | 40.06% | 42.90% | 26.80% | +2.84 pp | +7.09% |
| **DC2PSA-YOLO26s ★** | **42.71%** | **45.08%** | **29.83%** | **+2.37 pp** | **+5.55%** |

> Multi-task supervision universally boosts bounding-box localization (+5.6% to +15.3%).

---

## 5. Training Convergence & Loss Analysis

| Model | Task | Best Val Loss | Best Epoch | Final seg_loss |
|---|---|---|---|---|
| **DC2PSA-YOLO26s** | **det** | **1.2163** | **61** | — |
| **DC2PSA-YOLO26s** | **seg** | **2.4980** | **88** | **1.4160 ★** |
| YOLO26s | det | 1.1922 | 86 | — |
| YOLO26s | seg | 2.4805 | 66 | 2.1930 |
| YOLO12n | det | 1.1611 | 98 | — |
| YOLO11s | seg | 2.5036 | 80 | 1.4317 |

> DC2PSA-YOLO26s achieves the **lowest final seg_loss (1.4160)** among all evaluated models.

---

## 6. Model Efficiency Comparison

| Model | Det Checkpoint | Seg Checkpoint | Params (M) | FLOPs (G) |
|---|---|---|---|---|
| YOLOv8n | 6.3 MB | 6.8 MB | 3.2 | 8.7 |
| YOLOv8s | 22.5 MB | 23.9 MB | 11.2 | 28.6 |
| YOLO11n | 5.5 MB | 6.0 MB | 2.6 | 6.4 |
| YOLO11s | 19.2 MB | 20.5 MB | 9.4 | 21.5 |
| YOLO12n | 5.5 MB | — | 2.6 | 6.9 |
| YOLO12s | 18.9 MB | — | 9.3 | 21.4 |
| YOLO26n | 5.4 MB | 6.6 MB | 2.6 | 6.9 |
| YOLO26s | 20.3 MB | — | 9.6 | 26.4 |
| **DC2PSA-YOLO26s ★** | **20.3 MB** | **23.4 MB** | **10.28** | **20.8** |

> DC2PSA-YOLO26s reduces FLOPs by **21.2%** (26.4G → 20.8G) while improving accuracy.

---

## 7. Baseline Paper Reference (Sensors 2026)

Road pavement detection results from the published baseline for cross-domain context:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YOLOv8n | 86.4% | 48.9% | 80.6% | 81.7% | 81.2% |
| YOLOv8s | 88.1% | 49.5% | 82.4% | 82.3% | 82.4% |
| YOLO11n | 87.6% | 50.1% | 82.2% | 79.5% | 80.8% |
| YOLO11s | 88.6% | 51.6% | 82.2% | 81.8% | 82.0% |
| YOLO12n | 87.9% | 50.5% | 81.6% | 81.5% | 81.6% |
| YOLO12s | 89.2% | 52.2% | 83.9% | 81.4% | 82.9% |
| YOLO26n | 85.7% | 49.0% | 80.9% | 79.4% | 80.1% |
| **YOLO26s** | **89.0%** | **51.6%** | **84.0%** | **82.5%** | **83.2%** |

**C2PSA Ablation**: YOLO26s without C2PSA drops to 77.74% mAP@0.5 (−11.51 pp), validating spatial attention as the primary performance driver.

---

## 8. Key Novelty Claims

1. **First 9-Architecture Benchmark on Airport Runway Pavement**: Comprehensive dual-task evaluation under identical 100-epoch conditions
2. **First Dynamic Snake Deformable Attention in YOLO26**: DC2PSA replaces fixed-grid spatial attention with iterative coordinate-deforming convolutions
3. **First Quantitative Cross-Domain Disparity Study**: Road vs. airport domain barrier measurement
4. **FAA PCI / ASTM D5340 Compliance**: Automated mask-to-maintenance pipeline bridging neural predictions to regulatory thresholds

---

*Generated from 16/16 completed experiments · 19 figures at 600 DPI · 9 result tables · Seed 42 deterministic*

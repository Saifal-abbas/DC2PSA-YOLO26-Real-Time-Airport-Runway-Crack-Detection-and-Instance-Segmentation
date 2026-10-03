# Research Proposal

---

## Project Title

**DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for Real-Time Airport Runway Crack Detection and Instance Segmentation Toward Sustainable Aviation Infrastructure Inspection**

---

## Executive Summary

Ensuring the structural integrity of airport runway pavements is a non-negotiable prerequisite for flight safety, Foreign Object Debris (FOD) prevention, and long-term infrastructure asset management. Despite recent advances in deep learning–based pavement distress detection, three persistent deficiencies limit practical adoption in the aviation domain: **(i)** existing models have been developed and validated exclusively on municipal road datasets, leaving their applicability to airport runway surfaces—whose material composition, operational stresses, and crack morphologies differ fundamentally—entirely unexamined; **(ii)** state-of-the-art detectors output only coarse axis-aligned bounding boxes that cannot quantify crack geometry (length, width, area) as mandated by FAA and ASTM pavement condition assessment standards; and **(iii)** the standard spatial attention mechanisms employed in modern YOLO architectures rely on fixed-grid, axis-aligned receptive fields that are ill-suited for the thin, curvilinear, and morphologically irregular crack patterns characteristic of aviation pavement.

Building directly upon our recently published baseline study (*Sensors* 2026, 26, 5113), this proposal introduces **DC2PSA-YOLO26-Seg**—a purpose-built, lightweight, end-to-end framework that simultaneously performs **bounding-box object detection** and **pixel-accurate instance segmentation** on airport runway pavements. At the architectural core of the proposed framework lies **DC2PSA (Deformable C2PSA with Dynamic Snake Convolution)**, a novel spatial attention module whose receptive field dynamically deforms along continuous curvilinear trajectories to faithfully capture thin, meandering, and vegetation-infiltrated runway cracks. Trained and evaluated on the **CrackAirport** benchmark—comprising 2,262 high-resolution drone-captured images from operational airfield facilities—the system provides automated geometric extraction of crack length, width, and distressed area in full alignment with **FAA Advisory Circular 150/5380-6C** and **ASTM D5340 Pavement Condition Index (PCI)** standards, while sustaining real-time, edge-deployable inference efficiency suitable for UAV-mountable hardware.

---

## 1. Research Background and Continuity with Published Work

### 1.1 Foundation: Our Published Baseline

This thesis constitutes a direct and systematic advancement of our recently published investigation:

> **Abbas, S.; Shawon, M.T.I.; Qamar, S.; Adeel, M.**  
> *Evaluating YOLO26s for Multi-Class Pavement Crack Detection: A Lightweight Approach for Sustainable Edge Deployment.*  
> **Sensors 2026**, 26(16), 5113. [DOI: 10.3390/s26165113](https://doi.org/10.3390/s26165113)

In that study, we systematically evaluated YOLO26s—a lightweight, anchor-free object detector incorporating C3k2 compact backbone bottlenecks, C2PSA spatial attention, and stride-normalized bounding-box regression—on a curated 6,972-image subset of the multinational Road Damage Dataset 2022 (RDD2022), annotated with four structural damage classes (longitudinal crack, transverse crack, pothole, and alligator crack). YOLO26s achieved **89.0% mAP@0.5** while reducing parameters by 14.3% (9.6M vs. 11.2M) and floating-point operations by 7.7% (26.4 GFLOPs vs. 28.6 GFLOPs) relative to YOLOv8s. Edge-deployment viability was validated on Raspberry Pi 5 hardware.

### 1.2 Evolutionary Research Trajectory

The relationship between the baseline and the proposed work is summarized below:

| Dimension | Baseline Paper (*Sensors* 2026) | Proposed Thesis Paper |
|---|---|---|
| **Infrastructure Domain** | Municipal road highways (RDD2022) | Airport runways and taxiways (CrackAirport) |
| **Detection Task** | 2-D bounding-box detection only | Dual-task: bounding-box detection **+** instance segmentation |
| **Spatial Attention** | Standard axis-aligned C2PSA | **DC2PSA** — Dynamic Snake Convolution–enhanced deformable attention |
| **Primary Metrics** | Box mAP@0.5 | Box mAP + Mask mAP + FAA PCI geometric quantification |
| **Distress Assessment** | Qualitative bounding-box overlays | Quantitative morphological metrics (crack length, width, area) |

---

## 2. Problem Statement and Critical Gaps

### 2.1 Three Fundamental Limitations of the Baseline Study

Although the baseline paper established the computational efficiency and detection accuracy of YOLO26s on road pavement surfaces, three fundamental domain-specific and architectural limitations necessitate the present advanced investigation:

**Gap 1 — Domain Disparity (Road versus Aviation Infrastructure).**
Airport runways differ from municipal roads across every dimension relevant to visual crack detection:

- *Material Composition:* Runways comprise thick Portland Cement Concrete (PCC) slabs and polymer-modified asphalt overlays engineered to withstand heavy aircraft wheel loadings (e.g., Boeing 777-300ER main gear loads exceeding 45 tonnes per axle), producing surface textures and deterioration patterns markedly different from those of passenger-vehicle roads.
- *Operational Environment:* Runway surfaces are subjected to high-temperature jet blast scouring, dense tire rubber deposition concentrated in touchdown zones, cyclic freeze–thaw from de-icing chemical application, and hydraulic jet erosion—environmental stressors absent from road infrastructure.
- *Crack Morphologies:* Airport cracks exhibit distinctive visual characteristics including micro-fissures along deteriorated joint sealants, shallow surface spalling, and pronounced vegetation infiltration at concrete joint interfaces—patterns that confound models trained exclusively on road imagery.

**Gap 2 — Absence of Instance Segmentation for Regulatory-Compliant Geometric Quantification.**
Standard bounding-box detectors produce only rectangular coordinate outputs $(x, y, w, h)$. However, civil aviation regulatory frameworks—specifically **FAA Advisory Circular 150/5380-6C** (*Guidelines and Procedures for Maintenance of Airport Pavements*) and **ASTM D5340-20** (*Standard Test Method for Airport Pavement Condition Index Surveys*)—mandate precise geometric measurement of individual distress features:

$$\text{Severity Level} = f\bigl(\text{Crack Width } W,\; \text{Crack Length } L,\; \text{Distressed Area } A\bigr)$$

A bounding box inevitably encloses substantial undamaged pavement within its perimeter, rendering accurate estimation of crack width and surface area impossible. **Pixel-level instance segmentation** is essential to extract true topological skeletons and boundary contours from which physical dimensions can be derived.

**Gap 3 — Inadequacy of Standard C2PSA for Irregular Crack Geometries.**
The standard Cross-Stage Partial with Spatial Attention (C2PSA) module in YOLO26 relies on fixed-grid, axis-aligned $3 \times 3$ convolutions and rectilinear self-attention projections. Runway cracks, however, are non-rigid, curvilinear, snake-like structures spanning only 1–5 pixels in width across extensive spatial extents. Standard square receptive fields consequently waste 60–75% of their computational budget sampling background asphalt texture and fail to maintain representational continuity across thin, branching crack trajectories.

---

## 3. Research Objectives

This thesis will accomplish five precisely scoped research objectives:

1. **Design and implement DC2PSA-YOLO26-Seg** — an enhanced YOLO26 architecture integrating:
   - A novel **Deformable C2PSA (DC2PSA)** module equipped with Dynamic Snake Convolutions (DSConv) that adaptively deform their receptive fields to trace irregular, curvilinear crack paths.
   - A decoupled **Proto-Mask Instance Segmentation Head** operating concurrently with the anchor-free detection head to produce pixel-accurate crack masks.

2. **Prepare the CrackAirport dataset in dual-format annotation** — processing all 2,262 drone-acquired aerial images into both YOLO bounding-box labels $(c,\; x_c,\; y_c,\; w,\; h)$ and YOLO segmentation polygon labels $(c,\; x_1,\; y_1,\; \dots,\; x_n,\; y_n)$ with stratified 70/20/10 partitioning.

3. **Execute a comprehensive 9-architecture, dual-task benchmark** — training and evaluating YOLOv8n/s, YOLO11n/s, YOLO12n/s, YOLO26n/s, and the proposed DC2PSA-YOLO26s across both detection and segmentation tasks under identical hyperparameter protocols.

4. **Develop an automated FAA PCI geometric quantification pipeline** — extracting physical crack length ($L$), mean width ($W$), and distressed surface area ($A$) from predicted instance masks and classifying distress severity according to ASTM D5340 criteria.

5. **Conduct cross-domain comparative analysis and edge-deployment validation** — formulating the first controlled comparison of YOLO26 performance on road pavement (*Sensors* 2026) versus airport runway pavement (this study), alongside inference latency benchmarking on embedded edge hardware and robustness profiling under five degraded atmospheric conditions.

---

## 4. Dataset Specification and Annotation Pipeline

### 4.1 Primary Dataset: CrackAirport

**Source:** Lyu, Z.; Yu, A.; Starr, C. *CrackAirport: A Dataset for Segmentation of Cracks in Airport Pavements.* Mendeley Data, 2026, V1. [DOI: 10.17632/3v5r2fxf89.1](https://doi.org/10.17632/3v5r2fxf89.1)

| Parameter | Specification |
|---|---|
| **Total Images** | 2,262 RGB images (512 × 512 pixels, 24-bit JPEG) |
| **Acquisition Platform** | Sony ILCE-7RM4A camera mounted on UAV at 100 ft (30.5 m) above ground level |
| **Ground Sampling Distance** | Approximately 1.5 mm per pixel |
| **Operational Facilities** | Multiple active airfield runways and taxiways across Tennessee, USA |
| **Pavement Types Represented** | Rigid Portland Cement Concrete (PCC) slabs; flexible asphalt overlays |
| **Positive Images (containing cracks)** | 1,209 images (53.4%) |
| **Negative Images (background only)** | 1,053 images (46.6%) — essential for false-alarm suppression against tire marks, joint sealant, and runway markings |
| **Total Extracted Crack Instances** | 2,798 discrete crack contour regions |
| **Environmental Challenges** | Joint sealant weathering, rubber skid deposits, edge vegetation encroachment, paint stripe interference, shadow casting |

### 4.2 Stratified Data Partitioning

| Split | Total Images | Positive (Crack) | Negative (Background) | Crack Instances |
|---|---|---|---|---|
| **Training (70%)** | 1,583 | 846 (53.4%) | 737 (46.6%) | ~1,958 |
| **Validation (20%)** | 452 | 242 (53.5%) | 210 (46.5%) | ~560 |
| **Test (10%)** | 227 | 121 (53.3%) | 106 (46.7%) | ~280 |

Stratified random sampling ensures that the crack-to-background image ratio is preserved across all partitions, preventing distributional bias in evaluation.

### 4.3 Dual-Task Label Generation Protocol

The raw grayscale segmentation masks ($M \in [0,\, 255]^{512 \times 512}$) are transformed into YOLO-compatible annotations through a five-stage automated pipeline:

1. **Binary thresholding:** $\mathcal{B}(x, y) = \mathbb{I}\bigl(M(x, y) \geq 128\bigr)$
2. **Morphological closure:** A $3 \times 3$ elliptical structuring element fills single-pixel disconnections introduced by JPEG compression artifacts.
3. **Contour extraction and area gating:** Topological contours $\mathcal{C}_i$ are extracted via the Suzuki–Abe border-following algorithm; instances with pixel area $\mathcal{A}(\mathcal{C}_i) < 40$ are pruned as sensor noise.
4. **Douglas–Peucker polygon simplification:** Contour vertices are approximated with tolerance $\epsilon = 1.0$, yielding normalized polygon coordinates $\{(\hat{x}_k, \hat{y}_k)\}_{k=1}^{K} \in [0,\, 1]^2$ for the segmentation label format.
5. **Bounding-box derivation:** Minimal axis-aligned bounding rectangles $[\hat{x}_c,\, \hat{y}_c,\, \hat{w},\, \hat{h}]$ are computed from polygon vertex extrema for the detection label format.

This dual-output pipeline produces matched label pairs, enabling simultaneous training and evaluation of both detection and segmentation tasks on identical data splits.

---

## 5. Proposed Architecture: DC2PSA-YOLO26-Seg

### 5.1 System-Level Architecture

```
                          DC2PSA-YOLO26-Seg — END-TO-END ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ INPUT: UAV Drone Imagery (640 × 640 × 3)                                              │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ BACKBONE: C3k2 Compact Bottleneck Blocks + Strided Downsampling Convolutions          │
 │           P1 (320×320) → P2 (160×160) → P3 (80×80) → P4 (40×40) → P5 (20×20)         │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ SPPF: Spatial Pyramid Pooling – Fast (Multi-Scale Receptive Field Aggregation)         │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ ★ DC2PSA: Deformable Parallel Spatial Attention with Dynamic Snake Convolution         │
 │   • DSConv adaptively deforms receptive field along curvilinear crack trajectories     │
 │   • Multi-head self-attention selectively amplifies crack-salient feature responses    │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ NECK: PAN-FPN Bidirectional Multi-Scale Feature Fusion Network                         │
 └──────────────────┬───────────────────────────────────────────────┬─────────────────────┘
                    ▼                                               ▼
 ┌────────────────────────────────────┐          ┌────────────────────────────────────────┐
 │ DETECTION HEAD (Anchor-Free)      │          │ SEGMENTATION HEAD (Proto-Mask)          │
 │ • Class Prediction: BCE Loss      │          │ • 32 Learned Prototype Masks             │
 │ • Box Regression: CIoU + L1       │          │ • Per-Instance Mask Coefficients         │
 │ → Outputs: [BBox, Confidence, Cls]│          │ → Outputs: Pixel-Level Crack Masks       │
 └────────────────────────────────────┘          └────────────────────────────────────────┘
```

### 5.2 DC2PSA: Dynamic Snake Deformable Spatial Attention

#### Motivation
Standard C2PSA processes feature maps through multi-head self-attention paired with fixed-grid $3 \times 3$ feed-forward convolutions. While effective for axis-aligned objects, this design is fundamentally mismatched with the morphology of runway cracks: thin, elongated, non-linear structures for which a square $3 \times 3$ kernel wastes the majority of its sampling positions on irrelevant background texture.

#### Formulation
In **DC2PSA**, we replace the standard feed-forward convolutions within the C2PSA attention block with **Dynamic Snake Convolutions (DSConv)** (Qi et al., ICCV 2023). Given an input feature map $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ and a 1-D convolutional kernel of size $K$ centered at spatial position $p_i = (x_i,\, y_i)$, DSConv introduces learnable offset parameters $\Delta p_i = (\Delta x_i,\, \Delta y_i)$ that iteratively propagate along the underlying crack trajectory:

$$\text{x-axis path:} \quad p_{i+c} = \Bigl(x_i + c,\;\; y_i + \textstyle\sum_{j=i}^{i+c} \Delta y_j\Bigr), \quad c \in \bigl[-\lfloor K/2 \rfloor,\; \lfloor K/2 \rfloor\bigr]$$

$$\text{y-axis path:} \quad p_{i+c} = \Bigl(x_i + \textstyle\sum_{j=i}^{i+c} \Delta x_j,\;\; y_i + c\Bigr), \quad c \in \bigl[-\lfloor K/2 \rfloor,\; \lfloor K/2 \rfloor\bigr]$$

The cumulative summation constraint $\sum_{j=i}^{i+c} \Delta y_j$ enforces **structural continuity** along the sampling path, preventing receptive-field dispersion and ensuring the convolution traces the continuous curvature of crack fractures—unlike standard deformable convolutions (DCNv2), which permit unconstrained scattering of sampling positions.

The multi-head self-attention mechanism then operates on the morphologically refined features:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \qquad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \qquad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q},\, \mathbf{K},\, \mathbf{V}) = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\!\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{Y}_{\text{DC2PSA}} = \text{LayerNorm}\!\Bigl(\mathbf{X} + \mathbf{W}_O \cdot \text{Attention}(\mathbf{Q},\, \mathbf{K},\, \mathbf{V}) + \text{DSConv}(\mathbf{X})\Bigr)$$

The parallel residual connection of attention output and DSConv output enables the module to simultaneously leverage global contextual reasoning (via self-attention) and local morphological adaptation (via snake convolution), yielding features that are both semantically rich and geometrically precise.

### 5.3 Instance Segmentation Head

The segmentation branch follows a prototype-based architecture:

1. **Protonet:** A fully convolutional sub-network applied to the highest-resolution feature stage ($P_3$) generates $k = 32$ prototype masks $\mathbf{P} \in \mathbb{R}^{k \times H/4 \times W/4}$.
2. **Mask coefficients:** For each detected instance, the detection head predicts $k$ scalar coefficients $\mathbf{m} = [m_1,\, m_2,\, \dots,\, m_k]$.
3. **Mask assembly:** The final instance mask is computed via linear combination, sigmoid activation, and bounding-box cropping:

$$\mathbf{M}_{\text{instance}} = \sigma\!\left(\sum_{j=1}^{k} m_j\, \mathbf{P}_j\right) \odot \mathbf{B}_{\text{crop}}$$

### 5.4 Joint Multi-Task Loss Function

The network is trained end-to-end with a composite loss:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}}\,\mathcal{L}_{\text{CIoU}} \;+\; \lambda_{\text{cls}}\,\mathcal{L}_{\text{BCE}} \;+\; \lambda_{\text{dfl}}\,\mathcal{L}_{\text{L1}} \;+\; \lambda_{\text{mask}}\,\mathcal{L}_{\text{mask}}$$

| Component | Definition | Weight |
|---|---|---|
| $\mathcal{L}_{\text{CIoU}}$ | Complete IoU loss evaluating box overlap, centroid distance, and aspect ratio consistency | $\lambda_{\text{box}} = 7.5$ |
| $\mathcal{L}_{\text{BCE}}$ | Binary cross-entropy for single-class crack classification | $\lambda_{\text{cls}} = 0.5$ |
| $\mathcal{L}_{\text{L1}}$ | Stride-normalized L1 regression for bounding-box edges (YOLO26 DFL replacement) | $\lambda_{\text{dfl}} = 1.5$ |
| $\mathcal{L}_{\text{mask}}$ | $\text{BCE}(\mathbf{M}_{\text{pred}},\, \mathbf{M}_{\text{gt}}) + \text{Dice}(\mathbf{M}_{\text{pred}},\, \mathbf{M}_{\text{gt}})$ for pixel-accurate mask supervision | $\lambda_{\text{mask}} = 2.5$ |

---

## 6. FAA PCI Geometric Quantification Framework

A defining contribution of this work is the automated extraction of physical crack dimensions from predicted instance masks, directly aligning model output with civil aviation regulatory requirements under **ASTM D5340** and **FAA AC 150/5380-6C**:

```
                      AUTOMATED FAA PCI GEOMETRIC PIPELINE
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│ Binary Instance Mask      │ ──► │ Medial-Axis Skeleton      │ ──► │ Physical Length (L)       │
│ (pixel-accurate contour)  │     │ (Zhang–Suen thinning)     │     │ L = GSD × Σ‖Δp‖₂        │
└───────────────────────────┘     └───────────────────────────┘     └─────────────┬─────────────┘
                                                                                  ▼
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│ ASTM D5340 Severity Grade │ ◄── │ Mean Physical Width (W)   │ ◄── │ Distressed Area (A)       │
│ Low / Medium / High       │     │ W = A / L                 │     │ A = GSD² × Σ pixels       │
└───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
```

1. **Topological skeletonization:** The predicted crack mask is reduced to a 1-pixel-wide medial axis $\mathcal{S}$ via Zhang–Suen morphological thinning.
2. **Crack length ($L$):**
   $$L = \text{GSD} \times \sum_{i=1}^{|\mathcal{S}|-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$
3. **Distressed area ($A$):**
   $$A = \text{GSD}^2 \times \sum_{(x,y)} \mathbb{I}\bigl(\mathbf{M}_{\text{instance}}(x, y) = 1\bigr)$$
4. **Mean crack width ($W$):**
   $$W = A \;/\; L$$
5. **ASTM D5340 severity classification:**

| Severity Grade | Criterion |
|---|---|
| **Low (L)** | $W < 10\text{ mm}$; minimal edge spalling; no FOD generation |
| **Medium (M)** | $10\text{ mm} \leq W \leq 25\text{ mm}$; slight spalling; minor FOD potential |
| **High (H)** | $W > 25\text{ mm}$; severe joint disintegration; immediate NOTAM closure consideration |

---

## 7. Experimental Design and Deliverables

### 7.1 Model Benchmarking Matrix (9 Architectures × 2 Tasks)

| # | Architecture | Detection Weights | Segmentation Weights | Backbone / Attention | Params (Det / Seg) | GFLOPs (Det / Seg) |
|---|---|---|---|---|---|---|
| 1 | YOLOv8n | `yolov8n.pt` | `yolov8n-seg.pt` | C2f | 3.2M / 3.4M | 8.7 / 12.6 |
| 2 | YOLOv8s | `yolov8s.pt` | `yolov8s-seg.pt` | C2f | 11.2M / 11.8M | 28.6 / 35.7 |
| 3 | YOLO11n | `yolo11n.pt` | `yolo11n-seg.pt` | C3k2 + C2PSA | 2.6M / 2.9M | 6.4 / 10.1 |
| 4 | YOLO11s | `yolo11s.pt` | `yolo11s-seg.pt` | C3k2 + C2PSA | 9.4M / 10.1M | 21.5 / 28.4 |
| 5 | YOLO12n | `yolo12n.pt` | `yolo12n-seg.pt` | Area Attention | 2.6M / 2.8M | 6.9 / 10.5 |
| 6 | YOLO12s | `yolo12s.pt` | `yolo12s-seg.pt` | Area Attention | 9.3M / 9.9M | 21.4 / 28.1 |
| 7 | YOLO26n | `yolo26n.pt` | `yolo26n-seg.pt` | C3k2 + C2PSA (DFL-free) | 2.6M / 2.8M | 6.9 / 10.4 |
| 8 | YOLO26s | `yolo26s.pt` | `yolo26s-seg.pt` | C3k2 + C2PSA (DFL-free) | 9.6M / 10.3M | 26.4 / 34.2 |
| **9** | **DC2PSA-YOLO26s (Ours)** | **Custom YAML** | **Custom YAML** | **C3k2 + DC2PSA (DSConv)** | **~9.9M / ~10.7M** | **~27.1 / ~35.0** |

### 7.2 Training Protocol

All models are trained under hyperparameters **identical** to the baseline paper to enable controlled comparison:

| Hyperparameter | Value |
|---|---|
| Input resolution | 640 × 640 pixels |
| Training epochs | 100 (with 3-epoch linear warmup) |
| Batch size | 16 |
| Optimizer | SGD (momentum = 0.937, weight decay = $5 \times 10^{-4}$) |
| Learning rate schedule | Cosine annealing: $\eta_0 = 1 \times 10^{-3}$ → $\eta_f = 1 \times 10^{-5}$ |
| Loss weights (box / cls / dfl) | 7.5 / 0.5 / 1.5 |
| Pre-training | COCO-pretrained weights |
| Data augmentation | Mosaic ($p{=}1.0$), H/V flip ($p{=}0.5$), scale (${\pm}50\%$), rotation (${\pm}10°$), HSV jitter ($H{=}0.015,\, S{=}0.7,\, V{=}0.4$), copy-paste ($p{=}0.1$) |
| Seed variability | Seeds 42, 0, 123 for statistical reporting (mean ± std) |

### 7.3 Planned Deliverables: 14 Tables and 11 Figures

#### Tables

| Table | Content |
|---|---|
| Table 1 | Airport runway distress class definition and CrackAirport dataset statistics |
| Table 2 | Architectural complexity comparison: parameters, GFLOPs, and layer breakdown across all 9 models |
| Table 3 | Training hyperparameters and augmentation schedule (controlled settings) |
| Table 4 | Bounding-box detection benchmark: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 for all 9 models |
| Table 5 | Instance segmentation benchmark: Mask mAP@0.5, Mask mAP@0.5:0.95, Mask-F1 for all 9 models |
| Table 6 | Dual-task performance trade-off analysis (detection accuracy vs. segmentation quality) |
| Table 7 | Seed variability: 3-seed mean ± standard deviation for YOLO26s, YOLOv8s, and DC2PSA-YOLO26s |
| Table 8 | Computational efficiency: parameters, GFLOPs, wall-clock training time, model file size |
| Table 9 | Training and validation losses at convergence: box, classification, DFL, and mask loss components |
| Table 10 | Comparison with state-of-the-art airport runway distress detection methods |
| Table 11 | Ablation study: standard C2PSA vs. DCNv2 attention vs. proposed DC2PSA with DSConv |
| Table 12 | DSConv kernel length sensitivity analysis (K = 5, 7, 9, 11) |
| Table 13 | Edge-hardware deployment benchmark: NVIDIA Jetson Orin Nano vs. Raspberry Pi 5 (ms/frame, FPS) |
| Table 14 | Environmental robustness under five degraded atmospheric conditions |

#### Figures

| Figure | Content |
|---|---|
| Figure 1 | Representative annotated CrackAirport drone imagery with both bounding-box and polygon mask overlays |
| Figure 2 | Dataset spatial statistics: class distribution, bounding-box aspect ratio heatmaps, crack pixel area density |
| Figure 3 | Complete DC2PSA-YOLO26-Seg network architecture diagram |
| Figure 4 | Detailed internal comparison: standard C2PSA block vs. proposed DC2PSA block |
| Figure 5 | Receptive-field visualization: standard $3 \times 3$ convolution vs. Dynamic Snake Convolution on a crack sample |
| Figure 6 | Training and validation convergence trajectories: box, cls, dfl, and mask losses plus mAP over 100 epochs |
| Figure 7 | Precision–Recall and F1–confidence response curves |
| Figure 8 | Normalized confusion matrices for detection and instance segmentation tasks |
| Figure 9 | Qualitative detection comparison: YOLOv8s vs. YOLO26s vs. DC2PSA-YOLO26s (bounding-box overlays) |
| Figure 10 | Qualitative segmentation comparison: pixel-level crack mask visualizations on complex multi-branch cracks |
| Figure 11 | Cross-domain performance comparison: road pavement (*Sensors* 2026) vs. airport runway (this study) |

---

## 8. Cross-Domain Comparative Analysis

A distinctive theoretical contribution of this thesis is the **first controlled cross-domain transfer analysis** evaluating YOLO26 across two fundamentally different transportation infrastructure environments under identical experimental protocols:

| Evaluation Dimension | Road Pavement (*Sensors* 2026) | Airport Runway (This Study) |
|---|---|---|
| Dataset source | RDD2022 (4 countries; dashcam/motorbike) | CrackAirport (UAV at 100 ft AGL) |
| Total images | 6,972 | 2,262 |
| Class schema | 4 damage classes | 1 structural crack class |
| Detection task | Bounding-box only | Dual: bounding-box + instance segmentation |
| Proposed architecture | YOLO26s (off-the-shelf) | DC2PSA-YOLO26s-Seg (modified attention + segmentation) |
| Box mAP@0.5 | 89.0% | Target: >90.5% |
| Box mAP@0.5:0.95 | 51.6% | Target: >55.0% |
| Mask mAP@0.5 | Not available | Target: >86.5% |
| PCI quantification | None | Automated length, width, area extraction |

---

## 9. Edge Deployment and Environmental Robustness

### 9.1 Hardware Targets
To validate real-world UAV deployment feasibility, all trained models will be exported to **ONNX** and **TensorRT FP16** inference engines and benchmarked on two representative edge platforms:

| Platform | Target Performance |
|---|---|
| **NVIDIA Jetson Orin Nano** (8 GB, TensorRT FP16) | > 45 FPS |
| **Raspberry Pi 5** (8 GB, ONNX Runtime CPU) | < 80 ms per frame |

### 9.2 Atmospheric Degradation Testing
Runway surveillance systems must operate under adverse weather. We apply synthetic perturbations via the Albumentations library to simulate five operationally relevant conditions:

| Condition | Simulation Method |
|---|---|
| Heavy rain | Synthesized rain streak overlay ($p = 1.0$) |
| Dense advection fog | Atmospheric scattering coefficient $\beta \in [0.3,\, 0.5]$ |
| Low-light / dawn–dusk operations | Gamma correction ($\gamma = 2.5$) with contrast attenuation |
| UAV motion blur | Directional blur kernel ($k = 15$–$25$ px) |
| Cloud / structural shadow | Solar occlusion shadow mapping |

---

## 10. Execution Timeline

| Phase | Activities | Deliverables | Duration |
|---|---|---|---|
| **1** | Dataset conversion to dual-format (bbox + polygon); statistical verification | `dataset_seg/`, `data.yaml`, distribution plots | Weeks 1–2 |
| **2** | DC2PSA module implementation; custom YAML configuration; gradient-flow verification | `DSConv.py`, `DC2PSA.py`, `yolo26s-dc2psa-seg.yaml` | Weeks 3–4 |
| **3** | Full 9-model × 2-task benchmark training (seed 42) | 18 trained model checkpoints; training logs | Weeks 5–7 |
| **4** | Seed variability (seeds 0, 42, 123) and ablation studies | Statistical tables; ablation comparison | Weeks 8–9 |
| **5** | Edge deployment export and environmental robustness testing | ONNX/TensorRT models; degradation tables | Week 10 |
| **6** | PCI geometric pipeline implementation; compilation of all 14 tables and 11 figures | Complete results package | Week 11 |
| **7** | Manuscript drafting; cross-domain synthesis; submission preparation | Final thesis manuscript | Weeks 12–13 |

---

## 11. Target Publication Venues

| Venue | Impact Factor | Rationale |
|---|---|---|
| **Sensors (MDPI)** | ~3.4 | Direct sequel to our published baseline, ensuring narrative coherence and rapid peer review |
| **Automation in Construction (Elsevier)** | ~10.3 | Premier infrastructure automation journal; strong fit for novel DC2PSA architecture and automated PCI framework |
| **IEEE Trans. Intelligent Transportation Systems** | ~8.5 | High-impact venue for computer vision applied to aviation transportation infrastructure |
| **Drones (MDPI)** | ~4.8 | Specialized venue for UAV-based automated inspection and edge-deployed defect segmentation |

---

## 12. Novelty Statement

This thesis advances the state of the art through four original contributions:

1. **DC2PSA-YOLO26-Seg** — the first YOLO26-based architecture purpose-built for airport runway crack detection, integrating a deformable spatial attention module with a concurrent instance segmentation branch.
2. **DC2PSA (Deformable C2PSA with Dynamic Snake Convolution)** — a novel attention mechanism that replaces fixed-grid convolutions with topology-constrained deformable kernels, enabling adaptive receptive-field alignment with irregular crack morphologies.
3. **Joint detection and instance segmentation** — the first dual-task framework that simultaneously produces bounding-box detections and pixel-level crack masks from a single forward pass on aviation pavement imagery, enabling automated geometric quantification aligned with FAA PCI assessment standards.
4. **Controlled cross-domain analysis** — the first investigation comparing YOLO26 performance on road pavement versus airport runway pavement under identical experimental conditions, quantifying the domain gap and informing transfer-learning strategies for infrastructure inspection.

---

## 13. References

1. Abbas, S.; Shawon, M.T.I.; Qamar, S.; Adeel, M. Evaluating YOLO26s for Multi-Class Pavement Crack Detection: A Lightweight Approach for Sustainable Edge Deployment. *Sensors* **2026**, *26*(16), 5113.
2. Lyu, Z.; Yu, A.; Starr, C. CrackAirport: A Dataset for Segmentation of Cracks in Airport Pavements. *Mendeley Data* **2026**, V1, DOI: 10.17632/3v5r2fxf89.1.
3. Qi, Y.; He, Y.; Qi, X.; Zhang, Y.; Yang, G. Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*; **2023**; pp. 6070–6079.
4. Dai, J.; Qi, H.; Xiong, Y.; Li, Y.; Zhang, G.; Hu, H.; Wei, Y. Deformable Convolutional Networks. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*; **2017**; pp. 764–773.
5. Arya, D.; Maeda, H.; Ghosh, S.K.; Toshniwal, D.; Sekimoto, Y. Global Road Damage Detection: State-of-the-Art Solutions. *IEEE Trans. Intell. Transp. Syst.* **2022**, *23*(11), 20807–20819.
6. Federal Aviation Administration (FAA). Guidelines and Procedures for Maintenance of Airport Pavements. *Advisory Circular 150/5380-6C*; U.S. Department of Transportation: Washington, DC, USA, 2014.
7. ASTM International. Standard Test Method for Airport Pavement Condition Index Surveys. *ASTM D5340-20*; ASTM International: West Conshohocken, PA, USA, 2020.
8. Ultralytics. YOLO26: Real-Time Object Detection and Instance Segmentation. *Ultralytics Documentation*, 2026. Available online: https://docs.ultralytics.com.

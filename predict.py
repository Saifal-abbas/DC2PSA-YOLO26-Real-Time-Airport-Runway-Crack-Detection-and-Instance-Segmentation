"""
DC2PSA-YOLO26-Seg Inference and Runway Crack Inspection CLI
===========================================================

Official inference & crack quantification script for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Usage:
------
# 1. Inference on single image with ASTM D5340 crack severity quantification
python predict.py --weights runs/seg/dc2psa-yolo26s_seg_seed42/weights/best.pt --source path/to/runway_image.jpg

# 2. Batch inference on an entire folder of runway drone images
python predict.py --weights runs/seg/dc2psa-yolo26s_seg_seed42/weights/best.pt --source path/to/drone_images/ --conf 0.25
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def extract_crack_geometry(binary_mask: np.ndarray, gsd_mm: float = 1.5) -> dict:
    """
    Extract physical crack dimensions from a binary segmentation mask.

    Implements Section 6 of the Research Proposal & Sensors 2026 paper:
    - Topological skeletonization (Zhang-Suen thinning)
    - Length quantification via skeleton path integration (L = ∑ dist × GSD)
    - Surface area quantification (A = ∑ pixels × GSD²)
    - Mean crack width (W = A / L)
    - ASTM D5340 Severity Classification:
        - Low    : Width < 10.0 mm
        - Medium : 10.0 mm <= Width <= 25.0 mm
        - High   : Width > 25.0 mm

    Args:
        binary_mask (np.ndarray): Binary mask (H, W) where crack pixels > 0.
        gsd_mm (float): Ground Sampling Distance in mm/pixel (CrackAirport: 1.5 mm/px).

    Returns:
        dict: Physical crack measurements and severity category.
    """
    mask_binary = (binary_mask > 0).astype(np.uint8)
    area_px = int(mask_binary.sum())
    if area_px == 0:
        return {"length_mm": 0.0, "width_mm": 0.0, "area_mm2": 0.0, "severity": "None"}

    area_mm2 = area_px * (gsd_mm ** 2)

    # Skeletonization (morphological thinning)
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask_binary).astype(np.uint8)
    except ImportError:
        if hasattr(cv2, "ximgproc"):
            skeleton = cv2.ximgproc.thinning(mask_binary * 255)
            skeleton = (skeleton > 0).astype(np.uint8)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, kernel)

    skeleton_px = int(skeleton.sum())

    if skeleton_px > 0:
        ys, xs = np.where(skeleton > 0)
        if len(xs) > 1:
            dists = np.sqrt(np.diff(xs.astype(float)) ** 2 + np.diff(ys.astype(float)) ** 2)
            length_px = float(np.sum(dists))
        else:
            length_px = 1.0
    else:
        length_px = max(1.0, float(area_px ** 0.5))

    length_mm = length_px * gsd_mm
    width_mm = area_mm2 / max(length_mm, 1e-6)

    # ASTM D5340 airport pavement condition index severity thresholds
    if width_mm < 10.0:
        severity = "Low (<10mm)"
    elif width_mm <= 25.0:
        severity = "Medium (10-25mm)"
    else:
        severity = "High (>25mm)"

    return {
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "severity": severity,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and crack quantification using DC2PSA-YOLO26-Seg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image, directory, or video source")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Inference device ('0', 'cpu', etc.)")
    parser.add_argument("--gsd", type=float, default=1.5, help="Ground Sampling Distance in mm/pixel")
    parser.add_argument("--save-dir", type=str, default="runs/predict", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display visual predictions")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("  DC2PSA-YOLO26-Seg Automated Crack Inspection & Quantification")
    print("=" * 80)
    print(f"  Model Weights : {args.weights}")
    print(f"  Input Source  : {args.source}")
    print(f"  Confidence    : {args.conf}")
    print(f"  Resolution    : {args.imgsz}x{args.imgsz}")
    print(f"  GSD Calib.    : {args.gsd} mm/pixel")
    print(f"  Output Dir    : {args.save_dir}")
    print("=" * 80 + "\n")

    model = YOLO(args.weights)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("  Inspection & ASTM D5340 Severity Assessment")
    print("=" * 80)

    total_cracks = 0
    for idx, r in enumerate(results):
        img_name = Path(r.path).name if hasattr(r, "path") else f"frame_{idx}"
        n_instances = len(r.boxes) if r.boxes is not None else 0
        total_cracks += n_instances
        print(f"\nImage: {img_name} -> {n_instances} crack instance(s) detected")

        if r.masks is not None:
            # Process each segmentation mask
            for c_idx, mask_data in enumerate(r.masks.data):
                m_np = mask_data.cpu().numpy()
                geo = extract_crack_geometry(m_np, gsd_mm=args.gsd)
                box_conf = float(r.boxes.conf[c_idx].cpu().numpy()) if r.boxes is not None else 0.0
                print(
                    f"  Crack #{c_idx + 1:02d} | Conf: {box_conf:.2f} | "
                    f"Length: {geo['length_mm']:6.1f} mm | "
                    f"Width: {geo['width_mm']:5.1f} mm | "
                    f"Area: {geo['area_mm2']:8.1f} mm² | "
                    f"Severity: {geo['severity']}"
                )

    print("\n" + "=" * 80)
    print(f"  [OK] Batch Inspection Complete: {len(results)} images inspected, {total_cracks} cracks analyzed.")
    print(f"  Visualizations saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

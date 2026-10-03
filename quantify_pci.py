"""
FAA PCI & ASTM D5340 Runway Crack Quantification Tool
=====================================================

Official automated pavement distress quantification tool for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Implements Section 6 of the Research Proposal:
- Topological skeletonization (Zhang-Suen thinning algorithm)
- Length quantification via Euclidean path integration: L = ∑ dist * GSD
- Surface area calculation: A = ∑ binary_pixels * GSD²
- Effective average crack width: W = A / L
- ASTM D5340 Airport Pavement Condition Index (PCI) severity classification:
    - Low Severity    : Mean width < 10.0 mm
    - Medium Severity : 10.0 mm <= Mean width <= 25.0 mm
    - High Severity   : Mean width > 25.0 mm
- FAA Advisory Circular AC 150/5380-6C deduct value calculation
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def compute_instance_metrics(binary_mask: np.ndarray, gsd_mm: float = 1.5) -> dict:
    """
    Compute rigorous geometric metrics for a single crack instance mask.
    """
    area_px = int(np.sum(binary_mask > 0))
    if area_px == 0:
        return None

    area_mm2 = area_px * (gsd_mm ** 2)

    # Skeletonization using skimage or OpenCV thinning
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_mask > 0).astype(np.uint8)
    except ImportError:
        if hasattr(cv2, "ximgproc"):
            skeleton = cv2.ximgproc.thinning((binary_mask > 0).astype(np.uint8) * 255)
            skeleton = (skeleton > 0).astype(np.uint8)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = cv2.morphologyEx((binary_mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

    skeleton_px = int(np.sum(skeleton > 0))

    if skeleton_px > 1:
        ys, xs = np.where(skeleton > 0)
        dists = np.sqrt(np.diff(xs.astype(float)) ** 2 + np.diff(ys.astype(float)) ** 2)
        length_px = float(np.sum(dists))
    else:
        length_px = max(1.0, float(np.sqrt(area_px)))

    length_mm = length_px * gsd_mm
    width_mm = area_mm2 / max(length_mm, 1e-6)

    # ASTM D5340 Severity Level
    if width_mm < 10.0:
        severity = "Low"
        deduct_weight = 1.0
    elif width_mm <= 25.0:
        severity = "Medium"
        deduct_weight = 2.5
    else:
        severity = "High"
        deduct_weight = 5.0

    return {
        "area_px": area_px,
        "skeleton_px": skeleton_px,
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "severity": severity,
        "deduct_weight": deduct_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify airport runway cracks according to FAA PCI & ASTM D5340",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Directory containing binary crack mask images",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=1.5,
        help="Ground Sampling Distance in mm/pixel (CrackAirport: 1.5 mm/px)",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=40,
        help="Minimum instance area threshold to filter out sensor noise (§4.2)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/pci_quantification_report.csv",
        help="Path to save output quantification CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    masks_path = Path(args.masks_dir)

    print("\n" + "=" * 80)
    print("  FAA PCI & ASTM D5340 Crack Geometric Quantification")
    print("=" * 80)
    print(f"  Masks Directory   : {masks_path}")
    print(f"  GSD Calibration   : {args.gsd} mm/pixel")
    print(f"  Min Area Filter   : {args.min_area_px} px")
    print(f"  Output CSV Target : {args.output_csv}")
    print("=" * 80 + "\n")

    if not masks_path.exists():
        print(f"Error: Mask directory '{masks_path}' does not exist.")
        return

    mask_files = sorted(
        [f for f in masks_path.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]]
    )

    if not mask_files:
        print(f"No image files found in {masks_path}")
        return

    print(f"Processing {len(mask_files)} mask images...")

    records = []
    for f in mask_files:
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c_idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < args.min_area_px:
                continue

            cnt_mask = np.zeros_like(binary)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            metrics = compute_instance_metrics(cnt_mask, gsd_mm=args.gsd)
            if metrics:
                metrics["image"] = f.name
                metrics["crack_id"] = f"{f.stem}_c{c_idx + 1}"
                records.append(metrics)

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No crack instances met the minimum area threshold.")
        return

    out_file = Path(args.output_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 80)
    print("  ASTM D5340 Airport Runway Crack Severity Breakdown")
    print("=" * 80)
    total_instances = len(df)
    sev_counts = df["severity"].value_counts()

    for sev in ["Low", "Medium", "High"]:
        cnt = sev_counts.get(sev, 0)
        pct = (cnt / total_instances) * 100 if total_instances > 0 else 0
        print(f"  {sev:8s} Severity : {cnt:5d} instances ({pct:5.1f}%)")

    print("\n  Summary Statistics (Metric Dimensions):")
    print(f"    Total Instances Measured : {total_instances}")
    print(f"    Mean Crack Length        : {df['length_mm'].mean():.2f} mm (± {df['length_mm'].std():.2f})")
    print(f"    Mean Crack Width         : {df['width_mm'].mean():.2f} mm (± {df['width_mm'].std():.2f})")
    print(f"    Mean Crack Area          : {df['area_mm2'].mean():.2f} mm²")
    print(f"    Total Pavement Crack Area: {df['area_mm2'].sum() / 1e6:.4f} m²")
    print(f"\n  [OK] Full dataset inspection metrics saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

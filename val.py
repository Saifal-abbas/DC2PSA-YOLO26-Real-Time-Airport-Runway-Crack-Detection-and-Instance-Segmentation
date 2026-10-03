"""
DC2PSA-YOLO26-Seg Model Validation and Benchmark Evaluation CLI
===============================================================

Official evaluation script for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Usage:
------
# Evaluate Detection Model on Test Set
python val.py --weights runs/det/dc2psa-yolo26s_det_seed42/weights/best.pt --data dataset/data.yaml --split test

# Evaluate Segmentation Model on Test Set
python val.py --weights runs/seg/dc2psa-yolo26s_seg_seed42/weights/best.pt --data dataset_seg/data.yaml --split test
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DC2PSA-YOLO26s or baseline models on validation/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weights file (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Object confidence threshold for mAP computation",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.65,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Computation device ('0', 'cpu', etc.)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save metrics to JSON output file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/val",
        help="Directory to save validation outputs and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("  DC2PSA-YOLO26-Seg Model Validation")
    print("=" * 80)
    print(f"  Weights      : {args.weights}")
    print(f"  Dataset YAML : {args.data}")
    print(f"  Split        : {args.split}")
    print(f"  Image Size   : {args.imgsz}")
    print(f"  Batch Size   : {args.batch}")
    print(f"  Output Dir   : {args.save_dir}")
    print("=" * 80 + "\n")

    model = YOLO(args.weights)

    val_kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "plots": True,
        "project": args.save_dir,
        "name": Path(args.weights).stem + f"_{args.split}",
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)

    print("\n" + "=" * 80)
    print("  Validation Results Summary")
    print("=" * 80)

    # Box Metrics
    if hasattr(metrics, "box"):
        b_p = metrics.box.mp
        b_r = metrics.box.mr
        b_map50 = metrics.box.map50
        b_map = metrics.box.map
        b_f1 = (2 * b_p * b_r / (b_p + b_r + 1e-8)) if (b_p + b_r) > 0 else 0.0
        print(f"  [Bounding Box]")
        print(f"    Precision   (P)     : {b_p * 100:.2f}%")
        print(f"    Recall      (R)     : {b_r * 100:.2f}%")
        print(f"    F1-Score    (F1)    : {b_f1 * 100:.2f}%")
        print(f"    mAP@0.5     (mAP50) : {b_map50 * 100:.2f}%")
        print(f"    mAP@0.5:0.95(mAP)   : {b_map * 100:.2f}%")

    # Mask Metrics
    if hasattr(metrics, "seg") and metrics.seg is not None:
        m_p = metrics.seg.mp
        m_r = metrics.seg.mr
        m_map50 = metrics.seg.map50
        m_map = metrics.seg.map
        m_f1 = (2 * m_p * m_r / (m_p + m_r + 1e-8)) if (m_p + m_r) > 0 else 0.0
        print(f"\n  [Instance Segmentation Mask]")
        print(f"    Mask Precision (P)  : {m_p * 100:.2f}%")
        print(f"    Mask Recall    (R)  : {m_r * 100:.2f}%")
        print(f"    Mask F1-Score  (F1) : {m_f1 * 100:.2f}%")
        print(f"    Mask mAP@0.5        : {m_map50 * 100:.2f}%")
        print(f"    Mask mAP@0.5:0.95   : {m_map * 100:.2f}%")

    print("=" * 80)

    if args.save_json:
        out_json = Path(args.save_dir) / f"{Path(args.weights).stem}_{args.split}_metrics.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        results_data = metrics.results_dict if hasattr(metrics, "results_dict") else {}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\n[OK] Metrics saved to {out_json}")


if __name__ == "__main__":
    main()

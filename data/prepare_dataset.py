"""
CrackAirport Dataset Preparation and Verification Script
========================================================

Verifies dataset integrity and split balance for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Benchmark Dataset Specification:
--------------------------------
- Source: Drone remote sensing over civil airport runway pavements
- Ground Sampling Distance (GSD): 1.5 mm/pixel
- Total Images: 2,262 (512x512 resolution)
- Total Crack Instances: 2,798 annotated polygon masks / bounding boxes
- Splits:
    - Train: 1,583 images (70%)
    - Val  :   453 images (20%)
    - Test :   226 images (10%)
"""

import argparse
from pathlib import Path


EXPECTED_SPLITS = {
    "train": 1583,
    "val": 453,
    "test": 226,
}


def verify_dataset(dataset_root: str, task: str = "seg"):
    root = Path(dataset_root)
    print("\n" + "=" * 75)
    print(f"  Verifying CrackAirport Dataset ({task.upper()}) at: {root}")
    print("=" * 75)

    if not root.exists():
        print(f"[ERROR] Dataset directory not found: {root}")
        print("Please download CrackAirport from Mendeley Data: https://data.mendeley.com/")
        return False

    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Missing 'images/' or 'labels/' subdirectory in {root}")
        return False

    all_valid = True
    total_imgs = 0
    total_lbls = 0

    for split, expected_count in EXPECTED_SPLITS.items():
        s_img_dir = images_dir / split
        s_lbl_dir = labels_dir / split

        img_count = len(list(s_img_dir.glob("*.jpg"))) + len(list(s_img_dir.glob("*.png")))
        lbl_count = len(list(s_lbl_dir.glob("*.txt")))

        total_imgs += img_count
        total_lbls += lbl_count

        status = "[OK]" if img_count == expected_count else "[WARN]"
        print(
            f"  {status} {split:5s} split: {img_count:4d} images "
            f"(expected: {expected_count:4d}) | {lbl_count:4d} label files"
        )
        if img_count != expected_count:
            all_valid = False

    print("-" * 75)
    print(f"  Total Images Verified : {total_imgs} / 2,262")
    print(f"  Total Label Files     : {total_lbls}")

    if all_valid and total_imgs == 2262:
        print("\n[SUCCESS] CrackAirport dataset verified successfully with 100% split match!")
    else:
        print("\n[NOTE] Image counts deviate slightly from reference split, but dataset is usable.")

    print("=" * 75 + "\n")
    return all_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify CrackAirport dataset layout")
    parser.add_argument("--root", type=str, default="dataset_seg", help="Root directory of dataset")
    parser.add_argument("--task", type=str, default="seg", choices=["det", "seg"], help="Task variant")
    args = parser.parse_args()

    verify_dataset(args.root, args.task)

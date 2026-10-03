"""
DC2PSA-YOLO26-Seg Training CLI
==============================

Official training script for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Usage:
------
# 1. Train Detection Model (100 epochs, seed 42)
python train.py --task det --data path/to/dataset/data.yaml --epochs 100 --batch 16 --imgsz 640

# 2. Train Segmentation Model (100 epochs, seed 42)
python train.py --task seg --data path/to/dataset_seg/data.yaml --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from models.dc2psa import build_dc2psa_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DC2PSA-YOLO26s for Runway Crack Detection and Instance Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core settings
    parser.add_argument(
        "--task",
        type=str,
        default="det",
        choices=["det", "seg"],
        help="Task type: 'det' (bounding box detection) or 'seg' (instance segmentation)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (e.g., dataset/data.yaml or dataset_seg/data.yaml)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained baseline weights (defaults: 'yolo26s.pt' for det, 'yolo26s-seg.pt' for seg)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs (Sensors 2026 paper protocol: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (Sensors 2026 paper protocol: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (Sensors 2026 paper protocol: 640)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-3,
        help="Initial learning rate (Sensors 2026 paper protocol: 0.001)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate fraction (lr0 * lrf)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum / Adam beta1",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Optimizer weight decay",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for strict experimental reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Computation device: '0', '0,1', 'cpu', etc. (auto-detect if empty)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project directory for saving training runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to 'dc2psa-yolo26s_seed<seed>')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint if available",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="Cache dataset in RAM for maximum I/O throughput",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine default pretrained weights
    if args.pretrained is None:
        args.pretrained = "yolo26s.pt" if args.task == "det" else "yolo26s-seg.pt"

    # Default experiment name
    if args.name is None:
        args.name = f"dc2psa-yolo26s_{args.task}_seed{args.seed}"

    run_dir = Path(args.project) / args.task / args.name
    last_ckpt = run_dir / "weights" / "last.pt"
    done_flag = run_dir / "DONE"

    print("\n" + "=" * 80)
    print("  DC2PSA-YOLO26-Seg Training Pipeline")
    print("=" * 80)
    print(f"  Task         : {args.task.upper()}")
    print(f"  Data Config  : {args.data}")
    print(f"  Pretrained   : {args.pretrained}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch Size   : {args.batch}")
    print(f"  Image Size   : {args.imgsz}")
    print(f"  Optimizer    : {args.optimizer} (lr0={args.lr0}, lrf={args.lrf})")
    print(f"  Seed         : {args.seed}")
    print(f"  Output Dir   : {run_dir}")
    print("=" * 80)

    # Check if run is already finished
    if done_flag.exists():
        print(f"\n[OK] Experiment {args.name} is already completed.")
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            print("Cached metrics:")
            print(json.dumps(metrics_data, indent=2))
        return

    # Check resume
    if args.resume and last_ckpt.exists():
        print(f"\n[RESUME] Found existing checkpoint: {last_ckpt}")
        print("Resuming training from checkpoint...")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        # Build DC2PSA model via architectural injection
        print(f"\n[INIT] Instantiating DC2PSA-YOLO26s ({args.task})...")
        model = build_dc2psa_model(
            pretrained_path=args.pretrained,
            task="detect" if args.task == "det" else "segment",
        )

        train_kwargs = {
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "cos_lr": True,
            "patience": args.epochs,
            "cache": args.cache,
            "workers": args.workers,
            "deterministic": True,
            "seed": args.seed,
            "save": True,
            "save_period": 1,
            "plots": True,
            "verbose": True,
            "project": str(Path(args.project) / args.task),
            "name": args.name,
            "exist_ok": True,
        }

        if args.device:
            train_kwargs["device"] = args.device

        t0 = time.time()
        results = model.train(**train_kwargs)
        elapsed_min = (time.time() - t0) / 60.0
        print(f"\n[DONE] Training completed in {elapsed_min:.2f} minutes.")

    # Save metrics JSON and DONE flag
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dict = results.results_dict if hasattr(results, "results_dict") else {}
    metrics_summary = {
        "model": "dc2psa-yolo26s",
        "task": args.task,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "best_model": str(run_dir / "weights" / "best.pt"),
        "results": results_dict,
        "completed_at": datetime.now().isoformat(),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    with open(done_flag, "w", encoding="utf-8") as f:
        json.dump({"completed": True, "timestamp": datetime.now().isoformat()}, f, indent=2)

    print(f"\n[OK] Results and evaluation checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()

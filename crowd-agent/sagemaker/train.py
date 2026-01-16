import os
import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# Simple training wrapper for fine-tuning a YOLOv8 model on person detection
# Expects dataset in YOLO format at /opt/ml/input/data/training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', 20)))
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get('BATCH_SIZE', 16)))
    parser.add_argument('--imgsz', type=int, default=int(os.environ.get('IMGSZ', 640)))
    parser.add_argument('--dataset_yaml', type=str, default=os.environ.get('DATASET_YAML', 'dataset.yaml'))
    parser.add_argument('--pretrained', type=str, default=os.environ.get('PRETRAINED', 'yolov8n.pt'))
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path('/opt/ml/input/data/training')
    model_dir = Path('/opt/ml/model')
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Using dataset at: {input_path}")
    print(f"[train] Hyperparameters: epochs={args.epochs}, batch={args.batch_size}, imgsz={args.imgsz}")

    model = YOLO(args.pretrained)
    # Train only on person class if dataset is configured so
    results = model.train(
        data=str(input_path / args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=str(model_dir),
        name='yolo-person',
        exist_ok=True
    )

    # Save best model to model_dir
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    if best.exists():
        dest = model_dir / 'model.pt'
        dest.write_bytes(best.read_bytes())
        print(f"[train] Saved best model to {dest}")
    else:
        print("[train] WARNING: best.pt not found; saving last.pt")
        last = Path(results.save_dir) / 'weights' / 'last.pt'
        dest = model_dir / 'model.pt'
        dest.write_bytes(last.read_bytes())

    # Save metadata
    meta = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'imgsz': args.imgsz,
        'timestamp': int(time.time())
    }
    (model_dir / 'meta.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    main()

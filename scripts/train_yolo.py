from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "data" / "processed" / "food_subset_yolo" / "data.yaml"
PROJECT_DIR = PROJECT_ROOT / "runs"


_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Force weights_only=False for old Ultralytics checkpoints on newer PyTorch."""
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


def main() -> None:
    torch.load = _patched_torch_load
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_CONFIG),
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project=str(PROJECT_DIR),
        name="food_subset_yolov8n",
    )


if __name__ == "__main__":
    main()

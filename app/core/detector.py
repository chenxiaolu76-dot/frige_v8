from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "config" / "system.yaml"
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Force weights_only=False for compatibility with this Ultralytics version."""
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


def load_system_config(config_path: str | Path = SYSTEM_CONFIG_PATH) -> dict[str, Any]:
    """
    Load system configuration from a YAML file.

    Input:
        config_path: Path to the YAML config file.

    Output:
        Parsed configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_model(config_path: str | Path = SYSTEM_CONFIG_PATH) -> YOLO:
    """
    Load a YOLOv8 model from the configured local weights path.

    Input:
        config_path: Path to the YAML config file.

    Output:
        Loaded YOLO model instance.
    """
    config = load_system_config(config_path)
    weights_path = config["model"]["yolo"]["weights_path"]
    model_path = (PROJECT_ROOT / weights_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")

    torch.load = _patched_torch_load
    return YOLO(str(model_path))


def detect_food(
    image: Any,
    model: YOLO,
    config_path: str | Path = SYSTEM_CONFIG_PATH,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Run YOLOv8 food detection on a preprocessed image.

    Input:
        image: Preprocessed image as a numpy array.
        model: Loaded YOLO model instance.
        config_path: Path to the YAML config file.

    Output:
        Detection results as a list of dictionaries.
    """
    config = load_system_config(config_path)
    yolo_config = config["model"]["yolo"]

    results = model.predict(
        source=image,
        conf=confidence_threshold if confidence_threshold is not None else yolo_config.get("confidence_threshold", 0.25),
        iou=iou_threshold if iou_threshold is not None else yolo_config.get("iou_threshold", 0.45),
        device=yolo_config.get("device", "cpu"),
        verbose=False,
    )

    return parse_detections(results)


def parse_detections(results: list[Any]) -> list[dict[str, Any]]:
    """
    Convert YOLO outputs into structured detection records.

    Input:
        results: Raw YOLO detection results.

    Output:
        Structured detection list with class, confidence, box and center point.
    """
    parsed_results: list[dict[str, Any]] = []

    for result in results:
        names = result.names
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            parsed_results.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                }
            )

    return parsed_results


def detections_to_dataframe(detections: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert parsed detections into a Pandas DataFrame.

    Input:
        detections: Structured detection list.

    Output:
        Detection DataFrame with class, confidence, box and center columns.
    """
    columns = [
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    ]
    return pd.DataFrame(detections, columns=columns)


def count_food(detections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count detected food items by class name.

    Input:
        detections_df: Detection DataFrame.

    Output:
        Food count DataFrame sorted by quantity.
    """
    if detections_df.empty:
        return pd.DataFrame(columns=["class_name", "count"])

    counts_df = (
        detections_df.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "class_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return counts_df


if __name__ == "__main__":
    from app.core.image_preprocess import load_image, preprocess_pipeline

    config = load_system_config()
    sample_image_path = PROJECT_ROOT / config["paths"]["sample_image"]

    if not sample_image_path.exists():
        print(f"Sample image not found: {sample_image_path}")
    else:
        image = load_image(sample_image_path)
        processed = preprocess_pipeline(
            image,
            use_gray=False,
            use_contrast=True,
            use_denoise=True,
            use_edge=False,
            use_morphology=False,
        )["final"]

        yolo_model = load_model()
        detections = detect_food(processed, yolo_model)
        detections_df = detections_to_dataframe(detections)
        counts_df = count_food(detections_df)

        print("Detection Results:")
        print(detections_df)
        print("\nFood Counts:")
        print(counts_df)

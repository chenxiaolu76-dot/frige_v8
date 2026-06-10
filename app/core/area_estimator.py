from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIXEL_AREA_MM2 = 4.0
REFERENCE_COIN_CLASS = "reference_coin"
REFERENCE_COIN_DIAMETER_MM = 25.0
REFERENCE_COIN_AREA_MM2 = 490.87


def _calc_box_pixel_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate bounding-box pixel area."""
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return width * height


def detect_reference_object(
    detections_df: pd.DataFrame,
    reference_class_name: str = REFERENCE_COIN_CLASS,
) -> dict[str, Any] | None:
    """
    Detect the reference object from detection results and compute its pixel area.

    Input:
        detections_df: Detection DataFrame from detector.py.
        reference_class_name: Reference object class name.

    Output:
        Reference object information dictionary, or None if not found.
    """
    if detections_df.empty:
        return None

    reference_df = detections_df[detections_df["class_name"] == reference_class_name].copy()
    if reference_df.empty:
        return None

    reference_df["pixel_area"] = reference_df.apply(
        lambda row: _calc_box_pixel_area(row["x1"], row["y1"], row["x2"], row["y2"]),
        axis=1,
    )
    best_match = reference_df.sort_values(by="confidence", ascending=False).iloc[0]

    return {
        "class_name": best_match["class_name"],
        "confidence": float(best_match["confidence"]),
        "pixel_area": float(best_match["pixel_area"]),
        "x1": float(best_match["x1"]),
        "y1": float(best_match["y1"]),
        "x2": float(best_match["x2"]),
        "y2": float(best_match["y2"]),
    }


def calc_pixel_to_mm(
    reference_pixel_area: float | None,
    reference_area_mm2: float = REFERENCE_COIN_AREA_MM2,
    default_pixel_area_mm2: float = DEFAULT_PIXEL_AREA_MM2,
) -> dict[str, Any]:
    """
    Calculate the pixel-to-area conversion coefficient using the reference object.

    Input:
        reference_pixel_area: Pixel area of the reference object.
        reference_area_mm2: Real area of the reference object in mm^2.
        default_pixel_area_mm2: Default mm^2 per pixel fallback value.

    Output:
        Conversion information dictionary.
    """
    if reference_pixel_area is None or reference_pixel_area <= 0:
        return {
            "mm2_per_pixel": default_pixel_area_mm2,
            "used_default": True,
            "message": "Reference object not found. Using default pixel coefficient.",
        }

    mm2_per_pixel = reference_area_mm2 / reference_pixel_area
    return {
        "mm2_per_pixel": mm2_per_pixel,
        "used_default": False,
        "message": "Reference object detected successfully.",
    }


def estimate_food_area(
    detection_row: pd.Series | dict[str, Any],
    mm2_per_pixel: float,
) -> dict[str, Any]:
    """
    Estimate the real-world area of one detected food item.

    Input:
        detection_row: One detection record as a Series or dictionary.
        mm2_per_pixel: Area conversion coefficient in mm^2 per pixel.

    Output:
        Area estimation dictionary.
    """
    row = detection_row if isinstance(detection_row, dict) else detection_row.to_dict()
    pixel_area = _calc_box_pixel_area(row["x1"], row["y1"], row["x2"], row["y2"])
    estimated_area_mm2 = pixel_area * mm2_per_pixel

    return {
        "class_name": row["class_name"],
        "confidence": float(row["confidence"]),
        "pixel_area": round(pixel_area, 2),
        "estimated_area_mm2": round(estimated_area_mm2, 2),
        "x1": float(row["x1"]),
        "y1": float(row["y1"]),
        "x2": float(row["x2"]),
        "y2": float(row["y2"]),
        "center_x": float(row["center_x"]),
        "center_y": float(row["center_y"]),
    }


def estimate_all_areas(
    detections_df: pd.DataFrame,
    reference_class_name: str = REFERENCE_COIN_CLASS,
    full_ratio_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Estimate real areas for all detected food items and label them as full or partial.

    Input:
        detections_df: Detection DataFrame from detector.py.
        reference_class_name: Reference object class name.
        full_ratio_threshold: Threshold used to mark a detection as partial.

    Output:
        Tuple of area estimation DataFrame and conversion info dictionary.
    """
    if detections_df.empty:
        empty_columns = [
            "class_name",
            "confidence",
            "pixel_area",
            "estimated_area_mm2",
            "portion_state",
            "area_ratio_to_max",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ]
        return pd.DataFrame(columns=empty_columns), calc_pixel_to_mm(None)

    reference_info = detect_reference_object(detections_df, reference_class_name=reference_class_name)
    reference_pixel_area = None if reference_info is None else reference_info["pixel_area"]
    conversion_info = calc_pixel_to_mm(reference_pixel_area)

    food_df = detections_df[detections_df["class_name"] != reference_class_name].copy()
    if food_df.empty:
        empty_columns = [
            "class_name",
            "confidence",
            "pixel_area",
            "estimated_area_mm2",
            "portion_state",
            "area_ratio_to_max",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ]
        return pd.DataFrame(columns=empty_columns), conversion_info

    estimation_rows = [
        estimate_food_area(row, conversion_info["mm2_per_pixel"])
        for _, row in food_df.iterrows()
    ]
    area_df = pd.DataFrame(estimation_rows)

    max_area_by_class = area_df.groupby("class_name")["estimated_area_mm2"].transform("max")
    area_df["area_ratio_to_max"] = (area_df["estimated_area_mm2"] / max_area_by_class).round(4)
    area_df["portion_state"] = area_df["area_ratio_to_max"].apply(
        lambda ratio: "半个/部分食材" if ratio < full_ratio_threshold else "完整食材"
    )

    return area_df, conversion_info


if __name__ == "__main__":
    from app.core.detector import count_food, detect_food, detections_to_dataframe, load_model
    from app.core.image_preprocess import load_image, preprocess_pipeline
    from app.core.detector import load_system_config

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

        model = load_model()
        detections = detect_food(processed, model)
        detections_df = detections_to_dataframe(detections)
        counts_df = count_food(detections_df)
        area_df, conversion_info = estimate_all_areas(detections_df)

        print("Detection Results:")
        print(detections_df)
        print("\nFood Counts:")
        print(counts_df)
        print("\nArea Conversion Info:")
        print(conversion_info)
        print("\nArea Estimation Results:")
        print(area_df)

from __future__ import annotations

import pandas as pd

from app.core.area_estimator import (
    REFERENCE_COIN_AREA_MM2,
    calc_pixel_to_mm,
    detect_reference_object,
    estimate_all_areas,
    estimate_food_area,
)


def test_detect_reference_object_returns_best_coin():
    detections_df = pd.DataFrame(
        [
            {"class_name": "reference_coin", "confidence": 0.81, "x1": 10, "y1": 10, "x2": 30, "y2": 30, "center_x": 20, "center_y": 20},
            {"class_name": "reference_coin", "confidence": 0.95, "x1": 40, "y1": 40, "x2": 60, "y2": 60, "center_x": 50, "center_y": 50},
            {"class_name": "apple", "confidence": 0.90, "x1": 70, "y1": 70, "x2": 120, "y2": 120, "center_x": 95, "center_y": 95},
        ]
    )

    reference = detect_reference_object(detections_df)

    assert reference is not None
    assert reference["confidence"] == 0.95
    assert reference["pixel_area"] == 400.0


def test_calc_pixel_to_mm_uses_default_without_reference():
    conversion = calc_pixel_to_mm(None)

    assert conversion["used_default"] is True
    assert conversion["mm2_per_pixel"] == 4.0


def test_calc_pixel_to_mm_with_reference_area():
    conversion = calc_pixel_to_mm(100.0)

    assert conversion["used_default"] is False
    assert round(conversion["mm2_per_pixel"], 4) == round(REFERENCE_COIN_AREA_MM2 / 100.0, 4)


def test_estimate_food_area_returns_expected_values():
    row = {
        "class_name": "apple",
        "confidence": 0.9,
        "x1": 0,
        "y1": 0,
        "x2": 10,
        "y2": 20,
        "center_x": 5,
        "center_y": 10,
    }

    result = estimate_food_area(row, mm2_per_pixel=2.0)

    assert result["pixel_area"] == 200.0
    assert result["estimated_area_mm2"] == 400.0


def test_estimate_all_areas_handles_no_reference_object():
    detections_df = pd.DataFrame(
        [
            {"class_name": "apple", "confidence": 0.91, "x1": 0, "y1": 0, "x2": 20, "y2": 20, "center_x": 10, "center_y": 10},
            {"class_name": "banana", "confidence": 0.88, "x1": 30, "y1": 30, "x2": 50, "y2": 50, "center_x": 40, "center_y": 40},
        ]
    )

    area_df, conversion = estimate_all_areas(detections_df)

    assert conversion["used_default"] is True
    assert len(area_df) == 2
    assert "portion_state" in area_df.columns


def test_estimate_all_areas_marks_partial_for_single_class():
    detections_df = pd.DataFrame(
        [
            {"class_name": "reference_coin", "confidence": 0.99, "x1": 0, "y1": 0, "x2": 20, "y2": 20, "center_x": 10, "center_y": 10},
            {"class_name": "onion", "confidence": 0.95, "x1": 50, "y1": 50, "x2": 100, "y2": 100, "center_x": 75, "center_y": 75},
            {"class_name": "onion", "confidence": 0.90, "x1": 120, "y1": 50, "x2": 145, "y2": 75, "center_x": 132.5, "center_y": 62.5},
        ]
    )

    area_df, conversion = estimate_all_areas(detections_df, full_ratio_threshold=0.5)

    assert conversion["used_default"] is False
    assert len(area_df) == 2
    assert set(area_df["class_name"]) == {"onion"}
    assert "完整食材" in set(area_df["portion_state"])
    assert "半个/部分食材" in set(area_df["portion_state"])

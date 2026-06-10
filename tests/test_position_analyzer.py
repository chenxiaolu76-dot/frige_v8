from __future__ import annotations

import pandas as pd

from app.core.position_analyzer import (
    analyze_all_positions,
    count_region_occupancy,
    get_food_position,
)


def test_get_food_position_top_left():
    result = get_food_position(center_x=10, center_y=10, image_width=300, image_height=300)

    assert result["vertical_zh"] == "上层"
    assert result["horizontal_zh"] == "左侧"
    assert result["position"] == "上层左侧"


def test_get_food_position_bottom_right_edge_case():
    result = get_food_position(center_x=299, center_y=299, image_width=300, image_height=300)

    assert result["vertical_zh"] == "下层"
    assert result["horizontal_zh"] == "右侧"
    assert result["position"] == "下层右侧"


def test_analyze_all_positions_handles_empty_dataframe():
    detections_df = pd.DataFrame(columns=["class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"])

    result_df = analyze_all_positions(detections_df, image_width=300, image_height=300)

    assert result_df.empty
    assert "position" in result_df.columns


def test_analyze_all_positions_assigns_positions():
    detections_df = pd.DataFrame(
        [
            {"class_name": "apple", "confidence": 0.91, "x1": 0, "y1": 0, "x2": 30, "y2": 30, "center_x": 15, "center_y": 15},
            {"class_name": "milk", "confidence": 0.89, "x1": 200, "y1": 120, "x2": 260, "y2": 240, "center_x": 230, "center_y": 180},
            {"class_name": "bread", "confidence": 0.87, "x1": 250, "y1": 250, "x2": 295, "y2": 295, "center_x": 272.5, "center_y": 272.5},
        ]
    )

    result_df = analyze_all_positions(detections_df, image_width=300, image_height=300)

    assert list(result_df["position"]) == ["上层左侧", "中层右侧", "下层右侧"]


def test_count_region_occupancy_counts_multiple_items_in_same_region():
    position_df = pd.DataFrame(
        [
            {"class_name": "apple", "position": "上层左侧"},
            {"class_name": "banana", "position": "上层左侧"},
            {"class_name": "milk", "position": "中层右侧"},
        ]
    )

    occupancy_df = count_region_occupancy(position_df)

    assert len(occupancy_df) == 2
    assert occupancy_df.iloc[0]["position"] == "上层左侧"
    assert occupancy_df.iloc[0]["count"] == 2

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION_CONFIG_PATH = PROJECT_ROOT / "config" / "fridge_regions.yaml"


def load_fridge_regions(config_path: str | Path = REGION_CONFIG_PATH) -> dict[str, Any]:
    """
    Load fridge region definitions from a YAML file.

    Input:
        config_path: Path to the fridge region config file.

    Output:
        Parsed fridge region configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Fridge region config not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _find_vertical_region(center_y: float, image_height: int, regions: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_y = center_y / image_height
    for region in regions:
        if region["y_min_ratio"] <= ratio_y < region["y_max_ratio"]:
            return region
    return regions[-1]


def _find_horizontal_region(center_x: float, image_width: int, regions: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_x = center_x / image_width
    for region in regions:
        if region["x_min_ratio"] <= ratio_x < region["x_max_ratio"]:
            return region
    return regions[-1]


def get_food_position(
    center_x: float,
    center_y: float,
    image_width: int,
    image_height: int,
    regions_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Map a detection center point to a fridge position label.

    Input:
        center_x: Detection center x coordinate in pixels.
        center_y: Detection center y coordinate in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        regions_config: Optional loaded fridge region config.

    Output:
        Position dictionary containing vertical, horizontal and combined labels.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive integers.")

    config = regions_config or load_fridge_regions()
    vertical_regions = config["regions"]["vertical"]
    horizontal_regions = config["regions"]["horizontal"]

    vertical_region = _find_vertical_region(center_y, image_height, vertical_regions)
    horizontal_region = _find_horizontal_region(center_x, image_width, horizontal_regions)

    return {
        "vertical": vertical_region["name"],
        "vertical_zh": vertical_region["label_zh"],
        "horizontal": horizontal_region["name"],
        "horizontal_zh": horizontal_region["label_zh"],
        "position": f'{vertical_region["label_zh"]}{horizontal_region["label_zh"]}',
    }


def analyze_all_positions(
    detections_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    config_path: str | Path = REGION_CONFIG_PATH,
) -> pd.DataFrame:
    """
    Analyze fridge positions for all detections.

    Input:
        detections_df: Detection results DataFrame.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        config_path: Path to the fridge region config file.

    Output:
        DataFrame containing detection information with mapped positions.
    """
    if detections_df.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "vertical_zh",
                "horizontal_zh",
                "position",
            ]
        )

    config = load_fridge_regions(config_path)
    analyzed_rows: list[dict[str, Any]] = []

    for _, row in detections_df.iterrows():
        position_info = get_food_position(
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            image_width=image_width,
            image_height=image_height,
            regions_config=config,
        )
        analyzed_row = row.to_dict()
        analyzed_row.update(position_info)
        analyzed_rows.append(analyzed_row)

    return pd.DataFrame(analyzed_rows)


def count_region_occupancy(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many detections fall into each fridge region.

    Input:
        position_df: Position analysis DataFrame.

    Output:
        DataFrame containing region occupancy statistics.
    """
    if position_df.empty:
        return pd.DataFrame(columns=["position", "count"])

    occupancy_df = (
        position_df.groupby("position")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "position"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return occupancy_df


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        [
            {"class_name": "apple", "confidence": 0.91, "x1": 10, "y1": 20, "x2": 110, "y2": 120, "center_x": 60, "center_y": 70},
            {"class_name": "milk", "confidence": 0.88, "x1": 250, "y1": 180, "x2": 320, "y2": 360, "center_x": 285, "center_y": 270},
        ]
    )
    positions = analyze_all_positions(sample_data, image_width=360, image_height=480)
    occupancy = count_region_occupancy(positions)
    print("Position Analysis:")
    print(positions[["class_name", "position"]])
    print("\nRegion Occupancy:")
    print(occupancy)

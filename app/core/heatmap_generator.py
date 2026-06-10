from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd


def generate_heatmap_overlay(
    image: np.ndarray,
    detections_df: pd.DataFrame,
    radius: int = 60,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Generate a heatmap overlay from detection center points.

    Input:
        image: Original or processed BGR image as a numpy array.
        detections_df: Detection or position DataFrame with center_x and center_y columns.
        radius: Circle radius used for each detection point.
        alpha: Overlay blend ratio.

    Output:
        Heatmap overlay image as a numpy array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if detections_df.empty:
        return image.copy()

    height, width = image.shape[:2]
    heatmap_canvas = np.zeros((height, width), dtype=np.float32)

    for _, row in detections_df.iterrows():
        center_x = int(round(float(row["center_x"])))
        center_y = int(round(float(row["center_y"])))
        cv2.circle(heatmap_canvas, (center_x, center_y), radius, 1.0, thickness=-1)

    heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (0, 0), sigmaX=25, sigmaY=25)
    heatmap_normalized = cv2.normalize(heatmap_canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def build_region_heatmap_summary(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build region-level occupancy summary for heatmap-related display.

    Input:
        position_df: Position analysis DataFrame.

    Output:
        Region summary DataFrame.
    """
    if position_df.empty:
        return pd.DataFrame(columns=["position", "count"])

    return (
        position_df.groupby("position")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "position"], ascending=[False, True])
        .reset_index(drop=True)
    )

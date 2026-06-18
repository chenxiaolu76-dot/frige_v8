from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  原有的原图叠加热力图（保留不动）
# ──────────────────────────────────────────────


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


# ──────────────────────────────────────────────
#  新增：独立热力图（热辐射风格，不叠原图）
# ──────────────────────────────────────────────


def generate_standalone_heatmap(
    detections_df: pd.DataFrame,
    image_shape: tuple[int, int],
    confidence_exponent: float = 1.5,
    blur_sigma: int = 35,
    base_radius_scale: float = 0.8,
    colormap: int = cv2.COLORMAP_INFERNO,
    draw_colorbar: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """
    生成一张独立的热辐射风格热力图（不叠加在原图上）。

    设计思路（思路二）：
    - 深色背景，模拟热成像仪效果
    - 每个检测物是一个"热源"，辐射半径与检测框尺寸成正比
    - 热源强度（亮度）与置信度成正比：高置信度 → 白/黄，低置信度 → 暗红
    - 多个热源重叠区域自动叠加升温，形成"热点"
    - 高斯模糊模拟热量扩散效果
    - 可选 colorbar 和类别标签

    Input:
        detections_df: 检测结果 DataFrame，需包含列：
            center_x, center_y, x1, y1, x2, y2, confidence, class_name
        image_shape: 输出图片尺寸 (height, width)。
        confidence_exponent: 置信度指数，>1 让高置信度更突出。
        blur_sigma: 高斯模糊 σ，控制热量扩散范围。
        base_radius_scale: 辐射半径相对于检测框最大边的比例。
        colormap: OpenCV 色图，推荐 COLORMAP_INFERNO 或 COLORMAP_MAGMA。
        draw_colorbar: 是否在右侧绘制强度色标。
        draw_labels: 是否在每个热点旁绘制类别名称。

    Output:
        独立热力图 BGR 图像 (numpy array)。
    """
    if detections_df is None or detections_df.empty:
        raise ValueError("Detections DataFrame is empty.")

    height, width = image_shape[:2]

    # ── 1. 热力累加画布（float32，黑底） ──
    heatmap_canvas = np.zeros((height, width), dtype=np.float32)

    # ── 2. 遍历每个检测物，绘制置信度加权椭圆热源 ──
    for _, row in detections_df.iterrows():
        cx = int(round(float(row["center_x"])))
        cy = int(round(float(row["center_y"])))
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        confidence = float(row["confidence"])

        # 检测框尺寸（防止过小）
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)

        # 辐射半径：基于检测框最大边
        radius = max(bw, bh) * base_radius_scale

        # 强度 = confidence ^ exponent（>1 让高置信度更亮）
        intensity = float(confidence ** confidence_exponent)
        intensity = min(max(intensity, 0.0), 1.0)

        # 椭圆轴：保持检测框宽高比
        major = max(bw, bh)
        axes = (
            max(int(radius * (bw / major)), 3),
            max(int(radius * (bh / major)), 3),
        )

        cv2.ellipse(heatmap_canvas, (cx, cy), axes, 0, 0, 360, intensity, thickness=-1)

    # ── 3. 高斯模糊 → 热量扩散 ──
    if blur_sigma > 0:
        heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # ── 4. 归一化 → 应用色图 ──
    heatmap_normalized = cv2.normalize(heatmap_canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, colormap)

    # ── 5. 绘制 colorbar（右侧） ──
    if draw_colorbar:
        heatmap_color = _draw_colorbar(heatmap_color, colormap=colormap)

    # ── 6. 绘制检测类别标签 ──
    if draw_labels:
        heatmap_color = _draw_detection_labels(heatmap_color, detections_df)

    # ── 7. 绘制等高线（强化热区边界） ──
    # 对归一化后的热力图找等高线，画在彩色图上
    # 只在热力值较高的区域绘制，避免画面太杂乱
    contour_level = int(255 * 0.35)  # 35% 以上强度才画等高线
    ret, thresh = cv2.threshold(heatmap_normalized, contour_level, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap_color, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    return heatmap_color


# ──────────────────────────────────────────────
#  内部辅助函数
# ──────────────────────────────────────────────


def _draw_colorbar(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    bar_width: int = 36,
    margin: int = 10,
) -> np.ndarray:
    """
    在图像右侧绘制竖条 colorbar。
    """
    h, w = image.shape[:2]
    bar_height = h - 2 * margin

    # 创建渐变条（从上到下 0→255）
    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(bar_height, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    gradient_color = cv2.applyColorMap(gradient, colormap)

    # 扩展画布
    new_w = w + bar_width + margin + 40
    canvas = np.zeros((h, new_w, 3), dtype=np.uint8)
    canvas[:h, :w] = image

    # 贴 colorbar
    x_offset = w + margin
    canvas[margin : margin + bar_height, x_offset : x_offset + bar_width] = gradient_color

    # 边框
    cv2.rectangle(
        canvas,
        (x_offset, margin),
        (x_offset + bar_width - 1, margin + bar_height - 1),
        (200, 200, 200),
        1,
    )

    # 标注文字
    label_x = x_offset + bar_width + 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    color = (200, 200, 200)
    thickness = 1

    cv2.putText(canvas, "高", (label_x, margin + 20), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(canvas, "↑", (label_x, margin + bar_height // 2), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(canvas, "低", (label_x, margin + bar_height - 4), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas


def _draw_detection_labels(
    image: np.ndarray,
    detections_df: pd.DataFrame,
) -> np.ndarray:
    """
    在每个检测物中心附近绘制类别名称和置信度。
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    for _, row in detections_df.iterrows():
        cx = int(round(float(row["center_x"])))
        cy = int(round(float(row["center_y"])))
        class_name = str(row.get("class_name", ""))
        confidence = float(row.get("confidence", 0))

        label = f"{class_name} {confidence:.2f}"

        # 文字背景尺寸
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 文字位置（热点上方）
        text_x = cx - text_w // 2
        text_y = cy - 20

        # 确保不超出边界
        text_x = max(2, min(text_x, image.shape[1] - text_w - 2))
        text_y = max(text_h + 4, text_y)

        # 半透明文字背景
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # 文字
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


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

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
#  新增（思路四）：按食材类别分区着色的热辐射图
# ──────────────────────────────────────────────

# 食材类别分组与对应 RGB 通道
CATEGORY_GROUPS: dict[str, dict[str, Any]] = {
    "fruits": {
        "items": ["apple", "banana", "orange"],
        "channel": 0,  # Red 通道
        "label": "水果",
        "legend_color": (60, 60, 255),  # BGR 红色
    },
    "vegetables": {
        "items": ["tomato", "cucumber", "potato", "onion", "carrot", "green_pepper"],
        "channel": 1,  # Green 通道
        "label": "蔬菜",
        "legend_color": (60, 255, 60),  # BGR 绿色
    },
    "dairy_other": {
        "items": ["egg", "milk", "bread"],
        "channel": 2,  # Blue 通道
        "label": "乳品/其他",
        "legend_color": (255, 200, 60),  # BGR 青色
    },
}

# 反向查找：class_name -> channel
_CLASS_TO_CHANNEL: dict[str, int] = {}
for _group in CATEGORY_GROUPS.values():
    for _item in _group["items"]:
        _CLASS_TO_CHANNEL[_item] = _group["channel"]


def generate_categorized_heatmap(
    detections_df: pd.DataFrame,
    image_shape: tuple[int, int],
    confidence_exponent: float = 1.5,
    blur_sigma: int = 30,
    base_radius_scale: float = 0.8,
    draw_colorbar: bool = True,
    draw_labels: bool = True,
    boost_low: float = 0.15,
) -> np.ndarray:
    """
    生成按食材类别分区着色的独立热辐射图（思路四）。

    设计思路：
    - 食材分为三大类，每类占用一个 RGB 颜色通道
      水果 🍎🍌🍊 → Red 通道（红/橙热源）
      蔬菜 🍅🥒🥔🧅🥕🫑 → Green 通道（绿热源）
      乳品/其他 🥚🥛🍞 → Blue 通道（蓝/青热源）
    - 每类独立绘制热源，独立高斯模糊
    - 三通道合成一张彩色热力图
    - 同一位置多类食材重叠会混合出复合色（如黄=红+绿，白=红+绿+蓝）

    Input:
        detections_df: DataFrame，需包含 center_x, center_y, x1, y1, x2, y2, confidence, class_name
        image_shape: (height, width)
        confidence_exponent: 置信度指数，>1 让高置信度更亮
        blur_sigma: 高斯模糊 σ
        base_radius_scale: 辐射半径相对于检测框最大边的比例
        draw_colorbar: 是否绘制图例和色标
        draw_labels: 是否绘制类别标签
        boost_low: 低强度通道的亮度补偿（0=不补偿，值越大弱信号越亮）

    Output:
        BGR 彩色热力图 (numpy array)
    """
    if detections_df is None or detections_df.empty:
        raise ValueError("Detections DataFrame is empty.")

    height, width = image_shape[:2]

    # ── 1. 三通道热力累加画布 ──
    heatmap_canvas = np.zeros((height, width, 3), dtype=np.float32)

    # ── 2. 遍历检测物，按类别写入对应通道 ──
    for _, row in detections_df.iterrows():
        cx = int(round(float(row["center_x"])))
        cy = int(round(float(row["center_y"])))
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        confidence = float(row["confidence"])
        class_name = str(row.get("class_name", "")).lower().strip()

        # 确定所属通道
        ch = _CLASS_TO_CHANNEL.get(class_name, 1)  # 未知默认绿色

        # 检测框尺寸
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)

        # 辐射半径
        radius = max(bw, bh) * base_radius_scale

        # 强度
        intensity = float(confidence ** confidence_exponent)
        intensity = min(max(intensity, 0.0), 1.0)

        # 椭圆轴（保持宽高比）
        major = max(bw, bh)
        axes = (
            max(int(radius * (bw / major)), 3),
            max(int(radius * (bh / major)), 3),
        )

        # 在对应通道画椭圆
        channel_data = heatmap_canvas[:, :, ch].copy()
        cv2.ellipse(channel_data, (cx, cy), axes, 0, 0, 360, intensity, thickness=-1)
        heatmap_canvas[:, :, ch] = channel_data

    # ── 3. 逐通道高斯模糊（热量扩散） ──
    if blur_sigma > 0:
        for ch in range(3):
            heatmap_canvas[:, :, ch] = cv2.GaussianBlur(
                heatmap_canvas[:, :, ch], (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma,
            )

    # ── 4. 归一化：逐通道独立拉伸到 [0, 1] ──
    for ch in range(3):
        channel = heatmap_canvas[:, :, ch]
        mx = channel.max()
        if mx > 0:
            # 独立拉伸，确保弱信号也可见
            channel = channel / mx
            # 弱信号补偿：提升低亮度区域
            if boost_low > 0:
                mask = channel < boost_low
                channel[mask] = channel[mask] * (1.0 / boost_low) * boost_low
                channel = np.clip(channel, 0, 1)
            heatmap_canvas[:, :, ch] = channel

    # ── 5. 转 8-bit ──
    heatmap_color = (heatmap_canvas * 255).astype(np.uint8)

    # ── 6. 绘制图例和色标 ──
    if draw_colorbar:
        heatmap_color = _draw_categorized_legend(heatmap_color)

    # ── 7. 绘制检测标签 ──
    if draw_labels:
        heatmap_color = _draw_detection_labels(heatmap_color, detections_df)

    # ── 8. 白色等高线（强化热区边界） ──
    gray_for_contour = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_for_contour, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap_color, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    return heatmap_color


# ──────────────────────────────────────────────
#  后续兼容：旧的独立热力图改名保留
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
    （旧版思路二）生成单色热辐射图，保留以兼容已有调用。
    新代码请使用 generate_categorized_heatmap()。
    """
    if detections_df is None or detections_df.empty:
        raise ValueError("Detections DataFrame is empty.")

    height, width = image_shape[:2]
    heatmap_canvas = np.zeros((height, width), dtype=np.float32)

    for _, row in detections_df.iterrows():
        cx = int(round(float(row["center_x"])))
        cy = int(round(float(row["center_y"])))
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        confidence = float(row["confidence"])

        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)
        radius = max(bw, bh) * base_radius_scale
        intensity = float(confidence ** confidence_exponent)
        intensity = min(max(intensity, 0.0), 1.0)

        major = max(bw, bh)
        axes = (
            max(int(radius * (bw / major)), 3),
            max(int(radius * (bh / major)), 3),
        )
        cv2.ellipse(heatmap_canvas, (cx, cy), axes, 0, 0, 360, intensity, thickness=-1)

    if blur_sigma > 0:
        heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    heatmap_normalized = cv2.normalize(heatmap_canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, colormap)

    if draw_colorbar:
        heatmap_color = _draw_colorbar(heatmap_color, colormap=colormap)
    if draw_labels:
        heatmap_color = _draw_detection_labels(heatmap_color, detections_df)

    contour_level = int(255 * 0.35)
    _, thresh = cv2.threshold(heatmap_normalized, contour_level, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap_color, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    return heatmap_color


# ──────────────────────────────────────────────
#  图例绘制：按类别分组的彩色图例
# ──────────────────────────────────────────────


def _draw_categorized_legend(
    image: np.ndarray,
    bar_width: int = 30,
    margin: int = 16,
) -> np.ndarray:
    """
    在图像右侧绘制三通道分类图例。
    每个类别显示对应通道的渐变条 + 颜色标识 + 类别名称。
    """
    h, w = image.shape[:2]
    groups = list(CATEGORY_GROUPS.values())

    # 每个类别占用高度
    group_h = (h - 2 * margin) // len(groups)

    new_w = w + bar_width + margin + 60
    canvas = np.zeros((h, new_w, 3), dtype=np.uint8)
    canvas[:h, :w] = image

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, group in enumerate(groups):
        ch = group["channel"]
        label = group["label"]
        legend_color = group["legend_color"]

        # 垂直位置
        y_start = margin + i * group_h
        y_end = y_start + group_h - 4

        # 创建该通道的渐变条
        gradient = np.linspace(255, 0, y_end - y_start, dtype=np.uint8).reshape(y_end - y_start, 1)
        gradient = np.repeat(gradient, bar_width, axis=1)

        # 转为 3 通道，只填充对应通道
        bar_rgb = np.zeros((y_end - y_start, bar_width, 3), dtype=np.uint8)
        bar_rgb[:, :, ch] = gradient

        # 贴到画布
        x_offset = w + margin
        canvas[y_start:y_end, x_offset : x_offset + bar_width] = bar_rgb

        # 边框
        cv2.rectangle(
            canvas,
            (x_offset, y_start),
            (x_offset + bar_width - 1, y_end - 1),
            (100, 100, 100),
            1,
        )

        # 颜色标识圆点 + 类别名称
        dot_x = x_offset + bar_width + 8
        dot_y = y_start + (group_h - 4) // 2

        cv2.circle(canvas, (dot_x, dot_y), 5, legend_color, -1)
        cv2.putText(
            canvas,
            label,
            (dot_x + 10, dot_y + 4),
            font,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ──────────────────────────────────────────────
#  检测标签绘制
# ──────────────────────────────────────────────


def _draw_detection_labels(
    image: np.ndarray,
    detections_df: pd.DataFrame,
) -> np.ndarray:
    """
    在每个检测物中心附近绘制类别名称和置信度。
    标签颜色与食材类别分组颜色一致。
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

        # 根据类别选择标签颜色
        ch = _CLASS_TO_CHANNEL.get(class_name.lower().strip(), 1)
        if ch == 0:
            label_color = (60, 60, 255)      # 红
        elif ch == 1:
            label_color = (60, 255, 60)      # 绿
        else:
            label_color = (255, 200, 60)     # 蓝/青

        label = f"{class_name} {confidence:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = cx - text_w // 2
        text_y = cy - 20
        text_x = max(2, min(text_x, image.shape[1] - text_w - 2))
        text_y = max(text_h + 4, text_y)

        # 半透明黑色背景
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # 彩色文字
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            font_scale,
            label_color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# ──────────────────────────────────────────────
#  旧的 colorbar（保留给 generate_standalone_heatmap）
# ──────────────────────────────────────────────


def _draw_colorbar(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    bar_width: int = 36,
    margin: int = 10,
) -> np.ndarray:
    h, w = image.shape[:2]
    bar_height = h - 2 * margin

    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(bar_height, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    gradient_color = cv2.applyColorMap(gradient, colormap)

    new_w = w + bar_width + margin + 40
    canvas = np.zeros((h, new_w, 3), dtype=np.uint8)
    canvas[:h, :w] = image

    x_offset = w + margin
    canvas[margin : margin + bar_height, x_offset : x_offset + bar_width] = gradient_color

    cv2.rectangle(
        canvas,
        (x_offset, margin),
        (x_offset + bar_width - 1, margin + bar_height - 1),
        (200, 200, 200),
        1,
    )

    label_x = x_offset + bar_width + 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "高", (label_x, margin + 20), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "↑", (label_x, margin + bar_height // 2), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "低", (label_x, margin + bar_height - 4), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def build_region_heatmap_summary(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build region-level occupancy summary for heatmap-related display.
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

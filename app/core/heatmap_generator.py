from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  冰箱框架布局图（食材名称 + 数量标注）
# ──────────────────────────────────────────────


def generate_fridge_layout_chart(
    detections_df: pd.DataFrame,
    image_shape: tuple[int, int] = (800, 1000),
    shelf_ratios: tuple[float, float] = (0.33, 0.66),
) -> np.ndarray:
    """
    生成冰箱内部框架布局图，在每个食材类别对应的位置标注名称和数量。

    设计思路：
    - 绘制冰箱框架轮廓和搁板分隔线，模拟冰箱内部俯视布局
    - 同类食材的所有检测框外包一个区域边框（四角带装饰线）
    - 每个检测框用类别颜色的半透明填充，形成"热区"
    - 在区域中心标注 "食材中文名 x N"
    - 数量多的用偏红色，数量少的用偏绿色（绿→黄→橙→红渐变）

    Input:
        detections_df: DataFrame，需包含 center_x, center_y, x1, y1, x2, y2, class_name
        image_shape: 输出图片尺寸 (height, width)
        shelf_ratios: 搁板位置比例

    Output:
        BGR 图像
    """
    if detections_df is None or detections_df.empty:
        raise ValueError("Detections DataFrame is empty.")

    height, width = image_shape[:2]
    margin = 40

    fx1, fy1 = margin, margin
    fx2, fy2 = width - margin, height - margin
    fw = fx2 - fx1
    fh = fy2 - fy1

    # ── 1. 创建画布（深色背景） ──
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 35, 45)

    # ── 2. 绘制冰箱外框 ──
    cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), (60, 70, 85), 3, cv2.LINE_AA)
    inner = canvas.copy()
    cv2.rectangle(inner, (fx1, fy1), (fx2, fy2), (45, 50, 60), -1)
    cv2.addWeighted(inner, 0.6, canvas, 0.4, 0, canvas)

    # ── 3. 绘制搁板分隔线 ──
    shelf_color = (80, 95, 115)
    shelf_thickness = 6
    shelves_y = []
    for ratio in shelf_ratios:
        sy = fy1 + int(fh * ratio)
        shelves_y.append(sy)
        cv2.line(canvas, (fx1 + 10, sy), (fx2 - 10, sy), shelf_color, shelf_thickness, cv2.LINE_AA)
        cv2.line(canvas, (fx1 + 10, sy + 1), (fx2 - 10, sy + 1), (100, 115, 135), 2, cv2.LINE_AA)

    # ── 4. 区域分隔虚线（左/中/右） ──
    for col_ratio in [0.33, 0.66]:
        sx = fx1 + int(fw * col_ratio)
        cv2.line(canvas, (sx, fy1 + 5), (sx, fy2 - 5), (55, 65, 80), 1, cv2.LINE_AA)

    # ── 5. 区域标签（上/中/下） ──
    font_small = cv2.FONT_HERSHEY_SIMPLEX
    zone_labels = ["上", "中", "下"]
    for zlabel, sy in zip(zone_labels, [fy1 + 20, shelves_y[0] + 20, shelves_y[1] + 20]):
        cv2.putText(canvas, zlabel, (fx1 + 8, sy), font_small, 0.5, (70, 80, 100), 1, cv2.LINE_AA)

    # ── 6. 加载中文名映射 ──
    import yaml
    from pathlib import Path
    try:
        config_path = Path(__file__).resolve().parents[1] / "config" / "classes.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            classes_config = yaml.safe_load(f)
        name_map = {item["name"]: item["label_zh"] for item in classes_config.get("classes", [])}
    except Exception:
        name_map = {}

    # ── 7. 按类别分组，计算区域边界 ──
    has_boxes = all(col in detections_df.columns for col in ["x1", "y1", "x2", "y2"])

    class_data = {}
    for _, row in detections_df.iterrows():
        cn = str(row["class_name"]).strip().lower()
        if cn not in class_data:
            class_data[cn] = {"count": 0, "boxes": []}
        class_data[cn]["count"] += 1
        if has_boxes:
            class_data[cn]["boxes"].append((
                int(round(float(row["x1"]))),
                int(round(float(row["y1"]))),
                int(round(float(row["x2"]))),
                int(round(float(row["y2"]))),
            ))

    if not class_data:
        return canvas

    counts = [d["count"] for d in class_data.values()]
    min_count = min(counts)
    max_count = max(counts)
    count_range = max(max_count - min_count, 1)

    font_large = cv2.FONT_HERSHEY_SIMPLEX

    def _count_to_color(cnt: int) -> tuple:
        """数量 → BGR 颜色：绿 → 黄 → 橙 → 红"""
        ratio = (cnt - min_count) / count_range
        if ratio < 0.33:
            r = int(120 + 135 * (ratio / 0.33))
            g = 220
            b = int(80 - 60 * (ratio / 0.33))
        elif ratio < 0.66:
            r = 255
            g = int(220 - 120 * ((ratio - 0.33) / 0.33))
            b = int(20 - 20 * ((ratio - 0.33) / 0.33))
        else:
            r = 255
            g = int(100 - 80 * ((ratio - 0.66) / 0.34))
            b = 0
        return (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))

    # ── 8. 画区域边框 ──
    region_centers = {}

    for cn, data in class_data.items():
        cnt = data["count"]
        boxes = data["boxes"]
        color = _count_to_color(cnt)
        alpha_color = (color[0] // 2, color[1] // 2, color[2] // 2)

        if has_boxes and boxes:
            xs1 = [b[0] for b in boxes]
            ys1 = [b[1] for b in boxes]
            xs2 = [b[2] for b in boxes]
            ys2 = [b[3] for b in boxes]

            region_x1 = max(min(xs1) - 12, fx1 + 2)
            region_y1 = max(min(ys1) - 12, fy1 + 2)
            region_x2 = min(max(xs2) + 12, fx2 - 2)
            region_y2 = min(max(ys2) + 12, fy2 - 2)

            region_cx = (region_x1 + region_x2) // 2
            region_cy = (region_y1 + region_y2) // 2
            region_centers[cn] = (region_cx, region_cy)

            # 每个检测框半透明填充
            overlay = canvas.copy()
            for bx1, by1, bx2, by2 in boxes:
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), alpha_color, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

            # 整体区域边框
            cv2.rectangle(canvas, (region_x1, region_y1), (region_x2, region_y2), color, 2, cv2.LINE_AA)

            # 四角装饰线
            cl = 12
            cv2.line(canvas, (region_x1, region_y1 + cl), (region_x1, region_y1), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x1, region_y1), (region_x1 + cl, region_y1), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x2 - cl, region_y1), (region_x2, region_y1), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x2, region_y1), (region_x2, region_y1 + cl), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x1, region_y2 - cl), (region_x1, region_y2), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x1, region_y2), (region_x1 + cl, region_y2), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x2 - cl, region_y2), (region_x2, region_y2), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (region_x2, region_y2), (region_x2, region_y2 - cl), color, 2, cv2.LINE_AA)
        else:
            xs = [float(row["center_x"]) for _, row in detections_df[detections_df["class_name"].str.strip().str.lower() == cn].iterrows()]
            ys = [float(row["center_y"]) for _, row in detections_df[detections_df["class_name"].str.strip().str.lower() == cn].iterrows()]
            region_centers[cn] = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    # ── 9. 在区域中心标注文字 ──
    for cn, data in class_data.items():
        cnt = data["count"]
        cx, cy = region_centers.get(cn, (fx1 + fw // 2, fy1 + fh // 2))
        color = _count_to_color(cnt)

        label_zh = name_map.get(cn, cn.capitalize())
        text = f"{label_zh} x{cnt}"

        (text_w, text_h), _ = cv2.getTextSize(text, font_large, 0.75, 2)

        lx = max(fx1 + 5, min(cx - text_w // 2, fx2 - text_w - 5))
        ly = max(fy1 + text_h + 12, min(cy, fy2 - 8))

        overlay = canvas.copy()
        pad = 10
        cv2.rectangle(overlay, (lx - pad, ly - text_h - pad), (lx + text_w + pad, ly + pad), (20, 25, 35), -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        cv2.rectangle(canvas, (lx - pad, ly - text_h - pad), (lx + text_w + pad, ly + pad), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (lx, ly), font_large, 0.75, color, 2, cv2.LINE_AA)

    # ── 10. 底部信息 ──
    total_items = len(detections_df)
    info_text = f"食材分布  |  共 {total_items} 件  |  颜色越红 = 数量越多"
    (info_w, _), _ = cv2.getTextSize(info_text, font_small, 0.45, 1)
    cv2.putText(canvas, info_text, ((width - info_w) // 2, height - 12), font_small, 0.45, (100, 110, 130), 1, cv2.LINE_AA)

    return canvas

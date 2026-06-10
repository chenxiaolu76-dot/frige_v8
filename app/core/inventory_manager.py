from __future__ import annotations

import pandas as pd

from app.utils.config_loader import load_classes_config


REFERENCE_CLASS_NAME = "reference_coin"
DEFAULT_RESTOCK_TARGETS = {
    "apple": 2,
    "banana": 2,
    "orange": 2,
    "tomato": 2,
    "cucumber": 1,
    "potato": 2,
    "onion": 2,
    "carrot": 2,
    "green_pepper": 1,
    "egg": 4,
    "milk": 1,
    "bread": 1,
}


def build_food_summary(detections_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a food inventory summary with count and total estimated area.

    Input:
        detections_df: Detection DataFrame from detector.py.
        area_df: Area estimation DataFrame from area_estimator.py.

    Output:
        Inventory summary DataFrame.
    """
    if detections_df.empty:
        return pd.DataFrame(columns=["class_name", "count", "total_area_mm2"])

    filtered_df = detections_df.copy()
    if REFERENCE_CLASS_NAME in filtered_df["class_name"].values:
        filtered_df = filtered_df[filtered_df["class_name"] != REFERENCE_CLASS_NAME].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=["class_name", "count", "total_area_mm2"])

    count_df = (
        filtered_df.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "class_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if area_df.empty:
        count_df["total_area_mm2"] = 0.0
        return count_df

    area_summary_df = (
        area_df.groupby("class_name")["estimated_area_mm2"]
        .sum()
        .reset_index(name="total_area_mm2")
    )

    summary_df = count_df.merge(area_summary_df, on="class_name", how="left")
    summary_df["total_area_mm2"] = summary_df["total_area_mm2"].fillna(0).round(2)
    return summary_df


def build_position_area_summary(position_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge position analysis with area estimation results.

    Input:
        position_df: Position analysis DataFrame.
        area_df: Area estimation DataFrame.

    Output:
        Position and area merged DataFrame.
    """
    if position_df.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "position",
                "estimated_area_mm2",
                "portion_state",
                "confidence",
                "center_x",
                "center_y",
            ]
        )

    if area_df.empty:
        merged_df = position_df.copy()
        merged_df["estimated_area_mm2"] = 0.0
        merged_df["portion_state"] = "未知"
        return merged_df

    merged_df = position_df.merge(
        area_df[["class_name", "center_x", "center_y", "estimated_area_mm2", "portion_state"]],
        on=["class_name", "center_x", "center_y"],
        how="left",
    )
    merged_df["estimated_area_mm2"] = merged_df["estimated_area_mm2"].fillna(0).round(2)
    merged_df["portion_state"] = merged_df["portion_state"].fillna("未知")
    return merged_df


def build_restock_suggestions(
    food_summary_df: pd.DataFrame,
    target_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Build replenishment suggestions by comparing detected counts with target stock.

    Input:
        food_summary_df: Food inventory summary DataFrame.
        target_counts: Optional per-class target counts.

    Output:
        DataFrame containing current count, target count and suggestion text.
    """
    classes_config = load_classes_config().get("classes", [])
    class_rows = [
        {
            "class_name": item["name"],
            "label_zh": item.get("label_zh", item["name"]),
        }
        for item in classes_config
        if item.get("name") != REFERENCE_CLASS_NAME
    ]
    class_df = pd.DataFrame(class_rows)

    if class_df.empty:
        return pd.DataFrame(columns=["class_name", "label_zh", "count", "target_count", "gap", "priority", "suggestion"])

    merged_df = class_df.merge(
        food_summary_df[["class_name", "count"]] if not food_summary_df.empty else pd.DataFrame(columns=["class_name", "count"]),
        on="class_name",
        how="left",
    )
    merged_df["count"] = merged_df["count"].fillna(0).astype(int)

    targets = target_counts or DEFAULT_RESTOCK_TARGETS
    merged_df["target_count"] = merged_df["class_name"].map(targets).fillna(1).astype(int)
    merged_df["gap"] = merged_df["target_count"] - merged_df["count"]

    suggestion_rows: list[dict[str, object]] = []
    for _, row in merged_df.iterrows():
        gap = int(row["gap"])
        if gap <= 0:
            continue

        priority = "高" if row["count"] == 0 else "中"
        suggestion_rows.append(
            {
                "class_name": row["class_name"],
                "label_zh": row["label_zh"],
                "count": int(row["count"]),
                "target_count": int(row["target_count"]),
                "gap": gap,
                "priority": priority,
                "suggestion": f'当前仅有 {int(row["count"])} 个，建议补充 {gap} 个左右。',
            }
        )

    if not suggestion_rows:
        return pd.DataFrame(columns=["class_name", "label_zh", "count", "target_count", "gap", "priority", "suggestion"])

    return (
        pd.DataFrame(suggestion_rows)
        .sort_values(by=["priority", "gap", "label_zh"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_space_advice(
    occupancy_df: pd.DataFrame,
    position_area_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build space-management suggestions from region occupancy and area distribution.

    Input:
        occupancy_df: Region occupancy count DataFrame.
        position_area_df: Position-area merged DataFrame.

    Output:
        DataFrame containing issue type, target region and actionable suggestion.
    """
    if occupancy_df.empty or position_area_df.empty:
        return pd.DataFrame(columns=["issue_type", "position", "severity", "suggestion"])

    advice_rows: list[dict[str, str]] = []
    busiest_row = occupancy_df.sort_values(by=["count", "position"], ascending=[False, True]).iloc[0]
    max_count = int(busiest_row["count"])
    if max_count >= 3:
        advice_rows.append(
            {
                "issue_type": "区域拥挤",
                "position": str(busiest_row["position"]),
                "severity": "高" if max_count >= 5 else "中",
                "suggestion": f'当前 {busiest_row["position"]} 检测到 {max_count} 个食材，建议把体积较大的食材分散到其他区域。',
            }
        )

    area_by_position = (
        position_area_df.groupby("position")["estimated_area_mm2"]
        .sum()
        .reset_index(name="total_area_mm2")
        .sort_values(by=["total_area_mm2", "position"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if not area_by_position.empty:
        top_area_row = area_by_position.iloc[0]
        total_area = float(area_by_position["total_area_mm2"].sum())
        top_ratio = 0.0 if total_area <= 0 else float(top_area_row["total_area_mm2"]) / total_area
        if top_ratio >= 0.4:
            advice_rows.append(
                {
                    "issue_type": "面积集中",
                    "position": str(top_area_row["position"]),
                    "severity": "中",
                    "suggestion": f'{top_area_row["position"]} 约占总可见食材面积的 {top_ratio * 100:.1f}%，建议优先整理该区域，提升取放效率。',
                }
            )

    if occupancy_df.shape[0] >= 2:
        emptier_row = occupancy_df.sort_values(by=["count", "position"], ascending=[True, True]).iloc[0]
        if int(emptier_row["count"]) <= 1:
            advice_rows.append(
                {
                    "issue_type": "可利用空位",
                    "position": str(emptier_row["position"]),
                    "severity": "低",
                    "suggestion": f'{emptier_row["position"]} 当前较空，可作为后续补货或转移拥挤食材的优先区域。',
                }
            )

    if not advice_rows:
        advice_rows.append(
            {
                "issue_type": "布局稳定",
                "position": "整体",
                "severity": "低",
                "suggestion": "当前各区域分布相对均衡，暂无明显拥挤位置。",
            }
        )

    severity_order = {"高": 0, "中": 1, "低": 2}
    advice_df = pd.DataFrame(advice_rows)
    advice_df["severity_rank"] = advice_df["severity"].map(severity_order).fillna(99)
    advice_df = advice_df.sort_values(by=["severity_rank", "position"]).drop(columns=["severity_rank"]).reset_index(drop=True)
    return advice_df

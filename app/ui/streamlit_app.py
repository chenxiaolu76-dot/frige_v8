from __future__ import annotations

import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.detector import count_food, detect_food, detections_to_dataframe, load_model_from_weights  # noqa: E402
from app.core.area_estimator import estimate_all_areas  # noqa: E402
from app.core.heatmap_generator import generate_heatmap_overlay, generate_standalone_heatmap  # noqa: E402
from app.core.image_preprocess import (  # noqa: E402
    convert_gray,
    edge_detection,
    enhance_contrast,
    load_image,
    morphological_process,
    preprocess_pipeline,
    remove_noise,
)
from app.core.inventory_manager import (  # noqa: E402
    build_food_summary,
    build_position_area_summary,
    build_restock_suggestions,
    build_space_advice,
)
from app.core.position_analyzer import analyze_all_positions, count_region_occupancy  # noqa: E402
from app.utils.image_io import bgr_to_rgb, load_image_from_upload, resize_image_keep_ratio  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.ui.components import (  # noqa: E402
    close_panel,
    inject_global_styles,
    open_panel,
    render_hero,
    render_metric_card,
    render_note,
    render_section_header,
)


MAX_DISPLAY_WIDTH = 800
LOGGER = get_logger(__name__)
MODEL_OPTIONS = {
    "当前 12 类新模型": "model/fridge_12class_best.pt",
    "之前的生鲜阶段模型": "model/fresh_stage1_best.pt",
}


@st.cache_resource
def get_detector_model(weights_path: str):
    """Load the selected YOLO model once for the Streamlit app."""
    return load_model_from_weights(weights_path)


def encode_image_to_png_bytes(image):
    """Encode an image to PNG bytes for download."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image for download.")
    return buffer.tobytes()


def draw_detection_annotations(image: np.ndarray, area_df: pd.DataFrame) -> np.ndarray:
    """Draw detection boxes, labels and estimated area on an image."""
    annotated = image.copy()
    for _, row in area_df.iterrows():
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        label = f'{row["class_name"]} | {row["estimated_area_mm2"]:.1f} mm2'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def draw_raw_detections(image: np.ndarray, detections_df: pd.DataFrame) -> np.ndarray:
    """Draw raw detection boxes and confidence scores on an image."""
    annotated = image.copy()
    if detections_df.empty:
        return annotated

    for _, row in detections_df.iterrows():
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        label = f'{row["class_name"]} | {row["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 165, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def main():
    st.set_page_config(
        page_title="SmartFridge",
        page_icon="⟡",
        layout="wide"
    )
    inject_global_styles()
    render_hero()

    st.sidebar.header("项目说明")
    st.sidebar.write(
        "上传冰箱图片后，系统会依次完成预处理、食材检测、面积估算、位置分析和热力图生成。"
    )

    st.sidebar.header("预处理参数")
    use_gray = st.sidebar.checkbox("启用灰度化", value=False)
    use_contrast = st.sidebar.checkbox("启用对比度增强", value=True)
    clip_limit = st.sidebar.slider("CLAHE 对比度系数", 1.0, 5.0, 2.0, 0.1)
    use_denoise = st.sidebar.checkbox("启用去噪", value=True)
    gaussian_kernel_size = st.sidebar.select_slider("高斯滤波核大小", options=[3, 5, 7, 9], value=5)
    median_kernel_size = st.sidebar.select_slider("中值滤波核大小", options=[3, 5, 7, 9], value=5)
    use_edge = st.sidebar.checkbox("启用边缘检测", value=False)
    canny_low = st.sidebar.slider("Canny 低阈值", 0, 255, 50, 5)
    canny_high = st.sidebar.slider("Canny 高阈值", 0, 255, 150, 5)
    use_morphology = st.sidebar.checkbox("启用形态学处理", value=False)
    morph_kernel_size = st.sidebar.select_slider("形态学核大小", options=[3, 5, 7], value=3)
    run_preprocess = st.sidebar.checkbox("检测前启用预处理", value=True)

    st.sidebar.header("检测参数")
    selected_model_label = st.sidebar.selectbox("检测模型", options=list(MODEL_OPTIONS.keys()), index=0)
    st.sidebar.caption("如果整张冰箱图漏检严重，可先切到“之前的生鲜阶段模型”做对比测试。")
    use_raw_image_for_detection = st.sidebar.checkbox("检测时直接使用原图", value=True)
    detection_confidence = st.sidebar.slider("检测置信度阈值", 0.05, 0.50, 0.15, 0.01)
    detection_iou = st.sidebar.slider("NMS IoU 阈值", 0.10, 0.80, 0.45, 0.01)

    st.sidebar.header("结果展示")
    show_gray = st.sidebar.checkbox("显示灰度图", value=True)
    show_denoised = st.sidebar.checkbox("显示去噪图", value=True)
    show_edges = st.sidebar.checkbox("显示边缘图", value=True)
    show_morphology = st.sidebar.checkbox("显示形态学结果", value=True)

    uploaded_file = st.file_uploader("上传冰箱图片", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is None:
        render_section_header(
            "开始一次冰箱分析",
            "建议优先使用一张清晰、光照稳定的冰箱内部照片。",
            badge="等待输入",
            tone="neutral",
        )
        render_note("左侧面板可调整预处理参数。上传图片后，页面会自动展示检测、面积估算、位置分析和热力图。")
        return

    try:
        original_image = load_image_from_upload(uploaded_file)
        LOGGER.info("Image uploaded successfully: %s", uploaded_file.name)
    except Exception as error:
        LOGGER.exception("Failed to load uploaded image.")
        st.error(f"图片读取失败：{error}")
        return

    working_image = original_image.copy()
    gray_image = convert_gray(working_image) if use_gray else None

    if use_gray:
        working_image = gray_image.copy()

    if use_contrast:
        working_image = enhance_contrast(working_image, clip_limit=clip_limit)

    denoised_image = None
    if use_denoise:
        denoised_image = remove_noise(
            working_image,
            gaussian_kernel=(gaussian_kernel_size, gaussian_kernel_size),
            median_kernel=median_kernel_size,
        )
        working_image = denoised_image.copy()

    edge_image = None
    if use_edge:
        edge_image = edge_detection(working_image, low_threshold=canny_low, high_threshold=canny_high)
        working_image = edge_image.copy()

    morphology_image = None
    if use_morphology:
        morphology_image = morphological_process(
            working_image,
            kernel_size=(morph_kernel_size, morph_kernel_size),
        )
        working_image = morphology_image.copy()

    if run_preprocess:
        pipeline_results = preprocess_pipeline(
            original_image,
            use_gray=use_gray,
            use_contrast=use_contrast,
            use_denoise=use_denoise,
            use_edge=use_edge,
            use_morphology=use_morphology,
        )
        final_image = working_image
        LOGGER.info("Preprocessing pipeline executed.")
    else:
        pipeline_results = {"original": original_image, "final": original_image}
        final_image = original_image
        LOGGER.info("Preprocessing skipped.")

    try:
        selected_model_path = MODEL_OPTIONS[selected_model_label]
        model = get_detector_model(selected_model_path)
        LOGGER.info("YOLO model loaded successfully.")
    except FileNotFoundError as error:
        LOGGER.warning("YOLO weights not found: %s", error)
        st.warning(f"未找到 YOLO 权重文件，暂时无法执行检测：{error}")
        return
    except Exception as error:
        LOGGER.exception("Failed to load YOLO model.")
        st.error(f"加载检测模型失败：{error}")
        return

    try:
        detection_source = original_image if use_raw_image_for_detection else final_image
        detections = detect_food(
            detection_source,
            model,
            confidence_threshold=detection_confidence,
            iou_threshold=detection_iou,
        )
        detections_df = detections_to_dataframe(detections)
        LOGGER.info("Detection finished. Total detections: %s", len(detections_df))
    except Exception as error:
        LOGGER.exception("Detection failed.")
        st.error(f"检测失败：{error}")
        return

    if detections_df.empty:
        LOGGER.info("No food detections found.")
        st.info("当前图片未检测到食材目标。")
        return

    image_height, image_width = final_image.shape[:2]
    area_df, conversion_info = estimate_all_areas(detections_df)
    position_df = analyze_all_positions(detections_df, image_width=image_width, image_height=image_height)
    occupancy_df = count_region_occupancy(position_df)
    food_summary_df = build_food_summary(detections_df, area_df)
    position_area_df = build_position_area_summary(position_df, area_df)
    restock_df = build_restock_suggestions(food_summary_df)
    space_advice_df = build_space_advice(occupancy_df, position_area_df)
    annotated_image = draw_detection_annotations(final_image, area_df) if not area_df.empty else final_image.copy()
    raw_detection_image = draw_raw_detections(detection_source, detections_df)
    heatmap_image = generate_heatmap_overlay(final_image, position_df)
    standalone_heatmap = generate_standalone_heatmap(
        position_df,
        image_shape=final_image.shape[:2],
        confidence_exponent=1.5,
        blur_sigma=35,
        draw_colorbar=True,
        draw_labels=True,
    )

    LOGGER.info("Area estimation and position analysis completed.")

    detection_count = len(food_summary_df)
    total_items = int(food_summary_df["count"].sum()) if not food_summary_df.empty else 0
    total_area = float(food_summary_df["total_area_mm2"].sum()) if not food_summary_df.empty else 0.0
    restock_count = len(restock_df)

    render_section_header(
        "分析总览",
        "先看整体识别结果，再进入图像、统计与空间布局细节。",
        badge="分析完成",
        tone="success",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("识别类别数", str(detection_count), "当前图片中被识别到的食材类别数量。")
    with metric_cols[1]:
        render_metric_card("食材总数", str(total_items), "按检测结果统计的食材个数。")
    with metric_cols[2]:
        render_metric_card("估算总面积", f"{total_area:.0f}", "所有食材估算面积总和，单位为 mm²。")
    with metric_cols[3]:
        render_metric_card("补货提示数", str(restock_count), "当前建议优先补充的食材类别数量。")

    tabs = st.tabs(["图像总览", "统计分析", "空间布局", "管理建议", "调试", "结果导出"])

    with tabs[0]:
        render_section_header(
            "图像处理链路",
            "对照原图、预处理结果和最终检测标注图，适合答辩时逐步讲解。"
        )
        open_panel()
        image_cols = st.columns(3)
        with image_cols[0]:
            st.markdown("原始图片")
            st.image(
                bgr_to_rgb(resize_image_keep_ratio(original_image, max_width=MAX_DISPLAY_WIDTH)),
                use_column_width=True,
            )
        with image_cols[1]:
            st.markdown("预处理结果")
            st.image(
                bgr_to_rgb(resize_image_keep_ratio(final_image, max_width=MAX_DISPLAY_WIDTH)),
                use_column_width=True,
                clamp=True,
            )
        with image_cols[2]:
            st.markdown("检测标注图")
            st.image(
                bgr_to_rgb(resize_image_keep_ratio(annotated_image, max_width=MAX_DISPLAY_WIDTH)),
                use_column_width=True,
            )
        close_panel()

        with st.expander("预处理细节图层", expanded=False):
            result_columns = st.columns(4)
            if show_gray and gray_image is not None:
                with result_columns[0]:
                    st.markdown("灰度图")
                    st.image(resize_image_keep_ratio(gray_image, max_width=MAX_DISPLAY_WIDTH), use_column_width=True, clamp=True)
            if show_denoised and denoised_image is not None:
                with result_columns[1]:
                    st.markdown("去噪图")
                    st.image(bgr_to_rgb(resize_image_keep_ratio(denoised_image, max_width=MAX_DISPLAY_WIDTH)), use_column_width=True)
            if show_edges and edge_image is not None:
                with result_columns[2]:
                    st.markdown("边缘图")
                    st.image(resize_image_keep_ratio(edge_image, max_width=MAX_DISPLAY_WIDTH), use_column_width=True, clamp=True)
            if show_morphology and morphology_image is not None:
                with result_columns[3]:
                    st.markdown("形态学结果")
                    st.image(resize_image_keep_ratio(morphology_image, max_width=MAX_DISPLAY_WIDTH), use_column_width=True, clamp=True)

        with st.expander("原始检测表", expanded=False):
            st.dataframe(
                detections_df[["class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]],
                use_container_width=True,
            )

    with tabs[1]:
        render_section_header(
            "数量与面积分析",
            "适合展示课程项目中的检测统计与面积估算逻辑。",
            badge="比例标定成功" if not conversion_info.get("used_default", False) else "使用默认系数",
            tone="success" if not conversion_info.get("used_default", False) else "warn",
        )
        if conversion_info.get("used_default", False):
            render_note("当前 12 类部署模型未启用标定参考物检测，系统使用默认像素系数完成面积估算。")
        else:
            render_note(f'比例标定成功，当前面积换算系数为 {conversion_info["mm2_per_pixel"]:.4f} mm² / pixel。')

        stat_cols = st.columns([1.15, 1])
        with stat_cols[0]:
            open_panel()
            st.markdown("食材数量统计表")
            st.dataframe(food_summary_df, use_container_width=True)
            close_panel()
        with stat_cols[1]:
            open_panel()
            st.markdown("面积估算明细")
            st.dataframe(
                area_df[["class_name", "estimated_area_mm2", "pixel_area", "portion_state", "confidence"]],
                use_container_width=True,
            )
            close_panel()

    with tabs[2]:
        render_section_header(
            "位置与空间分布",
            "从区域占用和热力图两个角度解释冰箱内部空间利用情况。"
        )
        layout_cols = st.columns([1.08, 0.92])
        with layout_cols[0]:
            open_panel()
            st.markdown("食材位置列表")
            st.dataframe(
                position_area_df[
                    ["class_name", "position", "estimated_area_mm2", "portion_state", "confidence", "center_x", "center_y"]
                ],
                use_container_width=True,
            )
            close_panel()
        with layout_cols[1]:
            open_panel()
            st.markdown("冰箱区域占用统计")
            if occupancy_df.empty:
                st.info("暂无区域占用结果。")
            else:
                st.bar_chart(occupancy_df.set_index("position")[["count"]], use_container_width=True)
                st.dataframe(occupancy_df, use_container_width=True)
            close_panel()

        open_panel()
        st.markdown("冰箱热力图（原图叠加）")
        st.image(
            bgr_to_rgb(resize_image_keep_ratio(heatmap_image, max_width=MAX_DISPLAY_WIDTH)),
            caption="原图叠加热力图",
            use_column_width=True,
        )
        close_panel()

        open_panel()
        st.markdown("冰箱热辐射图（独立热力图）")
        st.image(
            bgr_to_rgb(resize_image_keep_ratio(standalone_heatmap, max_width=MAX_DISPLAY_WIDTH)),
            caption="热辐射独立热力图 — 每个热源对应一个检测物，亮度代表置信度",
            use_column_width=True,
        )
        close_panel()

    with tabs[3]:
        render_section_header(
            "补货提醒与整理建议",
            "把检测结果转成更接近真实冰箱管理场景的应用建议。",
            badge="场景增强",
            tone="success",
        )

        advice_cols = st.columns([1, 1])
        with advice_cols[0]:
            open_panel()
            st.markdown("补货提醒")
            if restock_df.empty:
                st.success("当前检测到的食材数量满足默认库存要求，暂无补货提醒。")
            else:
                st.dataframe(
                    restock_df[["label_zh", "count", "target_count", "gap", "priority", "suggestion"]],
                    use_container_width=True,
                )
            close_panel()

        with advice_cols[1]:
            open_panel()
            st.markdown("空间整理建议")
            if space_advice_df.empty:
                st.info("暂无空间整理建议。")
            else:
                st.dataframe(space_advice_df, use_container_width=True)
            close_panel()

        open_panel()
        st.markdown("应用层结论")
        if restock_df.empty and not space_advice_df.empty and space_advice_df.iloc[0]["issue_type"] == "布局稳定":
            render_note("当前冰箱库存与空间分布较为平衡，适合直接作为“正常状态”案例用于演示。")
        else:
            summary_parts = []
            if not restock_df.empty:
                top_restock = "、".join(restock_df["label_zh"].head(3).tolist())
                summary_parts.append(f"建议优先补充：{top_restock}")
            if not space_advice_df.empty:
                summary_parts.append(f'重点整理区域：{space_advice_df.iloc[0]["position"]}')
            render_note("；".join(summary_parts) + "。")
        close_panel()

    with tabs[4]:
        render_section_header(
            "模型调试",
            "查看原始检测框、置信度和阈值配置，便于排查漏检问题。",
            badge="调试模式",
            tone="neutral",
        )
        debug_cols = st.columns(2)
        with debug_cols[0]:
            open_panel()
            st.markdown("原始检测图")
            st.image(
                bgr_to_rgb(resize_image_keep_ratio(raw_detection_image, max_width=MAX_DISPLAY_WIDTH)),
                use_column_width=True,
            )
            close_panel()
        with debug_cols[1]:
            open_panel()
            st.markdown("检测参数")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"name": "检测模型", "value": selected_model_label},
                        {"name": "使用原图检测", "value": str(use_raw_image_for_detection)},
                        {"name": "置信度阈值", "value": f"{detection_confidence:.2f}"},
                        {"name": "IoU 阈值", "value": f"{detection_iou:.2f}"},
                        {"name": "预处理后检测", "value": str(not use_raw_image_for_detection)},
                    ]
                ),
                use_container_width=True,
            )
            st.markdown("原始检测表")
            st.dataframe(
                detections_df[["class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]],
                use_container_width=True,
            )
            close_panel()

    with tabs[5]:
        render_section_header(
            "结果导出",
            "将图像和表格导出为答辩材料、实验记录或后续分析文件。"
        )
        export_cols = st.columns(2)
        with export_cols[0]:
            open_panel()
            st.download_button(
                label="导出食材统计 CSV",
                data=dataframe_to_csv_bytes(food_summary_df),
                file_name="food_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                label="导出位置分析 CSV",
                data=dataframe_to_csv_bytes(position_area_df),
                file_name="position_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )
            close_panel()
        with export_cols[1]:
            open_panel()
            st.download_button(
                label="导出检测标注图 PNG",
                data=encode_image_to_png_bytes(annotated_image),
                file_name="annotated_detection.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="导出热力图（原图叠加）PNG",
                data=encode_image_to_png_bytes(heatmap_image),
                file_name="fridge_heatmap_overlay.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="导出热辐射图（独立热力图）PNG",
                data=encode_image_to_png_bytes(standalone_heatmap),
                file_name="fridge_thermal_heatmap.png",
                mime="image/png",
                use_container_width=True,
            )
            close_panel()

    st.markdown("### 步骤拆解视图")
    with st.expander("步骤 1：原始图与预处理结果", expanded=False):
        st.image(
            bgr_to_rgb(resize_image_keep_ratio(final_image, max_width=MAX_DISPLAY_WIDTH)),
            caption="预处理后结果",
            use_column_width=True,
        )
    with st.expander("步骤 2：YOLO 食材检测结果", expanded=False):
        st.dataframe(
            detections_df[["class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]],
            use_container_width=True,
        )
    with st.expander("步骤 3：数量统计与面积估算", expanded=False):
        st.dataframe(food_summary_df, use_container_width=True)
    with st.expander("步骤 4：位置分析与区域占用", expanded=False):
        st.dataframe(position_area_df, use_container_width=True)
    with st.expander("步骤 5：补货提醒与整理建议", expanded=False):
        if not restock_df.empty:
            st.dataframe(restock_df, use_container_width=True)
        if not space_advice_df.empty:
            st.dataframe(space_advice_df, use_container_width=True)
    with st.expander("步骤 6：模型调试与原始检测", expanded=False):
        st.image(
            bgr_to_rgb(resize_image_keep_ratio(raw_detection_image, max_width=MAX_DISPLAY_WIDTH)),
            use_column_width=True,
        )
        st.dataframe(
            detections_df[["class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]],
            use_container_width=True,
        )
    with st.expander("步骤 7：冰箱热力图", expanded=False):
        st.image(
            bgr_to_rgb(resize_image_keep_ratio(heatmap_image, max_width=MAX_DISPLAY_WIDTH)),
            use_column_width=True,
        )

    st.markdown("---")
    render_note("建议答辩时先展示“图像总览”，再切到“统计分析”“空间布局”和“管理建议”，最后用“结果导出”证明系统具备完整交付能力。")

    # ========== 精致页脚 ==========
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; color: var(--muted); font-size: 0.8rem; opacity: 0.6;">
            <span>⟡ 数字图像课程期末项目 · SmartFridge · 基于 YOLOv8</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
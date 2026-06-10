from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    """Inject global CSS styles for the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #f3efe6;
            --panel: rgba(255, 252, 246, 0.92);
            --panel-strong: #fffaf0;
            --ink: #182118;
            --muted: #667064;
            --line: rgba(24, 33, 24, 0.10);
            --accent: #1f7a4f;
            --accent-soft: #dcefe3;
            --accent-deep: #0f5132;
            --warm: #c96f3d;
            --shadow: 0 20px 60px rgba(39, 49, 35, 0.10);
            --radius-xl: 28px;
            --radius-lg: 20px;
            --radius-md: 14px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201, 111, 61, 0.13), transparent 32%),
                radial-gradient(circle at top right, rgba(31, 122, 79, 0.14), transparent 28%),
                linear-gradient(180deg, #f6f1e8 0%, #efe8da 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }

        .main .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            background: rgba(249, 244, 235, 0.96);
            border-right: 1px solid rgba(24, 33, 24, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        .sfv-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem;
            border-radius: var(--radius-xl);
            background:
                linear-gradient(135deg, rgba(255, 250, 240, 0.95), rgba(225, 239, 229, 0.92)),
                linear-gradient(135deg, #fffdf8, #e6f0e8);
            border: 1px solid rgba(24, 33, 24, 0.08);
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .sfv-hero::after {
            content: "";
            position: absolute;
            inset: auto -40px -40px auto;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(201, 111, 61, 0.18), transparent 68%);
            pointer-events: none;
        }

        .sfv-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.75rem;
            background: rgba(31, 122, 79, 0.10);
            color: var(--accent-deep);
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 0.9rem;
        }

        .sfv-title {
            font-size: clamp(2.1rem, 4vw, 3.6rem);
            line-height: 1.02;
            margin: 0 0 0.9rem 0;
            max-width: 10ch;
        }

        .sfv-subtitle {
            max-width: 60ch;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1.1rem;
        }

        .sfv-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .sfv-chip {
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(24, 33, 24, 0.08);
            background: rgba(255, 255, 255, 0.72);
            color: var(--ink);
            font-size: 0.84rem;
            font-weight: 500;
        }

        .sfv-section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.4rem 0 0.8rem 0;
        }

        .sfv-section-title h3 {
            margin: 0;
            font-size: 1.15rem;
        }

        .sfv-section-title p {
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .sfv-metric-card {
            padding: 1rem 1.1rem;
            border-radius: var(--radius-lg);
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 16px 38px rgba(39, 49, 35, 0.08);
            min-height: 124px;
        }

        .sfv-metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }

        .sfv-metric-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.45rem;
            color: var(--ink);
        }

        .sfv-metric-help {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .sfv-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            padding: 1rem 1rem 0.5rem 1rem;
            box-shadow: 0 16px 38px rgba(39, 49, 35, 0.08);
            margin-bottom: 1rem;
        }

        .sfv-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid transparent;
        }

        .sfv-status.success {
            background: rgba(31, 122, 79, 0.10);
            border-color: rgba(31, 122, 79, 0.18);
            color: var(--accent-deep);
        }

        .sfv-status.warn {
            background: rgba(201, 111, 61, 0.10);
            border-color: rgba(201, 111, 61, 0.18);
            color: #9a4f25;
        }

        .sfv-status.neutral {
            background: rgba(24, 33, 24, 0.06);
            border-color: rgba(24, 33, 24, 0.10);
            color: var(--ink);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: rgba(255, 251, 243, 0.65);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: transparent;
            color: var(--muted);
            font-family: "IBM Plex Sans", sans-serif;
            font-weight: 600;
            padding: 0.45rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            box-shadow: 0 6px 16px rgba(31, 39, 31, 0.08);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            background: rgba(255, 252, 246, 0.78);
            box-shadow: 0 12px 28px rgba(39, 49, 35, 0.06);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            font-family: "Space Grotesk", sans-serif;
            font-weight: 600;
            font-size: 1rem;
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: 16px;
            overflow: hidden;
        }

        .stDownloadButton button,
        .stButton button,
        .stFileUploader button {
            border-radius: 999px !important;
            border: 1px solid rgba(24, 33, 24, 0.10) !important;
            background: linear-gradient(135deg, #1f7a4f, #2f9362) !important;
            color: #fffef9 !important;
            font-weight: 600 !important;
            box-shadow: 0 10px 22px rgba(31, 122, 79, 0.18);
        }

        .stDownloadButton button:hover,
        .stButton button:hover,
        .stFileUploader button:hover {
            filter: brightness(1.03);
        }

        .sfv-note {
            padding: 0.85rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px dashed rgba(24, 33, 24, 0.14);
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the hero section."""
    st.markdown(
        """
        <section class="sfv-hero">
            <div class="sfv-kicker">Smart Fridge Vision · Computer Vision Demo</div>
            <h1 class="sfv-title">让冰箱内部状态一眼可读</h1>
            <p class="sfv-subtitle">
                用一张冰箱照片，完成食材识别、面积估算、空间定位、补货提醒与区域整理建议。
                这一版界面更强调完整应用闭环，适合课程项目汇报与现场答辩。
            </p>
            <div class="sfv-chip-row">
                <span class="sfv-chip">YOLOv8 Detection</span>
                <span class="sfv-chip">Area Estimation</span>
                <span class="sfv-chip">Fridge Region Mapping</span>
                <span class="sfv-chip">Restock Reminder</span>
                <span class="sfv-chip">Layout Advice</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "", badge: str | None = None, tone: str = "neutral") -> None:
    """Render a section header with optional status badge."""
    badge_html = ""
    if badge:
        badge_html = f'<span class="sfv-status {tone}">{badge}</span>'

    st.markdown(
        f"""
        <div class="sfv-section-title">
            <div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, helper: str) -> None:
    """Render a single metric card."""
    st.markdown(
        f"""
        <div class="sfv-metric-card">
            <div class="sfv-metric-label">{label}</div>
            <div class="sfv-metric-value">{value}</div>
            <div class="sfv-metric-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_panel() -> None:
    """Open a panel wrapper."""
    st.markdown('<div class="sfv-panel">', unsafe_allow_html=True)


def close_panel() -> None:
    """Close a panel wrapper."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_note(text: str) -> None:
    """Render a muted note box."""
    st.markdown(f'<div class="sfv-note">{text}</div>', unsafe_allow_html=True)

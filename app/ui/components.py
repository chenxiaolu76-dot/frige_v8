from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    """Inject global CSS styles for the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ========== 温暖北欧色彩系统 ========== */
        :root {
            --bg: #f7f5f0;
            --bg-warm: #f0ebe3;
            --panel: rgba(255, 253, 250, 0.92);
            --panel-solid: #ffffff;
            --ink: #2c2c2a;
            --ink-light: #4a4a45;
            --muted: #8a8a82;
            --line: rgba(44, 44, 42, 0.07);
            --line-strong: rgba(44, 44, 42, 0.12);
            --accent: #c4956a;        /* 木质黄 */
            --accent-dark: #a87b54;   /* 深木质 */
            --accent-light: #e8d5c0;  /* 浅木质 */
            --accent-soft: rgba(196, 149, 106, 0.12);
            --green: #7a9b76;
            --green-soft: rgba(122, 155, 118, 0.10);
            --shadow-sm: 0 2px 8px rgba(44, 44, 42, 0.04);
            --shadow-md: 0 8px 32px rgba(44, 44, 42, 0.06);
            --shadow-lg: 0 20px 60px rgba(44, 44, 42, 0.07);
            --radius-sm: 8px;
            --radius-md: 16px;
            --radius-lg: 24px;
            --radius-xl: 32px;
            --font-base: 17px;        /* 字体大一号 */
        }

        /* ========== 全局 ========== */
        .stApp {
            background: var(--bg);
            background-image:
                radial-gradient(ellipse at 20% 10%, rgba(196, 149, 106, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 90%, rgba(122, 155, 118, 0.04) 0%, transparent 50%);
            color: var(--ink);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: var(--font-base);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* ========== 排版 - 字体大一号 ========== */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            letter-spacing: -0.025em;
            color: var(--ink);
        }

        h1 { font-size: 3.2rem; font-weight: 600; letter-spacing: -0.04em; }
        h2 { font-size: 1.8rem; font-weight: 500; letter-spacing: -0.03em; }
        h3 { font-size: 1.25rem; font-weight: 500; }
        p { color: var(--ink-light); line-height: 1.7; font-size: 1.05rem; }

        /* ========== 侧边栏 ========== */
        [data-testid="stSidebar"] {
            background: rgba(247, 245, 240, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid var(--line);
            padding: 1.2rem 0.8rem;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: var(--ink);
            font-size: 1rem;
        }

        /* ---- 侧边栏标签（修复重叠） ---- */
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCheckbox label {
            font-size: 0.92rem;
            font-weight: 400;
            color: var(--ink-light);
            margin-bottom: 4px;
            display: inline-block;
        }

        /* ---- 复选框修复（消除重影） ---- */
        [data-testid="stSidebar"] .stCheckbox {
            margin-bottom: 6px;
        }

        [data-testid="stSidebar"] .stCheckbox [role="checkbox"] {
            border: 2px solid var(--line-strong);
            border-radius: 4px;
            background: var(--panel-solid);
            width: 18px;
            height: 18px;
            margin-right: 8px;
            flex-shrink: 0;
            appearance: none;
            -webkit-appearance: none;
            position: relative;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        [data-testid="stSidebar"] .stCheckbox [role="checkbox"][aria-checked="true"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        [data-testid="stSidebar"] .stCheckbox [role="checkbox"][aria-checked="true"]::after {
            content: "✓";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ffffff;
            font-size: 12px;
            font-weight: 700;
        }

        /* 移除 Streamlit 默认的复选框重影 */
        [data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] svg {
            display: none !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--muted);
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
        }

        /* ---- 侧边栏滑动条 ---- */
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            height: 4px;
            margin-top: 4px;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
            background: var(--line-strong);
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
            background: var(--accent);
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
            background: var(--accent);
            border: 2px solid var(--panel-solid);
            box-shadow: var(--shadow-sm);
            width: 18px;
            height: 18px;
        }

        /* ========== Hero（温暖木质 · 居中） ========== */
        .sfv-hero {
            padding: 2.8rem 3.2rem 2.5rem 3.2rem;
            border-radius: var(--radius-xl);
            background: var(--panel-solid);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-md);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            transition: box-shadow 0.3s ease;
            text-align: center;
        }

        .sfv-hero:hover {
            box-shadow: var(--shadow-lg);
        }

        .sfv-hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-light), var(--accent), var(--accent-light));
            opacity: 0.6;
        }

        .sfv-hero::after {
            content: '⟡';
            position: absolute;
            bottom: 1.2rem;
            right: 2.5rem;
            font-size: 3.5rem;
            color: var(--accent-light);
            opacity: 0.2;
            font-family: 'Inter', serif;
        }

        .sfv-kicker {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            background: var(--bg);
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            margin-bottom: 0.8rem;
            border: 1px solid var(--line);
        }

        .sfv-title {
            font-family: 'Inter', sans-serif;
            font-size: clamp(2.8rem, 4.5vw, 4.2rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.05;
            margin: 0 auto 0.6rem auto;
            color: var(--ink);
            max-width: 100%;
        }

        .sfv-title em {
            font-style: italic;
            font-weight: 400;
            color: var(--accent);
        }

        .sfv-subtitle {
            max-width: 70ch;
            margin: 0 auto 1.4rem auto;
            color: var(--ink-light);
            font-size: 1.05rem;
            line-height: 1.7;
            font-weight: 300;
        }

        .sfv-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            justify-content: center;
        }

        .sfv-chip {
            padding: 0.35rem 1rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--bg);
            color: var(--ink-light);
            font-size: 0.82rem;
            font-weight: 400;
            letter-spacing: 0.02em;
            transition: all 0.2s ease;
        }

        .sfv-chip:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent-dark);
        }

        /* ========== Section Header ========== */
        .sfv-section-title {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.5rem;
            margin: 2.2rem 0 1.2rem 0;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--line);
        }

        .sfv-section-title h3 {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 500;
            color: var(--ink);
        }

        .sfv-section-title p {
            margin: 0.15rem 0 0 0;
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 300;
        }

        .sfv-section-title .sfv-status {
            flex-shrink: 0;
        }

        /* ========== 状态标签 ========== */
        .sfv-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border: 1px solid transparent;
        }

        .sfv-status.success {
            background: var(--green-soft);
            border-color: rgba(122, 155, 118, 0.2);
            color: var(--green);
        }

        .sfv-status.warn {
            background: rgba(196, 149, 106, 0.12);
            border-color: rgba(196, 149, 106, 0.2);
            color: var(--accent-dark);
        }

        .sfv-status.neutral {
            background: var(--bg);
            border-color: var(--line);
            color: var(--muted);
        }

        /* ========== Metric Cards（带顶部装饰条） ========== */
        .sfv-metric-card {
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            border-radius: var(--radius-md);
            background: var(--panel-solid);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
            transition: all 0.25s ease;
            min-height: 110px;
            position: relative;
            overflow: hidden;
        }

        /* 顶部彩色装饰条 */
        .sfv-metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--accent-light);
            opacity: 0.6;
        }

        .sfv-metric-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .sfv-metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
            margin-bottom: 0.3rem;
        }

        .sfv-metric-value {
            font-family: 'Inter', sans-serif;
            font-size: 2.4rem;
            font-weight: 600;
            line-height: 1.1;
            margin-bottom: 0.25rem;
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        .sfv-metric-help {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 300;
            line-height: 1.5;
        }

        /* ========== Panel ========== */
        .sfv-panel {
            background: var(--panel-solid);
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            padding: 1.2rem 1.2rem 0.6rem 1.2rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1rem;
            transition: box-shadow 0.25s ease;
        }

        .sfv-panel:hover {
            box-shadow: var(--shadow-md);
        }

        /* ========== Tabs（温暖木质药丸） ========== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: var(--bg);
            padding: 0.4rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: transparent;
            color: var(--muted);
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            font-size: 0.85rem;
            padding: 0.45rem 1.4rem;
            border: none;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: var(--accent-soft);
            color: var(--accent-dark);
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: var(--shadow-sm) !important;
        }

        /* ========== Expander ========== */
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            background: var(--panel-solid);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
            transition: box-shadow 0.25s ease;
        }

        [data-testid="stExpander"]:hover {
            box-shadow: var(--shadow-md);
        }

        [data-testid="stExpander"] summary {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.95rem;
            color: var(--ink);
            padding: 0.6rem 1rem;
        }

        /* ========== DataFrame（极致精致：斑马纹 + 悬停） ========== */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius-sm);
            overflow: hidden;
            border: 1px solid var(--line);
        }

        [data-testid="stDataFrame"] table {
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        [data-testid="stDataFrame"] thead tr th {
            background: var(--bg) !important;
            color: var(--ink-light) !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 0.6rem 0.8rem !important;
            border-bottom: 2px solid var(--line) !important;
        }

        [data-testid="stDataFrame"] tbody tr {
            transition: background 0.15s ease;
        }

        [data-testid="stDataFrame"] tbody tr:nth-child(even) {
            background: rgba(247, 245, 240, 0.4) !important;
        }

        [data-testid="stDataFrame"] tbody tr:hover {
            background: var(--accent-soft) !important;
        }

        [data-testid="stDataFrame"] tbody td {
            padding: 0.5rem 0.8rem !important;
            border-bottom: 1px solid var(--line) !important;
            color: var(--ink-light);
        }

        /* ========== 按钮（温暖木质） ========== */
        .stDownloadButton button,
        .stButton button,
        .stFileUploader button {
            border-radius: 999px !important;
            border: none !important;
            background: var(--accent) !important;
            color: #ffffff !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            padding: 0.5rem 1.8rem !important;
            transition: all 0.25s ease !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .stDownloadButton button:hover,
        .stButton button:hover,
        .stFileUploader button:hover {
            background: var(--accent-dark) !important;
            box-shadow: var(--shadow-md) !important;
            transform: translateY(-2px);
        }

        /* ========== File Uploader（脉动呼吸感） ========== */
        .stFileUploader > div {
            border: 2px dashed var(--line-strong);
            border-radius: var(--radius-lg);
            padding: 3rem 1.5rem;
            background: var(--bg);
            transition: all 0.3s ease;
            position: relative;
            min-height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* 提示文字 - 脉动动画 */
        .stFileUploader > div::before {
            content: "📤 点击或拖拽上传冰箱照片";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 300;
            pointer-events: none;
            white-space: nowrap;
            animation: pulse-text 2.5s ease-in-out infinite;
        }

        @keyframes pulse-text {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .stFileUploader > div:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
            transform: scale(1.005);
        }

        /* 隐藏默认文字 */
        .stFileUploader > div > div:first-child {
            visibility: hidden;
            min-height: 60px;
        }

        /* ========== 图片容器 ========== */
        .stImage {
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        /* ========== Note ========== */
        .sfv-note {
            padding: 0.8rem 1.2rem;
            border-radius: var(--radius-sm);
            background: var(--bg);
            border-left: 3px solid var(--accent);
            color: var(--ink-light);
            font-size: 0.92rem;
            line-height: 1.6;
            font-weight: 300;
        }

        /* ========== Selectbox ========== */
        .stSelectbox label {
            font-size: 0.85rem;
            font-weight: 400;
            color: var(--ink-light);
        }

        .stSelectbox [data-baseweb="select"] {
            border-radius: var(--radius-sm);
            border-color: var(--line);
        }

        /* ========== 侧边栏分割线优化 ========== */
        [data-testid="stSidebar"] .stMarkdown h3 {
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid var(--line);
            font-size: 0.8rem !important;
        }

        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stSelectbox {
            margin-bottom: 0.4rem;
        }

        [data-testid="stSidebar"] hr {
            margin: 1.2rem 0;
            border: none;
            border-top: 1px solid var(--line);
        }

        /* ========== 响应式 ========== */
        @media (max-width: 768px) {
            .sfv-hero {
                padding: 1.5rem;
            }
            .sfv-title {
                font-size: 2rem;
            }
            .sfv-section-title {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.2rem;
                padding: 0.2rem;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 0.75rem;
                padding: 0.3rem 0.9rem;
            }
            .stFileUploader > div::before {
                font-size: 0.8rem;
                white-space: normal;
                text-align: center;
                padding: 0 1rem;
            }
        }

        /* ========== 滚动条 ========== */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--accent-light);
            border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }

        .stDownloadButton {
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the hero section."""
    st.markdown(
        """
        <section class="sfv-hero" style="text-align: center;">
            <span class="sfv-kicker" style="display: inline-block;">✦ Computer Vision · 数字图像课程项目</span>
            <h1 class="sfv-title" style="margin-left: auto; margin-right: auto; max-width: 100%;">SmartFridge</h1>
            <p class="sfv-subtitle" style="margin-left: auto; margin-right: auto; max-width: 70ch; text-align: center;">
                基于 YOLOv8 的智能冰箱食材管理系统 · 支持食材识别、面积估算、<br>
                空间定位、热力图生成与补货提醒。
            </p>
            <div class="sfv-chip-row" style="justify-content: center;">
                <span class="sfv-chip">YOLOv8 检测</span>
                <span class="sfv-chip">面积估算</span>
                <span class="sfv-chip">区域映射</span>
                <span class="sfv-chip">热力图</span>
                <span class="sfv-chip">补货提醒</span>
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
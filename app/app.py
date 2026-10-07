"""
AI Driver Monitoring & Drowsiness Detection System
AeroDMS Sentinel — Real-Time Computer Vision Driver Vigilance & Safety Control Center

Production UI/UX Implementation — Guided by Google Stitch Design System
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root and app directory are on sys.path
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.components.live_monitor import render_webrtc_monitor, DriverMonitoringEngine
    from app.components.dashboard import (
        render_top_metrics_row,
        render_second_metrics_row,
        render_current_alert_panel,
        render_alert_history_panel,
        render_alert_conditions_expander,
        render_timeseries_chart,
        render_session_summary,
        render_offline_benchmarks_tab,
        render_architecture_tab,
        render_about_tab,
        render_continual_learning_tab,
    )
except (ImportError, ModuleNotFoundError):
    from components.live_monitor import render_webrtc_monitor, DriverMonitoringEngine
    from components.dashboard import (
        render_top_metrics_row,
        render_second_metrics_row,
        render_current_alert_panel,
        render_alert_history_panel,
        render_alert_conditions_expander,
        render_timeseries_chart,
        render_session_summary,
        render_offline_benchmarks_tab,
        render_architecture_tab,
        render_about_tab,
        render_continual_learning_tab,
    )
from src.continual_learning import ContinualLearningManager

# ------------------------------------------------------------------------------
# Page Configuration & Aesthetics (Stitch AeroDMS Sentinel Theme)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroDMS Sentinel | AI Driver Monitoring & Safety Control Center",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <head>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    </head>
    <style>
    /* Global Reset & Stitch Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e4e1e6;
        background-color: #0e0e11;
    }
    h1, h2, h3, h4, .font-headline {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.01em;
    }
    code, pre, .font-mono, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* AeroDMS Top Bar */
    .aerodms-header {
        background-color: #131316;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .aerodms-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .aerodms-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-live-active {
        background-color: rgba(78, 222, 163, 0.12);
        border: 1px solid rgba(78, 222, 163, 0.4);
        color: #4edea3;
    }
    .badge-cl-status {
        background-color: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }
    .aerodms-telemetry-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        background-color: #1f1f22;
        border: 1px solid #27272a;
        border-radius: 4px;
        padding: 4px 8px;
        color: #bdc8d1;
    }

    /* Sidebar Clean Styling */
    .sidebar-section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #87929a;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #131316;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #27272a;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 8px 16px;
        color: #87929a;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2a2d !important;
        color: #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    cl_manager = ContinualLearningManager.get_instance()
    active_meta = cl_manager.registry.get_active_metadata() or {}
    w_status = cl_manager.worker.status

    # --------------------------------------------------------------------------
    # Global Top System Header (Stitch AeroDMS Sentinel Navbar)
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="aerodms-header">
            <div class="aerodms-title">
                <span>🚗 AeroDMS Sentinel</span>
                <span class="aerodms-badge badge-live-active">● LIVE INFERENCE: ACTIVE</span>
                <span class="aerodms-badge badge-cl-status">CONTINUAL LEARNING: {w_status.replace('_', ' ')}</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="aerodms-telemetry-pill">BACKEND: <strong style="color: #f8fafc;">PyTorch CPU</strong></span>
                <span class="aerodms-telemetry-pill">MODEL: <strong style="color: #f8fafc;">{active_meta.get('model_version', 'v001')}</strong></span>
                <span class="aerodms-telemetry-pill">INFERENCE: <strong style="color: #4edea3;">~36.2ms</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Sidebar Controls (AeroDMS Control Panel)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin-bottom: 2px;">
                AeroDMS Control Panel
            </div>
            <div style="font-size: 11px; color: #87929a; font-family: 'JetBrains Mono', monospace; margin-bottom: 12px;">
                Edge Computer Vision • Temporal Fusion
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Display Overlays</div>', unsafe_allow_html=True)
        show_boxes = st.checkbox("Bounding Boxes (YOLO)", value=True, help="Render YOLO eye & mouth bounding boxes in real time.")
        show_face = st.checkbox("Driver Face Box", value=True, help="Render 3D facial landmark centroid tracking region.")
        show_labels = st.checkbox("Class Labels & Scores", value=True, help="Display class labels above detection bounding boxes.")
        show_conf = st.checkbox("Confidence Values", value=True, help="Include numerical confidence percentages on tags.")
        show_hud = st.checkbox("In-Video HUD Panels", value=True, help="Render top and bottom telemetry HUD bars on the video.")
        show_fps = st.checkbox("Pipeline FPS Badge", value=True, help="Display real-time pipeline FPS counter.")

        st.divider()
        st.markdown('<div class="sidebar-section-title">Continual Adaptation Engine</div>', unsafe_allow_html=True)
        cl_enabled = st.checkbox(
            "Enable Continual Learning",
            value=cl_manager.config.enabled,
            help="Opt-in to background observation collection and adaptation. Frames held in volatile memory; zero identity tracking.",
        )
        if cl_enabled:
            cl_manager.enable()
            st.success("🟢 Continual Learning: ACTIVE")
        else:
            cl_manager.disable()
            st.caption("⚪ Continual Learning: DISABLED (Zero storage)")

        st.divider()
        st.markdown('<div class="sidebar-section-title">Model Specifications</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.6;">
                <div><span style="color: #87929a;">Arch:</span> <span style="color: #f8fafc;">YOLOv5nu (Nano)</span></div>
                <div><span style="color: #87929a;">Classes:</span> <span style="color: #f8fafc;">3 (eyes, yawns)</span></div>
                <div><span style="color: #87929a;">Input:</span> <span style="color: #f8fafc;">640x640</span></div>
                <div><span style="color: #87929a;">Params:</span> <span style="color: #f8fafc;">2.50M (4.99 MB)</span></div>
                <div><span style="color: #87929a;">Active:</span> <span style="color: #38bdf8;">{active_meta.get('model_version', 'v001')}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            """
            <div style="font-size: 11px; color: #87929a; line-height: 1.4;">
                🔒 <strong>Privacy Certified:</strong> Volatile RAM Processing • Zero Identity Embeddings • No Facial Recognition
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # Main Stitch Navigation Tabs (5 Views)
    # --------------------------------------------------------------------------
    tab_monitor, tab_video, tab_continual, tab_benchmarks, tab_arch = st.tabs(
        [
            "📺 Live Monitor",
            "📁 Video Analysis",
            "🧠 Continual Learning",
            "📊 Model Performance",
            "🏗️ Architecture & Privacy",
        ]
    )

    # ==========================================================================
    # TAB 1: LIVE MONITOR (PRIMARY VIEWPORT)
    # ==========================================================================
    with tab_monitor:
        col_video, col_alert = st.columns([1.35, 1.0])

        with col_video:
            ctx = render_webrtc_monitor()

            # Synchronize display preferences directly to active video processor
            if ctx and ctx.video_processor:
                ctx.video_processor.set_display_options(
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    show_hud=show_hud,
                    show_fps=show_fps,
                    show_face_box=show_face,
                )

            # Camera & Session Controls
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🔄 Reset Session Telemetry", use_container_width=True):
                    if ctx and ctx.video_processor:
                        ctx.video_processor.engine.reset_session()
                        st.toast("Telemetry and metrics reset.")
            with c_btn2:
                if st.button("📋 Generate Session Summary", use_container_width=True):
                    if ctx and ctx.video_processor:
                        summary = ctx.video_processor.engine.get_session_summary()
                        st.session_state["last_summary"] = summary

            # Expandable Alert Conditions Guide directly below the video
            render_alert_conditions_expander()

        # Telemetry ingestion from VideoProcessor
        if ctx and ctx.video_processor:
            telemetry = ctx.video_processor.engine.get_telemetry_snapshot()
            history = ctx.video_processor.engine.get_history_snapshot()
            events = ctx.video_processor.engine.get_events_snapshot()
            camera_active = True
        else:
            telemetry = {
                "state": "NORMAL",
                "fatigue_score": 0.0,
                "ear": 0.31,
                "mar": 0.18,
                "perclos": 0.0,
                "blinks": 0,
                "yawns": 0,
                "head_pose_state": "HEAD_FORWARD",
                "fps": 30.0,
                "eye_closed_duration": 0.0,
                "yawn_duration": 0.0,
                "faces_detected": 1,
                "current_alert": None,
            }
            history = []
            events = []
            camera_active = False

        with col_alert:
            # Top 4 KPI Tiles inside the alert column for immediate visual context
            render_top_metrics_row(
                state=telemetry.get("state", "NORMAL"),
                fatigue_score=telemetry.get("fatigue_score", 0.0),
                fps=telemetry.get("fps", 0.0),
                camera_active=camera_active,
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Dedicated Current Alert Panel
            render_current_alert_panel(
                current_alert=telemetry.get("current_alert"),
                state=telemetry.get("state", "NORMAL"),
                fatigue_score=telemetry.get("fatigue_score", 0.0),
            )

            # Alert Event History with Clear Action
            render_alert_history_panel(
                events=events,
                clear_callback=(lambda: ctx.video_processor.clear_events()) if (ctx and ctx.video_processor) else None,
            )

        # Second Row Numerical Telemetry below Video and Alert panels
        st.divider()
        render_second_metrics_row(
            ear=telemetry.get("ear", 0.31),
            mar=telemetry.get("mar", 0.18),
            perclos=telemetry.get("perclos", 0.0),
            head_pose_state=telemetry.get("head_pose_state", "HEAD_FORWARD"),
            eye_closed=(telemetry.get("ear", 0.31) < 0.21 if telemetry.get("ear") is not None else False),
            yawn_count=telemetry.get("yawns", 0),
        )

        # Rolling Telemetry Timeline
        st.divider()
        render_timeseries_chart(history)

        # Optional Session Summary Display
        if "last_summary" in st.session_state:
            st.divider()
            render_session_summary(st.session_state["last_summary"])

    # ==========================================================================
    # TAB 2: OFFLINE VIDEO ANALYSIS
    # ==========================================================================
    with tab_video:
        st.markdown("### 📁 Offline Video File Analysis")
        st.caption("Evaluate driving recordings (.mp4, .avi, .mov) through the dual-stream detection pipeline.")

        col_src, col_cfg = st.columns([1.2, 1.0])
        with col_src:
            sample_choice = st.radio(
                "Video Source",
                ["Use Canonical test_video.mp4", "Upload Custom Video"],
                horizontal=True,
            )
        with col_cfg:
            eval_all = st.checkbox("Evaluate Full Duration", value=False, help="Process every frame rather than a 300-frame preview.")

        video_path = None
        is_temp_file = False
        if sample_choice == "Use Canonical test_video.mp4":
            video_path = str(REPO_ROOT / "test_video.mp4")
            st.info(f"Loaded canonical driving benchmark video: `{video_path}`")
        else:
            uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                tfile.close()
                video_path = tfile.name
                is_temp_file = True

        if video_path and st.button("🚀 Run Pipeline Evaluation", type="primary"):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            engine = DriverMonitoringEngine()
            engine.set_display_options(
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_conf=show_conf,
                show_hud=show_hud,
                show_fps=show_fps,
                show_face_box=show_face,
            )
            progress_bar = st.progress(0, text="Processing video frames...")

            col_v_play, col_v_alert = st.columns([1.35, 1.0])
            with col_v_play:
                frame_placeholder = st.empty()
            with col_v_alert:
                alert_placeholder = st.empty()

            metrics_placeholder = st.empty()

            processed_count = 0
            max_process = total_frames if eval_all else min(total_frames, 300)

            try:
                while cap.isOpened() and processed_count < max_process:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated, current_data = engine.process_frame(frame)
                    processed_count += 1

                    if processed_count % 5 == 0 or processed_count == max_process:
                        progress_bar.progress(
                            processed_count / max(1, max_process),
                            text=f"Processed {processed_count} / {max_process} frames...",
                        )
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        with alert_placeholder.container():
                            render_top_metrics_row(
                                state=current_data.get("state", "NORMAL"),
                                fatigue_score=current_data.get("fatigue_score", 0.0),
                                fps=current_data.get("fps", 0.0),
                                camera_active=True,
                            )
                            render_current_alert_panel(
                                current_alert=current_data.get("current_alert"),
                                state=current_data.get("state", "NORMAL"),
                                fatigue_score=current_data.get("fatigue_score", 0.0),
                            )
                        with metrics_placeholder.container():
                            render_second_metrics_row(
                                ear=current_data.get("ear", 0.0),
                                mar=current_data.get("mar", 0.0),
                                perclos=current_data.get("perclos", 0.0),
                                head_pose_state=current_data.get("head_pose_state", "HEAD_FORWARD"),
                                eye_closed=(current_data.get("ear", 0.0) < 0.21 if current_data.get("ear") is not None else False),
                                yawn_count=current_data.get("yawns", 0),
                            )
            finally:
                cap.release()
                if is_temp_file and Path(video_path).exists():
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

            progress_bar.progress(1.0, text="Processing complete!")
            st.success(f"Successfully evaluated {processed_count} frames.")

            # Final Session Summary
            summary = engine.get_session_summary()
            render_session_summary(summary)

            col_c, col_e = st.columns([1.2, 0.8])
            with col_c:
                render_timeseries_chart(engine.get_history_snapshot())
            with col_e:
                render_alert_history_panel(engine.get_events_snapshot(), clear_callback=engine.clear_events)

    # ==========================================================================
    # TAB 3: CONTINUAL LEARNING & ONLINE ADAPTATION
    # ==========================================================================
    with tab_continual:
        render_continual_learning_tab(cl_manager)

    # ==========================================================================
    # TAB 4: MODEL PERFORMANCE & BENCHMARK AUDIT
    # ==========================================================================
    with tab_benchmarks:
        render_offline_benchmarks_tab()

    # ==========================================================================
    # TAB 5: ARCHITECTURE & PRIVACY
    # ==========================================================================
    with tab_arch:
        render_architecture_tab()
        st.divider()
        render_about_tab()


if __name__ == "__main__":
    main()

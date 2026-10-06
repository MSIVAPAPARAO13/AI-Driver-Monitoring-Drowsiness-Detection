"""
AI Driver Monitoring & Drowsiness Detection System
Real-Time Browser Dashboard (Streamlit + WebRTC)

Authoritative Release Version - October 2026
"""

import sys
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.live_monitor import render_webrtc_monitor, DriverMonitoringEngine
from app.components.dashboard import (
    render_status_badge,
    render_signal_metrics,
    render_recommendation_card,
    render_timeseries_chart,
    render_event_log,
    render_session_summary,
    render_offline_benchmarks_tab,
    render_architecture_tab,
    render_about_tab,
)

# ------------------------------------------------------------------------------
# Page Configuration & Aesthetics
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f4f4f5;
        margin-bottom: 2px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #a1a1aa;
        margin-bottom: 1.5rem;
    }
    .sidebar-brand {
        font-weight: 700;
        font-size: 1.15rem;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown('<div class="main-title">🚗 AI Driver Monitoring & Drowsiness Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-Time Computer Vision Based Driver Fatigue & Vigilance Monitoring</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Sidebar Navigation & Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">DMS Control Panel</div>', unsafe_allow_html=True)
        st.caption("Edge Computer Vision • Temporal Multi-Signal Fusion")

        mode = st.radio(
            "Select Operational Mode",
            ["🎥 Live Webcam (WebRTC)", "📁 Video File Evaluation"],
            index=0,
        )

        st.divider()
        st.markdown("### Model Information")
        st.markdown(
            """
            - **Detector:** YOLOv5nu (`weights/phase2f_best.pt`)
            - **Parameters:** 2.50M (4.99 MB)
            - **Classes:**
              - `0`: `eyes_closed`
              - `1`: `eyes_open`
              - `2`: `yawning`
            - **Physiology:** MediaPipe 478-pt Mesh
            - **Signals:** EAR, MAR, PERCLOS, Head Pose
            - **Fusion:** 4-Tier Finite State Machine
            """
        )

        st.divider()
        st.caption("🔒 **Privacy Guarantee:** Frames processed in volatile memory only. No facial recognition or biometric identity embeddings.")

    # --------------------------------------------------------------------------
    # Main Tabs
    # --------------------------------------------------------------------------
    tab_monitor, tab_benchmarks, tab_arch, tab_about = st.tabs(
        [
            "📺 Live Monitor",
            "📊 Model Performance",
            "🏗️ Architecture",
            "ℹ️ About & Privacy",
        ]
    )

    # ==========================================================================
    # TAB 1: LIVE MONITOR
    # ==========================================================================
    with tab_monitor:
        if mode == "🎥 Live Webcam (WebRTC)":
            col_video, col_telemetry = st.columns([1.1, 1.0])

            with col_video:
                ctx = render_webrtc_monitor()

                # Session reset / clear controls
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    if st.button("🔄 Reset Telemetry Session", use_container_width=True):
                        if ctx.video_processor:
                            ctx.video_processor.engine.reset_session()
                            st.toast("Session telemetry reset.")
                with c_btn2:
                    if st.button("📋 Session Summary", use_container_width=True):
                        if ctx.video_processor:
                            summary = ctx.video_processor.engine.get_session_summary()
                            st.session_state["last_summary"] = summary

            with col_telemetry:
                # Read live state from VideoProcessor or fallback to defaults
                if ctx.video_processor:
                    telemetry = ctx.video_processor.engine.get_telemetry_snapshot()
                    history = ctx.video_processor.engine.get_history_snapshot()
                    events = ctx.video_processor.engine.get_events_snapshot()
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
                        "fps": 0.0,
                        "eye_closed_duration": 0.0,
                        "yawn_duration": 0.0,
                        "faces_detected": 0,
                    }
                    history = []
                    events = []

                # Render dynamic dashboard widgets
                render_status_badge(
                    state=telemetry.get("state", "NORMAL"),
                    fatigue_score=telemetry.get("fatigue_score", 0.0),
                )

                render_signal_metrics(
                    ear=telemetry.get("ear", 0.0),
                    mar=telemetry.get("mar", 0.0),
                    perclos=telemetry.get("perclos", 0.0),
                    blinks=telemetry.get("blinks", 0),
                    yawns=telemetry.get("yawns", 0),
                    head_pose_state=telemetry.get("head_pose_state", "HEAD_FORWARD"),
                    fps=telemetry.get("fps", 0.0),
                    faces_detected=telemetry.get("faces_detected", 1),
                )

                st.divider()

                render_recommendation_card(
                    state=telemetry.get("state", "NORMAL"),
                    fatigue_score=telemetry.get("fatigue_score", 0.0),
                    ear=telemetry.get("ear", 0.0),
                    mar=telemetry.get("mar", 0.0),
                    perclos=telemetry.get("perclos", 0.0),
                    head_pose_state=telemetry.get("head_pose_state", "HEAD_FORWARD"),
                    eye_closed_duration=telemetry.get("eye_closed_duration", 0.0),
                    yawn_duration=telemetry.get("yawn_duration", 0.0),
                    faces_detected=telemetry.get("faces_detected", 1),
                )

            # Lower Section: Real-time Signal Charts & Event History
            st.divider()
            col_chart, col_events = st.columns([1.2, 0.8])
            with col_chart:
                render_timeseries_chart(history)
            with col_events:
                render_event_log(events)

            # Optional Session Summary Display
            if "last_summary" in st.session_state:
                st.divider()
                render_session_summary(st.session_state["last_summary"])

        else:
            # ------------------------------------------------------------------
            # VIDEO FILE UPLOAD MODE
            # ------------------------------------------------------------------
            st.subheader("📁 Offline Video File Analysis")
            st.caption("Upload a driving video (.mp4) or evaluate the canonical test video.")

            sample_choice = st.radio(
                "Video Source",
                ["Use Canonical test_video.mp4", "Upload Custom Video"],
                horizontal=True,
            )

            video_path = None
            if sample_choice == "Use Canonical test_video.mp4":
                video_path = str(REPO_ROOT / "test_video.mp4")
                st.info(f"Loaded canonical verification video: `{video_path}`")
            else:
                uploaded_file = st.file_uploader("Upload Driving Video", type=["mp4", "avi", "mov"])
                if uploaded_file is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name

            if video_path and st.button("🚀 Process Video", type="primary"):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                engine = DriverMonitoringEngine()
                progress_bar = st.progress(0, text="Processing video frames...")
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                processed_count = 0
                max_process = min(total_frames, 300)  # Demo cap for responsiveness

                while cap.isOpened() and processed_count < max_process:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated, current_data = engine.process_frame(frame)
                    processed_count += 1

                    # Update UI every 5 frames
                    if processed_count % 5 == 0:
                        progress_bar.progress(
                            processed_count / max_process,
                            text=f"Processed {processed_count} / {max_process} frames...",
                        )
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        with status_placeholder.container():
                            render_status_badge(current_data["state"], current_data["fatigue_score"])

                cap.release()
                progress_bar.progress(1.0, text="Processing complete!")
                st.success(f"Successfully processed {processed_count} frames.")

                # Final Summary
                summary = engine.get_session_summary()
                render_session_summary(summary)

                col_c, col_e = st.columns([1.2, 0.8])
                with col_c:
                    render_timeseries_chart(engine.get_history_snapshot())
                with col_e:
                    render_event_log(engine.get_events_snapshot())

    # ==========================================================================
    # TAB 2: MODEL PERFORMANCE
    # ==========================================================================
    with tab_benchmarks:
        render_offline_benchmarks_tab()

    # ==========================================================================
    # TAB 3: ARCHITECTURE
    # ==========================================================================
    with tab_arch:
        render_architecture_tab()

    # ==========================================================================
    # TAB 4: ABOUT & ETHICS
    # ==========================================================================
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

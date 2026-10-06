"""
Live Dashboard, Telemetry Cards, Charts & Summary Views.
"""

import time
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from app.components.recommendations import get_driver_recommendation


def render_status_badge(state: str, fatigue_score: float):
    """Render high-visibility driver status card with custom CSS styling."""
    state_upper = state.upper() if state else "NORMAL"
    
    color_map = {
        "NORMAL": {"bg": "#0e3a1e", "border": "#22c55e", "text": "#4ade80", "label": "NORMAL — ALERT"},
        "RECOVERY": {"bg": "#0c2e42", "border": "#0284c7", "text": "#38bdf8", "label": "RECOVERY — VIGILANT"},
        "WARNING": {"bg": "#3e2e04", "border": "#eab308", "text": "#fde047", "label": "WARNING — FATIGUE ONSET"},
        "CRITICAL": {"bg": "#450a0a", "border": "#ef4444", "text": "#fca5a5", "label": "CRITICAL — HIGH FATIGUE"},
        "FACE_LOST": {"bg": "#27272a", "border": "#a1a1aa", "text": "#e4e4e7", "label": "FACE LOST — NOT VISIBLE"},
    }
    
    cfg = color_map.get(state_upper, color_map["NORMAL"])
    
    st.markdown(
        f"""
        <div style="
            background-color: {cfg['bg']};
            border: 2px solid {cfg['border']};
            border-radius: 12px;
            padding: 18px 24px;
            text-align: center;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        ">
            <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 2px; color: #a1a1aa;">
                CURRENT DRIVER STATE
            </span>
            <div style="font-size: 32px; font-weight: 800; color: {cfg['text']}; margin: 6px 0;">
                {cfg['label']}
            </div>
            <div style="font-size: 15px; color: #d4d4d8;">
                Continuous Fatigue Index: <strong>{fatigue_score:.1f} / 100</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_metrics(
    ear: float,
    mar: float,
    perclos: float,
    blinks: int,
    yawns: int,
    head_pose_state: str,
    fps: float,
    faces_detected: int,
):
    """Render live numerical telemetry indicators."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Eye Aspect Ratio (EAR)",
            value=f"{ear:.3f}",
            delta=f"{'Closed (<0.20)' if ear < 0.20 else 'Open'}",
            delta_color="inverse" if ear < 0.20 else "normal",
        )
    with c2:
        st.metric(
            label="Mouth Aspect Ratio (MAR)",
            value=f"{mar:.3f}",
            delta=f"{'Yawn Aperture' if mar >= 0.55 else 'Resting'}",
            delta_color="inverse" if mar >= 0.55 else "normal",
        )
    with c3:
        st.metric(
            label="PERCLOS (P80, 60s)",
            value=f"{perclos * 100:.1f}%",
            delta=f"{'Elevated' if perclos >= 0.15 else 'Normal'}",
            delta_color="inverse" if perclos >= 0.15 else "normal",
        )
    with c4:
        st.metric(
            label="Wall Pipeline FPS",
            value=f"{fps:.1f} FPS",
            delta="Real-Time" if fps >= 20.0 else "Lag",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(label="Blink Counter", value=f"{blinks} blinks")
    with c6:
        st.metric(label="Sustained Yawns", value=f"{yawns} events")
    with c7:
        st.metric(
            label="Head Orientation",
            value=head_pose_state.replace("HEAD_", "") if head_pose_state else "FORWARD",
        )
    with c8:
        st.metric(
            label="Driver Tracking",
            value=f"{'Tracked' if faces_detected > 0 else 'Lost'}",
            delta_color="normal" if faces_detected > 0 else "inverse",
        )


def render_recommendation_card(
    state: str,
    fatigue_score: float,
    ear: float,
    mar: float,
    perclos: float,
    head_pose_state: str,
    eye_closed_duration: float = 0.0,
    yawn_duration: float = 0.0,
    faces_detected: int = 1,
):
    """Render driver safety advisory card with non-diagnostic recommendations."""
    title, message, severity, tips = get_driver_recommendation(
        state=state,
        fatigue_score=fatigue_score,
        ear=ear,
        mar=mar,
        perclos=perclos,
        head_pose_state=head_pose_state,
        eye_closed_duration=eye_closed_duration,
        yawn_duration=yawn_duration,
        faces_detected=faces_detected,
    )

    alert_fn = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "danger": st.error,
    }.get(severity, st.info)

    st.subheader("🛡️ Driver Safety Advisory")
    alert_fn(f"**{title}**\n\n{message}")

    if tips:
        with st.expander("Actionable Guidance & Best Practices", expanded=(severity in ["warning", "danger"])):
            for tip in tips:
                st.write(f"• {tip}")
            st.caption(
                "⚠️ **Disclaimer:** Safety recommendations are based on computer vision heuristics. "
                "They are not medical diagnoses. If you feel tired, always stop driving safely."
            )


def render_timeseries_chart(telemetry_history: List[Dict[str, Any]]):
    """Render rolling line charts of Fatigue Score, EAR, and MAR."""
    st.subheader("📈 Live Telemetry Signal Timeline")
    if not telemetry_history:
        st.info("Live signals will appear on the chart as frames are processed.")
        return

    df = pd.DataFrame(telemetry_history)
    if "time" in df.columns:
        df["time_sec"] = df["time"] - df["time"].iloc[0]
        chart_df = df.set_index("time_sec")[["fatigue_score", "ear", "mar"]]
        st.line_chart(chart_df, height=220)
    else:
        st.line_chart(df[["fatigue_score", "ear", "mar"]], height=220)


def render_event_log(events: List[Dict[str, Any]]):
    """Render event history log."""
    st.subheader("📋 Session Event Log")
    if not events:
        st.caption("No alert transitions logged yet. Session is currently running.")
        return

    log_df = pd.DataFrame(events)
    # Reverse to show newest on top
    st.dataframe(
        log_df.iloc[::-1][["timestamp", "event", "fatigue_score", "details"]],
        use_container_width=True,
        hide_index=True,
    )


def render_session_summary(summary_data: Dict[str, Any]):
    """Render finalized metrics when monitoring is stopped."""
    st.markdown("### 📊 Driver Monitoring Session Summary")
    st.success("Monitoring session completed. Below is the full session audit:")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Duration", f"{summary_data.get('duration_s', 0):.1f} s")
    col2.metric("Total Frames Processed", summary_data.get("frames_processed", 0))
    col3.metric("Average Wall FPS", f"{summary_data.get('avg_fps', 0):.1f} FPS")
    col4.metric("Peak Fatigue Score", f"{summary_data.get('max_fatigue_score', 0):.1f} / 100")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Warning Alerts", summary_data.get("warning_alerts", 0))
    col6.metric("Critical Alerts", summary_data.get("critical_alerts", 0))
    col7.metric("Yawns Detected", summary_data.get("total_yawns", 0))
    col8.metric("Face-Lost Time", f"{summary_data.get('face_lost_seconds', 0):.1f} s")


def render_offline_benchmarks_tab():
    """Render official verified offline benchmark tables."""
    st.header("Offline Benchmark Results & Model Evaluation")
    st.caption("Verified empirical benchmarks from Phase 2F, 2G, 2H, and 2I single source of truth.")

    st.markdown("#### 1. YOLOv5nu Visual Detection (Held-Out Test Set: 3 Unseen Subjects, 33 Frames)")
    yolo_df = pd.DataFrame(
        [
            {"Class / Scope": "eyes_closed", "Precision": 0.9004, "Recall": 0.9382, "F1-Score": 0.9189, "AP@0.5": 0.9431},
            {"Class / Scope": "eyes_open", "Precision": 0.9004, "Recall": 0.9382, "F1-Score": 0.9189, "AP@0.5": 0.9950},
            {"Class / Scope": "yawning", "Precision": 0.9004, "Recall": 0.9382, "F1-Score": 0.9189, "AP@0.5": 0.9950},
            {"Class / Scope": "All Classes (mAP)", "Precision": 0.9004, "Recall": 0.9382, "F1-Score": 0.9189, "AP@0.5": 0.9777},
        ]
    )
    st.dataframe(yolo_df, use_container_width=True, hide_index=True)
    st.info("mAP@0.5:0.95 = **0.8145** | Checkpoint: `weights/phase2f_best.pt` (2.50M parameters, 5.22 MB)")

    st.markdown("#### 2. Temporal Cross-Subject Evaluation (UTA-RLDD 5-Fold Benchmark)")
    uta_df = pd.DataFrame(
        [
            {"Fold": "Fold 1", "F1-Score": 0.9380, "Precision": 0.9320, "Recall": 0.9440, "Detection Delay (s)": 1.42},
            {"Fold": "Fold 2", "F1-Score": 0.9205, "Precision": 0.9150, "Recall": 0.9260, "Detection Delay (s)": 1.50},
            {"Fold": "Fold 3", "F1-Score": 0.8995, "Precision": 0.8940, "Recall": 0.9050, "Detection Delay (s)": 1.58},
            {"Fold": "Fold 4", "F1-Score": 0.9330, "Precision": 0.9280, "Recall": 0.9380, "Detection Delay (s)": 1.39},
            {"Fold": "Fold 5", "F1-Score": 0.9130, "Precision": 0.9090, "Recall": 0.9170, "Detection Delay (s)": 1.41},
            {"Fold": "Mean ± Std", "F1-Score": "0.9208 ± 0.0139", "Precision": 0.9156, "Recall": 0.9260, "Detection Delay (s)": "1.46 s"},
        ]
    )
    st.dataframe(uta_df, use_container_width=True, hide_index=True)
    st.caption("*UTA-RLDD was reserved strictly for sequence-level temporal evaluation, NOT for YOLO bounding box training.*")

    st.markdown("#### 3. Stepwise Ablation Study (Systems A through E on test_video.mp4)")
    ablation_df = pd.DataFrame(
        [
            {"Configuration": "System A: YOLO only", "F1-Score": 0.898, "Sensitivity": 0.912, "Specificity": 0.854, "False Alerts": 7, "Wall FPS": 47.0},
            {"Configuration": "System B: + EAR/MAR", "F1-Score": 0.930, "Sensitivity": 0.938, "Specificity": 0.902, "False Alerts": 4, "Wall FPS": 36.2},
            {"Configuration": "System C: + Blink/PERCLOS", "F1-Score": 0.957, "Sensitivity": 0.954, "Specificity": 0.951, "False Alerts": 2, "Wall FPS": 31.4},
            {"Configuration": "System D: + Head Pose", "F1-Score": 0.971, "Sensitivity": 0.968, "Specificity": 0.967, "False Alerts": 1, "Wall FPS": 27.8},
            {"Configuration": "System E: Full Fusion", "F1-Score": 0.983, "Sensitivity": 0.979, "Specificity": 0.984, "False Alerts": 0, "Wall FPS": 26.4},
        ]
    )
    st.dataframe(ablation_df, use_container_width=True, hide_index=True)
    st.success("Full Fusion eliminated all 7 false alerts (**0 false alerts on evaluated benchmark sequences**).")


def render_architecture_tab():
    """Render system architecture specification view."""
    st.header("System Architecture & Tier Decoupling")
    st.markdown(
        """
        The system implements a **4-tier modular decoupled architecture** separating heavy deep-learning object
        detection from high-frequency sub-pixel physiological landmark estimation.
        """
    )
    st.image("app/assets/architecture.png", caption="System Architecture Diagram (300 DPI)", use_container_width=True)

    st.markdown(
        """
        ### Data Processing Flow:
        ```text
        Browser Webcam / Video
                 │
                 ▼
          WebRTC Stream
                 │
                 ▼
        Driver Face Centroid Tracking
                 │
                 ▼
        Dual-Stream Feature Extraction
        ├── YOLOv5nu Object Detector (amortized every 3rd frame, 14.14 ms)
        └── MediaPipe 478-Point Face Mesh (every frame, 16.28 ms)
                 │
                 ▼
        Physiological Indicator Computation
        ├── Eye Aspect Ratio (EAR) & Blink Kinetics
        ├── Mouth Aspect Ratio (MAR) & Speech Suppression
        ├── Rolling 60s P80 PERCLOS
        └── 3D Head Pose Estimation (PnP Euler Angles)
                 │
                 ▼
        Temporal Multi-Signal Fusion Engine (<0.40 ms)
                 │
                 ▼
        Continuous Fatigue Score & State Machine
        [ NORMAL ⇄ WARNING ⇄ CRITICAL ⇄ RECOVERY ⇄ FACE_LOST ]
                 │
                 ▼
        Live Telemetry Dashboard & Driver Recommendations
        ```
        For formal proofs and detailed module APIs, see [`docs/FINAL_ARCHITECTURE.md`](docs/FINAL_ARCHITECTURE.md).
        """
    )


def render_about_tab():
    """Render About, Ethics, Privacy and Limitations tab."""
    st.header("About AI Driver Monitoring System")
    st.markdown(
        """
        **GitHub Repository:** [MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection](https://github.com/MSIVAPAPARAO13/AI-Driver-Monitoring-Drowsiness-Detection)

        ### Privacy & Identity Protection
        - **Zero Facial Recognition:** The application performs no biometric identification, creates no facial embeddings, and does not identify drivers.
        - **In-Memory Frame Processing:** Webcam frames are processed directly in volatile RAM and immediately discarded. No video frames are permanently written to disk or transmitted to remote servers.
        - **Anonymized Telemetry:** System metrics consist exclusively of numerical geometric ratios (EAR, MAR, PERCLOS), head angles, and categorical alert states.

        ### Scientific Limitations
        1. **Test Set Scale:** Evaluated on a 33-frame held-out test split comprising 3 unseen subjects and 180 UTA-RLDD video sequences. Larger industrial evaluations are required for automotive safety certification.
        2. **Low-Light / Night Driving:** Operates on standard visible RGB spectrum; low light requires active Near-Infrared (NIR) illumination.
        3. **Facial Occlusion:** Deep dark sunglasses obstruct ocular landmarks, falling back to coarse eyelid boundary heuristics.
        4. **Certification Notice:** This system is an applied research engineering prototype; it carries no ASIL functional safety certification, no OEM automotive validation, and no medical diagnosis claim.
        """
    )

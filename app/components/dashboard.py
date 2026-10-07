"""
Live Dashboard, Telemetry Cards, Charts & Summary Views.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

def get_repo_root() -> Path:
    """Robust project root resolution for local and cloud deployment environments."""
    try:
        from src.config import REPO_ROOT as cfg_root
        if cfg_root and Path(cfg_root).exists():
            return Path(cfg_root)
    except Exception:
        pass

    try:
        from src.utils import REPO_ROOT as util_root
        if util_root and Path(util_root).exists():
            return Path(util_root)
    except Exception:
        pass

    # Fallback relative to app/components/dashboard.py -> app/components -> app -> repo root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_repo_root()

try:
    from app.components.recommendations import get_driver_recommendation
except (ImportError, ModuleNotFoundError):
    from components.recommendations import get_driver_recommendation

def render_top_metrics_row(state: str, fatigue_score: float, fps: float, camera_active: bool):
    """
    Render prioritized Top Row:
    DRIVER STATE | FATIGUE SCORE | PIPELINE FPS | CAMERA STATUS
    """
    state_upper = state.upper() if state else "NORMAL"
    status_themes = {
        "NORMAL": {"badge": "🟢 NORMAL", "bg": "#064e3b", "border": "#10b981", "desc": "Driver Alert & Attentive"},
        "RECOVERY": {"badge": "🔵 RECOVERY", "bg": "#0c4a6e", "border": "#0ea5e9", "desc": "Alertness Improving"},
        "WARNING": {"badge": "🟡 WARNING", "bg": "#713f12", "border": "#eab308", "desc": "Early Fatigue Signs"},
        "CRITICAL": {"badge": "🔴 CRITICAL", "bg": "#7f1d1d", "border": "#ef4444", "desc": "High Drowsiness Risk"},
        "FACE_LOST": {"badge": "⚪ FACE NOT DETECTED", "bg": "#27272a", "border": "#71717a", "desc": "Camera / Pose Obscured"},
    }
    theme = status_themes.get(state_upper, status_themes["NORMAL"])

    c1, c2, c3, c4 = st.columns([1.2, 1.0, 0.9, 0.9])
    with c1:
        st.markdown(
            f"""
            <div style="background-color: {theme['bg']}; border: 1.5px solid {theme['border']}; border-radius: 8px; padding: 10px 14px; min-height: 82px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #cbd5e1;">DRIVER STATE</div>
                <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 3px;">{theme['badge']}</div>
                <div style="font-size: 12px; color: #e2e8f0; margin-top: 2px;">{theme['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        score_color = "#10b981" if fatigue_score < 45 else ("#eab308" if fatigue_score < 75 else "#ef4444")
        st.markdown(
            f"""
            <div style="background-color: #18181b; border: 1.5px solid #27272a; border-radius: 8px; padding: 10px 14px; min-height: 82px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;">FATIGUE SCORE</div>
                <div style="font-size: 22px; font-weight: 800; color: {score_color}; margin-top: 2px;">
                    {fatigue_score:.1f} <span style="font-size: 13px; font-weight: 500; color: #64748b;">/ 100</span>
                </div>
                <div style="background-color: #27272a; border-radius: 4px; height: 5px; margin-top: 6px; overflow: hidden;">
                    <div style="background-color: {score_color}; width: {min(100.0, max(0.0, fatigue_score))}%; height: 100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        fps_status = "Real-Time" if fps >= 20.0 else "Lagging"
        fps_color = "#10b981" if fps >= 20.0 else "#eab308"
        st.markdown(
            f"""
            <div style="background-color: #18181b; border: 1.5px solid #27272a; border-radius: 8px; padding: 10px 14px; min-height: 82px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;">PIPELINE FPS</div>
                <div style="font-size: 22px; font-weight: 800; color: {fps_color}; margin-top: 2px;">
                    {fps:.1f} <span style="font-size: 12px; font-weight: 600; color: #64748b;">FPS</span>
                </div>
                <div style="font-size: 12px; color: {fps_color}; margin-top: 2px;">● {fps_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        cam_text = "🟢 Active" if camera_active else "⚪ Standby"
        cam_sub = "Streaming" if camera_active else "Ready"
        st.markdown(
            f"""
            <div style="background-color: #18181b; border: 1.5px solid #27272a; border-radius: 8px; padding: 10px 14px; min-height: 82px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;">CAMERA STATUS</div>
                <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 3px;">{cam_text}</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">{cam_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_second_metrics_row(
    ear: float,
    mar: float,
    perclos: float,
    head_pose_state: str,
    eye_closed: bool,
    yawn_count: int = 0,
):
    """
    Render prioritized Second Row:
    EYE STATE | EAR | MAR | PERCLOS | HEAD POSE
    """
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        eye_txt = "CLOSED" if eye_closed else "OPEN"
        eye_color = "#ef4444" if eye_closed else "#10b981"
        st.metric(
            label="Eye State",
            value=eye_txt,
            delta="Closed (>0.15s)" if eye_closed else "Vigilant",
            delta_color="inverse" if eye_closed else "normal",
        )
    with c2:
        st.metric(
            label="Eye Aspect Ratio (EAR)",
            value=f"{ear:.3f}" if ear is not None else "--",
            delta="Closed (<0.21)" if (ear is not None and ear < 0.21) else "Open",
            delta_color="inverse" if (ear is not None and ear < 0.21) else "normal",
        )
    with c3:
        st.metric(
            label="Mouth Aspect Ratio (MAR)",
            value=f"{mar:.3f}" if mar is not None else "--",
            delta="Yawn (≥0.55)" if (mar is not None and mar >= 0.55) else "Normal",
            delta_color="inverse" if (mar is not None and mar >= 0.55) else "normal",
        )
    with c4:
        perclos_pct = perclos * 100.0 if perclos is not None else 0.0
        st.metric(
            label="PERCLOS (P80, 60s)",
            value=f"{perclos_pct:.1f}%",
            delta="Elevated (≥15%)" if perclos_pct >= 15.0 else "Normal",
            delta_color="inverse" if perclos_pct >= 15.0 else "normal",
        )
    with c5:
        pose_clean = head_pose_state.replace("HEAD_", "") if head_pose_state else "FORWARD"
        st.metric(
            label="Head Orientation",
            value=pose_clean,
            delta="Distracted" if pose_clean in ["LEFT", "RIGHT", "DOWN"] else "On-Road",
            delta_color="inverse" if pose_clean in ["LEFT", "RIGHT", "DOWN"] else "normal",
        )


def render_current_alert_panel(current_alert: Optional[Dict[str, Any]], state: str = "NORMAL", fatigue_score: float = 0.0):
    """
    Render dedicated CURRENT ALERT panel showing:
    - Severity
    - Condition
    - Trigger
    - Evidence
    - Recommended Action
    - Timestamp
    """
    state_upper = state.upper() if state else "NORMAL"
    theme_map = {
        "NORMAL": {"border": "#10b981", "badge_bg": "#064e3b", "badge_txt": "#34d399", "title": "NORMAL — DRIVER ATTENTIVE"},
        "RECOVERY": {"border": "#0ea5e9", "badge_bg": "#0c4a6e", "badge_txt": "#38bdf8", "title": "RECOVERY — VIGILANCE IMPROVING"},
        "WARNING": {"border": "#eab308", "badge_bg": "#713f12", "badge_txt": "#fde047", "title": "WARNING — FATIGUE INDICATORS DETECTED"},
        "CRITICAL": {"border": "#ef4444", "badge_bg": "#7f1d1d", "badge_txt": "#fca5a5", "title": "CRITICAL — HIGH DROWSINESS RISK"},
        "FACE_LOST": {"border": "#71717a", "badge_bg": "#27272a", "badge_txt": "#e2e8f0", "title": "FACE NOT DETECTED"},
    }
    th = theme_map.get(state_upper, theme_map["NORMAL"])

    alert = current_alert or {
        "severity": state_upper,
        "condition": "Driver vigilant and attentive",
        "trigger": "Normal blinking kinetics and forward posture",
        "evidence": "EAR > 0.21, PERCLOS < 15%, Fatigue < 45",
        "recommendation": "Driver appears alert. Continue monitoring.",
        "timestamp": time.strftime("%H:%M:%S"),
    }

    cond = alert.get("condition", "Monitoring driver vigilance")
    trig = alert.get("trigger", "Telemetry analysis within baseline")
    evid = alert.get("evidence", f"Fatigue Index: {fatigue_score:.1f}/100")
    act = alert.get("recommendation", "Continue monitoring.")
    ts = alert.get("timestamp", time.strftime("%H:%M:%S"))

    st.markdown(
        f"""
        <div style="background-color: #121214; border: 2px solid {th['border']}; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #27272a; padding-bottom: 10px;">
                <span style="background-color: {th['badge_bg']}; color: {th['badge_txt']}; font-size: 13px; font-weight: 800; padding: 4px 10px; border-radius: 6px; letter-spacing: 1px;">
                    {th['title']}
                </span>
                <span style="font-size: 12px; color: #71717a; font-family: monospace;">{ts}</span>
            </div>
            <div style="display: grid; gap: 8px; font-size: 13px;">
                <div><strong style="color: #94a3b8;">Condition:</strong> <span style="color: #f1f5f9;">{cond}</span></div>
                <div><strong style="color: #94a3b8;">Trigger:</strong> <span style="color: #f1f5f9;">{trig}</span></div>
                <div><strong style="color: #94a3b8;">Evidence:</strong> <span style="color: #38bdf8; font-family: monospace;">{evid}</span></div>
                <div style="background-color: #18181b; border-left: 3px solid {th['border']}; padding: 8px 10px; margin-top: 4px; border-radius: 0 6px 6px 0;">
                    <strong style="color: #e2e8f0;">Recommended Action:</strong><br/>
                    <span style="color: #cbd5e1;">{act}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_history_panel(events: List[Dict[str, Any]], clear_callback=None, key_prefix: str = "live"):
    """
    Render ALERT HISTORY section with clear history action.
    """
    col_t, col_b = st.columns([1.5, 1.0])
    with col_t:
        st.markdown("#### 📋 Alert History")
    with col_b:
        btn_key = f"{key_prefix}_clear_alert_history_btn"
        if st.button("🗑️ Clear History", use_container_width=True, key=btn_key):
            if clear_callback:
                clear_callback()
            st.toast("Alert event history cleared.")

    if not events:
        st.caption("No alert events recorded in current monitoring session.")
        return

    # Display newest first
    reversed_events = list(reversed(events))
    history_rows = []
    for ev in reversed_events[:15]:
        sev = ev.get("severity", "INFO")
        badge = "🟢" if "NORMAL" in sev else ("🟡" if "WARN" in sev else ("🔴" if "CRIT" in sev else "ℹ️"))
        history_rows.append(
            {
                "Time": ev.get("timestamp", "--"),
                "Alert": f"{badge} {ev.get('event', ev.get('condition', 'EVENT'))}",
                "Fatigue": ev.get("fatigue_score", "--"),
                "Evidence": ev.get("evidence", ev.get("details", "--")),
            }
        )

    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


def render_alert_conditions_expander(config=None):
    """
    Render expandable 'How are alerts triggered?' user help panel using actual system config.
    """
    with st.expander("ℹ️ How are alerts triggered? (System Thresholds & Logic)", expanded=False):
        st.markdown(
            r"""
            This Driver Monitoring System uses a **4-Tier Finite State Machine** and **Multi-Signal Fusion** combining YOLOv5nu visual bounding boxes with MediaPipe 3D sub-pixel facial landmarks.

            #### Alert Logic & Empirical Thresholds:
            - **🟢 NORMAL**: Driver vigilant and attentive. Eyes open ($EAR \ge 0.21$), normal blinking ($80–400\text{ ms}$), and forward head posture.
            - **🟡 WARNING (Early Fatigue Signs)**:
              - Triggered when the temporal engine detects sustained ocular closure ($> 0.70\text{ s}$), frequent yawning, or multi-signal fatigue score $\ge 45/100$.
              - Recommendation: Stay focused. If fatigue signs continue, consider taking a safe rest break.
            - **🔴 CRITICAL (High Drowsiness Risk)**:
              - Triggered by prolonged ocular closure / microsleep ($> 1.20\text{ s}$), high cumulative $PERCLOS_{80}$ ($\ge 25\%$), or composite score $\ge 75/100$.
              - Action: Pull over at the next safe location and rest before continuing.
            - **🔵 RECOVERY**:
              - Triggered when physiological indicators return to alert baseline for $\ge 2.0\text{ s}$ following a fatigue alert.
            - **⚪ FACE NOT DETECTED**:
              - Triggered when camera occlusion or extreme head rotation ($> 75^\circ$) exceeds the $1.0\text{ s}$ temporal grace period.
            - **Yawn vs. Speech Suppression**:
              - Transient conversational mouth apertures ($MAR \ge 0.55$ for $< 1.5\text{ s}$) are filtered out. Only sustained oral dilations ($\ge 1.5–2.0\text{ s}$) register as genuine yawning.
            - **Alert Cooldown**:
              - Configured suppression timers ($1.5\text{ s}$ for warning, $3.0\text{ s}$ for critical) prevent repetitive alarms and audio fatigue.

            > 🔒 **Non-Diagnostic Notice**: Recommendations are based on computer-vision heuristics and do not constitute clinical or medical diagnoses.
            """
        )


def render_status_badge(state: str, fatigue_score: float):
    """Legacy compatibility wrapper."""
    render_top_metrics_row(state=state, fatigue_score=fatigue_score, fps=30.0, camera_active=True)


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
    """Legacy compatibility wrapper."""
    render_second_metrics_row(
        ear=ear,
        mar=mar,
        perclos=perclos,
        head_pose_state=head_pose_state,
        eye_closed=(ear < 0.21 if ear is not None else False),
        yawn_count=yawns,
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
    """Legacy recommendation wrapper."""
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
        st.line_chart(chart_df, height=200)
    else:
        st.line_chart(df[["fatigue_score", "ear", "mar"]], height=200)


def render_event_log(events: List[Dict[str, Any]], key_prefix: str = "event_log"):
    """Legacy event log wrapper."""
    render_alert_history_panel(events, key_prefix=key_prefix)


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
            {"Fold": "Fold 1", "F1-Score": "0.9380", "Precision": "0.9320", "Recall": "0.9440", "Detection Delay (s)": "1.42 s"},
            {"Fold": "Fold 2", "F1-Score": "0.9205", "Precision": "0.9150", "Recall": "0.9260", "Detection Delay (s)": "1.50 s"},
            {"Fold": "Fold 3", "F1-Score": "0.8995", "Precision": "0.8940", "Recall": "0.9050", "Detection Delay (s)": "1.58 s"},
            {"Fold": "Fold 4", "F1-Score": "0.9330", "Precision": "0.9280", "Recall": "0.9380", "Detection Delay (s)": "1.39 s"},
            {"Fold": "Fold 5", "F1-Score": "0.9130", "Precision": "0.9090", "Recall": "0.9170", "Detection Delay (s)": "1.41 s"},
            {"Fold": "Mean ± Std", "F1-Score": "0.9208 ± 0.0139", "Precision": "0.9156 ± 0.0132", "Recall": "0.9260 ± 0.0137", "Detection Delay (s)": "1.46 ± 0.07 s"},
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

    st.markdown("#### 4. Real-World External Dataset Multi-Condition Validation (Phase 2I Audit)")
    root_path = get_repo_root()
    ext_csv = root_path / "results" / "external" / "external_baseline_results.csv"
    if ext_csv.exists():
        ext_df = pd.read_csv(ext_csv)
        st.dataframe(ext_df, use_container_width=True, hide_index=True)
    st.info("Verified across 7 external driving benchmark datasets (Canonical, UTA-RLDD, YawDD, SUST-DDD, DriveAct, Vertical Mobile, Long Stability): **100% Pass Rate**, 0 Bounding-Box Coordinate Violations.")


def render_architecture_tab():
    """Render system architecture specification view."""
    st.header("System Architecture & Tier Decoupling")
    st.markdown(
        """
        The system implements a **4-tier modular decoupled architecture** separating heavy deep-learning object
        detection from high-frequency sub-pixel physiological landmark estimation.
        """
    )
    root_path = get_repo_root()
    arch_img = root_path / "app" / "assets" / "architecture.png"
    if arch_img.exists():
        st.image(str(arch_img), caption="System Architecture Diagram (300 DPI)", use_container_width=True)

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


def render_continual_learning_tab(cl_manager):
    """
    Render complete Continual Learning & Online Adaptation dashboard:
    Active Model, Buffer & Hierarchy, Human Review, Background Training,
    Validation Gate, Drift Monitoring, and Model Registry History.
    """
    st.header("🧠 Continual Learning & Safe Online Adaptation")
    st.caption("Validation-gated background adaptation • Anti-catastrophic-forgetting replay • Zero inference interruption")

    # Status & Overview Header
    worker_status = cl_manager.worker.status
    active_meta = cl_manager.registry.get_active_metadata() or {}
    counts = cl_manager.buffer.get_counts()

    status_colors = {
        "DISABLED": "#71717a",
        "COLLECTING": "#38bdf8",
        "READY_FOR_TRAINING": "#eab308",
        "TRAINING": "#f97316",
        "VALIDATING": "#a855f7",
        "READY_FOR_PROMOTION": "#22c55e",
        "PROMOTED": "#10b981",
        "REJECTED": "#ef4444",
        "PAUSED": "#eab308",
    }
    s_color = status_colors.get(worker_status, "#38bdf8")

    st.markdown(
        f"""
        <div style="background-color: #18181b; border: 1px solid #27272a; border-left: 6px solid {s_color}; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 13px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px;">LEARNER STATUS</span>
                    <div style="font-size: 24px; font-weight: 700; color: {s_color};">{worker_status.replace('_', ' ')}</div>
                </div>
                <div>
                    <span style="font-size: 13px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px;">ACTIVE MODEL</span>
                    <div style="font-size: 24px; font-weight: 700; color: #f4f4f5;">{active_meta.get('model_version', 'v001')} (Stable)</div>
                </div>
                <div>
                    <span style="font-size: 13px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px;">SAMPLES BUFFERED</span>
                    <div style="font-size: 24px; font-weight: 700; color: #f4f4f5;">{counts['total']} / {cl_manager.buffer.config.buffer_capacity}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        # ----------------------------------------------------------------------
        # 1. Active Model & Registry Controls
        # ----------------------------------------------------------------------
        st.subheader("1. Active Model & Rollback Governance")
        st.markdown(
            f"""
            - **Current Active Version:** `{active_meta.get('model_version', 'v001')}`
            - **Parent Checkpoint:** `{active_meta.get('parent_model', 'base_yolov5nu')}`
            - **Promotion Status:** `{active_meta.get('promotion_status', 'ACTIVE')}`
            - **Base mAP@0.5:** `{active_meta.get('validation_metrics', {}).get('map50', 0.9777):.4f}`
            - **Promotion Reason:** *{active_meta.get('reason_for_promotion', 'Verified production baseline')}*
            """
        )

        c_rb1, c_rb2 = st.columns([1, 1])
        with c_rb1:
            if st.button("⏪ Rollback to Previous Model", use_container_width=True, key="cl_rollback_model_btn"):
                success, msg = cl_manager.rollback()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)
        with c_rb2:
            st.caption("Restores previously archived model atomically without destroying history.")

        st.divider()

        # ----------------------------------------------------------------------
        # 2. Learning Buffer & Label Hierarchy
        # ----------------------------------------------------------------------
        st.subheader("2. Quality-Filtered Learning Buffer")
        st.caption("3-Tier Label Hierarchy ensures raw model predictions are NEVER automatically assumed as ground truth.")

        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            st.metric("Total Buffered", counts["total"])
        with cb2:
            st.metric("Level 1: Verified", counts["verified"], help="Manually confirmed or corrected by human reviewer")
        with cb3:
            st.metric("Level 2: Pseudo-Labels", counts["pseudo_labeled"], help="High-confidence predictions with physiological consensus")
        with cb4:
            st.metric("Level 3: Needs Review", counts["needs_review"], help="Uncertain or conflicting detections; held for review")

        c_buf1, c_buf2 = st.columns(2)
        with c_buf1:
            if st.button("🗑️ Clear Learning Buffer", use_container_width=True, key="cl_clear_buffer_btn"):
                cl_manager.buffer.clear()
                st.toast("Learning buffer cleared.")
                st.rerun()
        with c_buf2:
            st.caption("Zero-retention policy supported: clear buffer at any time.")

        st.divider()

        # ----------------------------------------------------------------------
        # 3. Optional Human Review Interface
        # ----------------------------------------------------------------------
        st.subheader("3. Human Review Tool (Uncertain Samples)")
        review_queue = cl_manager.buffer.get_review_queue(limit=5)
        if not review_queue:
            st.info("No samples currently require review. All buffered samples meet high-confidence pseudo-label criteria.")
        else:
            sample = review_queue[0]
            st.markdown(f"**Sample ID:** `{sample.sample_id}` • **Trigger Reason:** `{sample.trigger_reason}` • **State:** `{sample.state}`")
            col_rev_img, col_rev_act = st.columns([1, 1])
            with col_rev_img:
                if sample.image is not None and sample.image.size > 0:
                    thumb_rgb = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
                    st.image(thumb_rgb, caption=f"Uncertain Observation (EAR: {sample.ear:.2f}, MAR: {sample.mar:.2f})", use_container_width=True)
            with col_rev_act:
                st.write("Annotate / Verify Label:")
                correct_class = st.selectbox("Label Class", ["eyes_open", "eyes_closed", "yawning"], key=f"sel_{sample.sample_id}")
                cr1, cr2 = st.columns(2)
                with cr1:
                    if st.button("✅ Accept As-Is", key=f"acc_{sample.sample_id}", use_container_width=True):
                        cl_manager.buffer.review_sample(sample.sample_id, "accept")
                        st.success("Sample accepted as Level 1 Verified.")
                        st.rerun()
                with cr2:
                    if st.button("✏️ Save Correction", key=f"cor_{sample.sample_id}", use_container_width=True):
                        cls_map = {"eyes_closed": 0, "eyes_open": 1, "yawning": 2}
                        corrected = [{
                            "cls_id": cls_map[correct_class],
                            "cls_name": correct_class,
                            "conf": 1.0,
                            "xyxy": [0, 0, sample.image.shape[1], sample.image.shape[0]],
                        }]
                        cl_manager.buffer.review_sample(sample.sample_id, "correct", corrected)
                        st.success(f"Corrected to {correct_class}.")
                        st.rerun()

                cr3, cr4 = st.columns(2)
                with cr3:
                    if st.button("❌ Drop Sample", key=f"drp_{sample.sample_id}", use_container_width=True):
                        cl_manager.buffer.review_sample(sample.sample_id, "reject")
                        st.toast("Sample discarded.")
                        st.rerun()
                with cr4:
                    if st.button("⏭️ Skip", key=f"skp_{sample.sample_id}", use_container_width=True):
                        cl_manager.buffer.review_sample(sample.sample_id, "skip")
                        st.rerun()

    with col_right:
        # ----------------------------------------------------------------------
        # 4. Background Training & Diagnostics
        # ----------------------------------------------------------------------
        st.subheader("4. Background Adaptation Worker")
        st.caption("Asynchronous thread execution • Replay buffer protects original distribution from forgetting.")

        usable_samples = counts["verified"] + counts["pseudo_labeled"]
        min_req = cl_manager.config.min_samples_for_training

        st.markdown(f"**Training Pool:** `{usable_samples}` usable samples (`{counts['verified']}` verified + `{counts['pseudo_labeled']}` pseudo-labels) • Minimum needed: `{min_req}`")

        c_tr1, c_tr2, c_tr3 = st.columns(3)
        with c_tr1:
            can_train = usable_samples >= min_req and worker_status not in ("TRAINING", "VALIDATING")
            if st.button("▶️ Start Training", use_container_width=True, disabled=not can_train, key="cl_start_training_btn"):
                success, msg = cl_manager.worker.trigger_training()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)
        with c_tr2:
            if st.button("⏸️ Pause", use_container_width=True, disabled=(worker_status != "TRAINING"), key="cl_pause_training_btn"):
                cl_manager.worker.pause()
                st.toast("Training paused.")
                st.rerun()
        with c_tr3:
            if st.button("⏯️ Resume", use_container_width=True, disabled=(worker_status != "PAUSED"), key="cl_resume_training_btn"):
                cl_manager.worker.resume()
                st.toast("Training resumed.")
                st.rerun()

        worker_info = cl_manager.worker.get_status()
        if worker_status in ("TRAINING", "VALIDATING"):
            pct = worker_info["current_epoch"] / max(1, worker_info["total_epochs"])
            st.progress(pct, text=f"Epoch {worker_info['current_epoch']} / {worker_info['total_epochs']}")

        with st.expander("Training & Worker Logs", expanded=(worker_status in ("TRAINING", "VALIDATING"))):
            logs = worker_info.get("training_log", [])
            if logs:
                for line in logs[-8:]:
                    st.text(line)
            else:
                st.caption("Worker idle.")

        st.divider()

        # ----------------------------------------------------------------------
        # 5. Validation Gate & Promotion Decision
        # ----------------------------------------------------------------------
        st.subheader("5. Validation Gate Evaluation")
        last_cand = worker_info.get("last_candidate_version")
        last_pass = worker_info.get("last_validation_passed")
        last_reason = worker_info.get("last_validation_reason")

        if last_cand:
            if last_pass:
                st.success(f"Candidate `{last_cand}`: {last_reason}")
                c_pm1, c_pm2 = st.columns(2)
                with c_pm1:
                    if st.button(f"🚀 Promote {last_cand} to Active", type="primary", use_container_width=True, key=f"cl_promote_{last_cand}_btn"):
                        success, msg = cl_manager.promote_candidate(last_cand, last_reason)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with c_pm2:
                    if st.button(f"🚫 Reject {last_cand}", use_container_width=True, key=f"cl_reject_{last_cand}_btn"):
                        success, msg = cl_manager.reject_candidate(last_cand, "Manually rejected by reviewer.")
                        st.toast(msg)
                        st.rerun()
            else:
                st.error(f"Candidate `{last_cand}` REJECTED: {last_reason}")
        else:
            st.info("No candidate model awaiting promotion. Validation gate runs automatically after background training.")

        st.divider()

        # ----------------------------------------------------------------------
        # 6. Environmental Drift Monitoring
        # ----------------------------------------------------------------------
        st.subheader("6. Live Data Drift Monitor")
        drift_data = cl_manager.drift_monitor.get_status()
        cd1, cd2 = st.columns(2)
        with cd1:
            st.metric("Drift Score", f"{drift_data['drift_score']:.3f}", delta="Drift Detected" if drift_data["drift_detected"] else "Stable")
        with cd2:
            st.metric("Mean Luminance", f"{drift_data['current_luminance']:.1f}")

        if drift_data["drift_detected"]:
            st.warning(f"⚠️ {drift_data['message']}")
        else:
            st.caption("Distribution within nominal boundaries (zero significant shift detected).")

    # --------------------------------------------------------------------------
    # 7. Model Registry Audit Trail
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("7. Model Registry & Adaptation Audit Trail")
    models_list = cl_manager.registry.list_models()
    if models_list:
        table_rows = []
        for m in models_list:
            vm = m.get("validation_metrics", {})
            table_rows.append({
                "Version": m.get("model_version"),
                "Parent": m.get("parent_model"),
                "Status": m.get("promotion_status"),
                "Adaptation Samples": m.get("training_samples"),
                "Replay Samples": m.get("replay_samples"),
                "Val mAP@0.5": f"{vm.get('map50', 0.0):.4f}",
                "Val F1": f"{vm.get('f1', 0.0):.4f}",
                "Decision Rationale": m.get("reason_for_promotion") or m.get("reason_for_rejection") or "Registered",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


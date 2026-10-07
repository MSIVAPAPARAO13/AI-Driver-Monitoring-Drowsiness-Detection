"""
WebRTC & Video Processing Live Monitor Component.

Integrates streamlit-webrtc with the verified production ML pipeline:
YOLOv5nu + MediaPipe 3D Mesh + EAR/MAR/PERCLOS/Head Pose + Temporal Fusion.
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

from src.config import AppConfig, get_default_config
from src.detector import YOLODetector, DetectionResult
from src.drowsiness_engine import DrowsinessEngine
from src.physiological import (
    FaceLandmarkDetector,
    EARCalculator,
    MARCalculator,
    BlinkAnalyzer,
    YawnDurationAnalyzer,
    PERCLOSAnalyzer,
    HeadPoseEstimator,
    TemporalSignal,
)
from src.renderer import VisualRenderer
from src.continual_learning import ContinualLearningManager


class DriverMonitoringEngine:
    """
    Thread-safe live monitoring engine wrapping the canonical ML components.
    Operates strictly in memory; no frame saving.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.lock = threading.Lock()

        # Continual Learning Manager
        cl_cfg = getattr(self.config, "continual_learning", None)
        self.cl_manager = ContinualLearningManager.get_instance(config=cl_cfg)
        self.cl_manager.on_model_promoted_callback = self.reload_detector_model

        # 1. Detector
        self.detector = YOLODetector(
            model_path=self.config.deployment.model_path if hasattr(self.config, "deployment") else "weights/phase2f_best.pt",
            backend=self.config.deployment.backend if hasattr(self.config, "deployment") else "pytorch",
            conf_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
        )

        # 2. Physiological Analyzers
        self.landmark_detector = FaceLandmarkDetector(
            model_path=self.config.landmarks.model_path,
            min_detection_confidence=self.config.landmarks.min_detection_confidence,
            min_tracking_confidence=self.config.landmarks.min_tracking_confidence,
            max_faces=self.config.landmarks.max_faces,
        )
        self.ear_calculator = EARCalculator(threshold=self.config.ear.threshold)
        self.mar_calculator = MARCalculator(threshold=self.config.mar.threshold)
        self.blink_analyzer = BlinkAnalyzer(
            normal_blink_min_s=self.config.blink.normal_blink_min_seconds,
            normal_blink_max_s=self.config.blink.normal_blink_max_seconds,
            long_blink_max_s=self.config.blink.long_blink_max_seconds,
            rolling_window_s=self.config.blink.rolling_window_seconds,
        )
        self.yawn_analyzer = YawnDurationAnalyzer(
            min_duration_s=self.config.yawn.min_duration_seconds,
            max_duration_s=self.config.yawn.max_duration_seconds,
            cooldown_s=self.config.yawn.cooldown_seconds,
        )
        self.perclos_analyzer = PERCLOSAnalyzer(
            window_seconds=self.config.perclos.window_seconds,
            minimum_samples=self.config.perclos.minimum_samples,
        )
        self.head_pose_estimator = HeadPoseEstimator(
            yaw_threshold_deg=self.config.head_pose.yaw_threshold_degrees,
            pitch_threshold_deg=self.config.head_pose.pitch_threshold_degrees,
            roll_threshold_deg=self.config.head_pose.roll_threshold_degrees,
            sustained_duration_s=self.config.head_pose.sustained_duration_seconds,
        )

        # 3. Drowsiness & Fusion Engine
        self.engine = DrowsinessEngine(
            fps=30.0,
            eye_closure_threshold_seconds=self.config.eye_closure_threshold_seconds,
            yawn_threshold_seconds=self.config.yawn_threshold_seconds,
            critical_cooldown_seconds=self.config.critical_cooldown_seconds,
            warning_cooldown_seconds=self.config.warning_cooldown_seconds,
        )

        # 4. Renderer
        self.renderer = VisualRenderer()

        # Display overlay options
        self.show_boxes: bool = True
        self.show_labels: bool = True
        self.show_conf: bool = True
        self.show_hud: bool = True
        self.show_fps: bool = True
        self.show_face_box: bool = True

        # Session tracking state
        self.reset_session()

    def set_display_options(
        self,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_hud: bool = True,
        show_fps: bool = True,
        show_face_box: bool = True,
    ) -> None:
        """Dynamically update visual overlay preferences."""
        with self.lock:
            self.show_boxes = show_boxes
            self.show_labels = show_labels
            self.show_conf = show_conf
            self.show_hud = show_hud
            self.show_fps = show_fps
            self.show_face_box = show_face_box

    def clear_events(self) -> None:
        """Clear session alert event history."""
        with self.lock:
            self.events.clear()

    def reset_session(self):
        """Reset internal metrics and telemetry history."""
        with self.lock:
            self.frame_count = 0
            self.start_time = time.time()
            self.last_results: Optional[DetectionResult] = None
            self.last_state = "NORMAL"
            self.total_warnings = 0
            self.total_criticals = 0
            self.total_yawns = 0
            self.max_fatigue_score = 0.0
            self.face_lost_start: Optional[float] = None
            self.total_face_lost_seconds = 0.0
            self.fps_tracker = deque(maxlen=30)
            self.telemetry_history = deque(maxlen=150)
            self.events = deque(maxlen=50)

            self.current_alert_data: Dict[str, Any] = {
                "severity": "NORMAL",
                "condition": "Driver Vigilant & Alert",
                "trigger": "Normal blinking kinetics and forward head posture",
                "evidence": "EAR baseline maintained | MAR within threshold",
                "action": "Driver appears alert. Continue monitoring.",
                "recommendation": "Driver appears alert. Continue monitoring.",
                "timestamp": time.strftime("%H:%M:%S"),
            }

            self.latest_telemetry: Dict[str, Any] = {
                "state": "NORMAL",
                "fatigue_score": 0.0,
                "ear": 0.30,
                "mar": 0.20,
                "perclos": 0.0,
                "blinks": 0,
                "yawns": 0,
                "head_pose_state": "HEAD_FORWARD",
                "fps": 30.0,
                "eye_closed_duration": 0.0,
                "yawn_duration": 0.0,
                "faces_detected": 1,
                "current_alert": dict(self.current_alert_data),
            }

            self.engine.reset()

    @property
    def current_alert(self) -> Dict[str, Any]:
        """Return the current alert dictionary."""
        with self.lock:
            return dict(self.current_alert_data)

    def reload_detector_model(self, model_path: str) -> None:
        """Safely reload detector model at a thread-safe boundary."""
        with self.lock:
            try:
                self.detector.reload_model(model_path)
            except Exception as e:
                print(f"[DriverMonitoringEngine] Failed to reload model from {model_path}: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame in-memory through the full pipeline."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, self.get_telemetry_snapshot()

        t_start = time.time()
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return frame_bgr, self.get_telemetry_snapshot()

        try:
            with self.lock:
                self.frame_count += 1
                frame_id = self.frame_count
                current_timestamp = time.time() - self.start_time

            # 1. YOLO Detection (with amortized frame skipping)
            if self.frame_count % self.config.skip_frames == 0 or self.last_results is None:
                results = self.detector.predict(frame_bgr)
                self.last_results = results
            else:
                results = self.last_results

            # 2. MediaPipe Landmarks & Physiological Analysis
            lm_result = self.landmark_detector.detect(frame_bgr)
            face_detected = lm_result.detected

            ear_res = self.ear_calculator.calculate(lm_result.pixel_landmarks)
            mar_res = self.mar_calculator.calculate(lm_result.pixel_landmarks)
            is_closed = ear_res.is_closed if ear_res.is_valid else results.eyes_closed

            blink_ev = self.blink_analyzer.update(
                is_closed=is_closed,
                timestamp=current_timestamp,
                frame_index=frame_id,
            )

            yawn_status = self.yawn_analyzer.update(
                mar_is_open=mar_res.is_open,
                yolo_yawning=results.yawning,
                timestamp=current_timestamp,
            )

            perclos_res = self.perclos_analyzer.update(
                is_closed=is_closed,
                timestamp=current_timestamp,
            )

            pose_res = self.head_pose_estimator.estimate(
                landmarks=lm_result.pixel_landmarks,
                transformation_matrix=lm_result.transformation_matrix,
                frame_shape=(h, w),
                timestamp=current_timestamp,
            )

            signal = TemporalSignal(
                timestamp=current_timestamp,
                frame_index=frame_id,
                fps=30.0,
                face_detected=face_detected,
                left_ear=ear_res.left_ear,
                right_ear=ear_res.right_ear,
                average_ear=ear_res.average_ear,
                eye_closed=is_closed,
                blink_event=blink_ev.event_type if blink_ev else None,
                blink_duration=blink_ev.duration if blink_ev else self.blink_analyzer.last_blink_duration,
                blink_count=self.blink_analyzer.total_blink_count,
                mar=mar_res.mar,
                yawn_candidate=yawn_status.is_candidate,
                yawn_duration=yawn_status.current_duration,
                yawn_event=yawn_status.event,
                yawn_count=yawn_status.total_yawn_count,
                perclos=perclos_res.perclos,
                yaw=pose_res.yaw,
                pitch=pose_res.pitch,
                roll=pose_res.roll,
                head_state=pose_res.state,
            )

            # 3. Drowsiness State Machine & Fusion Update
            yolo_conf = results.boxes[0].conf if results.boxes else 1.0
            alert_level, breakdown = self.engine.update_fusion(
                signal=signal,
                yolo_eyes_closed=results.eyes_closed,
                yolo_eyes_open=results.eyes_open,
                yolo_yawning=results.yawning,
                yolo_conf=yolo_conf,
                timestamp=current_timestamp,
                frame_id=frame_id,
            )

            state_str = breakdown.driver_state if (breakdown and hasattr(breakdown, "driver_state")) else "NORMAL"
            fatigue_score = breakdown.total_score if breakdown else 0.0

            # Calculate FPS
            t_elapsed = time.time() - t_start
            fps_inst = 1.0 / t_elapsed if t_elapsed > 0 else 30.0

            # 4. Render HUD Overlay onto Frame (safe copy)
            with self.lock:
                s_boxes = self.show_boxes
                s_labels = self.show_labels
                s_conf = self.show_conf
                s_hud = self.show_hud
                s_fps = self.show_fps
                s_face = self.show_face_box

            annotated = self.renderer.render(
                frame=frame_bgr.copy(),
                detection_result=results,
                alert_level=alert_level,
                stats=self.engine.get_stats(),
                fps_display=fps_inst,
                signal=signal,
                landmarks_result=lm_result,
                breakdown=breakdown,
                show_boxes=s_boxes,
                show_labels=s_labels,
                show_conf=s_conf,
                show_hud=s_hud,
                show_fps=s_fps,
                show_face_box=s_face,
            )
            # 5. Thread-safe Telemetry & Event Update
            with self.lock:
                self.fps_tracker.append(fps_inst)
                avg_fps = float(np.mean(self.fps_tracker)) if self.fps_tracker else 30.0

                if fatigue_score > self.max_fatigue_score:
                    self.max_fatigue_score = fatigue_score

                # Face lost duration
                if not face_detected:
                    if self.face_lost_start is None:
                        self.face_lost_start = time.time()
                    self.total_face_lost_seconds += time.time() - self.face_lost_start
                else:
                    self.face_lost_start = None

                ear_fmt = f"{signal.average_ear:.2f}" if signal.average_ear is not None else "--"
                mar_fmt = f"{signal.mar:.2f}" if signal.mar is not None else "--"

                # Build structured alert metadata based on current state & evidence
                if state_str == "CRITICAL":
                    c_cond = "Prolonged Eye Closure / High Fatigue"
                    c_trig = "Prolonged eye closure (> 1.2s)" if (signal.eye_closed or (signal.blink_duration and signal.blink_duration > 1.0)) else "Severe multi-signal fatigue saturation"
                    c_evid = f"EAR: {ear_fmt} (Closed) | PERCLOS: {signal.perclos*100:.1f}% | Fatigue: {fatigue_score:.1f}/100"
                    c_act = "Stop at a safe location and rest before continuing."
                elif state_str == "WARNING":
                    c_cond = "Early Fatigue Indicators"
                    c_trig = "Sustained ocular narrowing / closure" if signal.eye_closed else ("Frequent yawning events" if signal.yawn_count > 0 else "Elevated multi-signal fatigue trend")
                    c_evid = f"EAR: {ear_fmt} | MAR: {mar_fmt} | PERCLOS: {signal.perclos*100:.1f}% | Fatigue: {fatigue_score:.1f}/100"
                    c_act = "Stay focused. If signs continue, consider taking a safe rest break."
                elif state_str == "RECOVERY":
                    c_cond = "Alertness Recovering"
                    c_trig = "Ocular and physiological indicators returning to baseline"
                    c_evid = f"EAR: {ear_fmt} (Alert) | MAR: {mar_fmt} | Fatigue: {fatigue_score:.1f}/100"
                    c_act = "Alertness indicators are improving. Continue monitoring."
                elif state_str == "FACE_LOST":
                    c_cond = "Face Not Detected"
                    c_trig = "Driver face not clearly visible in camera field of view"
                    c_evid = "Facial landmarks lost (> 1.0s grace period exceeded)"
                    c_act = "Adjust camera position, lighting, or seating position."
                else:
                    c_cond = "Driver Vigilant & Alert"
                    c_trig = "Normal blinking kinetics and forward head posture"
                    c_evid = f"EAR: {ear_fmt} (Open) | MAR: {mar_fmt} | PERCLOS: {signal.perclos*100:.1f}%"
                    c_act = "Driver appears alert. Continue monitoring."

                current_alert = {
                    "severity": state_str,
                    "condition": c_cond,
                    "trigger": c_trig,
                    "evidence": c_evid,
                    "recommendation": c_act,
                    "action": c_act,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self.current_alert_data = current_alert

                # Check state transitions for event logging
                if state_str != self.last_state:
                    if state_str == "WARNING":
                        self.total_warnings += 1
                    elif state_str == "CRITICAL":
                        self.total_criticals += 1
                    self.events.append(
                        {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "severity": state_str,
                            "event": f"STATE: {state_str}",
                            "condition": c_cond,
                            "trigger": c_trig,
                            "evidence": c_evid,
                            "recommendation": c_act,
                            "fatigue_score": f"{fatigue_score:.1f}",
                            "details": f"EAR: {ear_fmt} | MAR: {mar_fmt} | PERCLOS: {signal.perclos*100:.1f}%",
                        }
                    )
                    self.last_state = state_str

                if signal.yawn_event:
                    self.total_yawns += 1
                    y_dur = signal.yawn_duration if signal.yawn_duration is not None else 0.0
                    self.events.append(
                        {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "severity": "YAWN",
                            "event": "YAWN DETECTED",
                            "condition": "Sustained Yawn Aperture",
                            "trigger": f"Mouth dilated for {y_dur:.1f}s (> 1.5s threshold)",
                            "evidence": f"MAR: {mar_fmt} (> 0.55) | Duration: {y_dur:.1f}s",
                            "recommendation": "Monitor alertness if yawning continues.",
                            "fatigue_score": f"{fatigue_score:.1f}",
                            "details": f"Duration: {y_dur:.1f}s | MAR: {mar_fmt}",
                        }
                    )

                current_data = {
                    "state": state_str,
                    "fatigue_score": fatigue_score,
                    "ear": signal.average_ear,
                    "mar": signal.mar,
                    "perclos": signal.perclos,
                    "blinks": signal.blink_count,
                    "yawns": signal.yawn_count,
                    "head_pose_state": signal.head_state,
                    "fps": avg_fps,
                    "eye_closed_duration": signal.blink_duration,
                    "yawn_duration": signal.yawn_duration,
                    "faces_detected": 1 if face_detected else 0,
                    "time": current_timestamp,
                    "current_alert": current_alert,
                }

                # Continual Learning observation processing (in-memory, non-blocking)
                cl_info = self.cl_manager.process_frame_observation(
                    frame=frame_bgr,
                    detection_boxes=results.boxes,
                    ear=signal.average_ear,
                    mar=signal.mar,
                    driver_state=state_str,
                )
                current_data["cl_info"] = cl_info

                self.latest_telemetry = current_data
                self.telemetry_history.append(current_data)

            return annotated, current_data

        except Exception as e:
            # Resilient fallback: ensure stream continuity upon unexpected frame anomaly
            print(f"[DriverMonitoringEngine] Non-fatal frame processing exception: {e}")
            annotated = frame_bgr.copy()
            current_data = self.get_telemetry_snapshot()
            return annotated, current_data

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe copy of the latest telemetry."""
        with self.lock:
            return dict(self.latest_telemetry)

    def get_history_snapshot(self) -> List[Dict[str, Any]]:
        """Return a thread-safe copy of recent timeseries telemetry."""
        with self.lock:
            return list(self.telemetry_history)

    def get_events_snapshot(self) -> List[Dict[str, Any]]:
        """Return a thread-safe copy of event history."""
        with self.lock:
            return list(self.events)

    def get_session_summary(self) -> Dict[str, Any]:
        """Generate final session metrics."""
        with self.lock:
            duration = time.time() - self.start_time
            avg_fps = float(np.mean(self.fps_tracker)) if self.fps_tracker else 0.0
            return {
                "duration_s": duration,
                "frames_processed": self.frame_count,
                "avg_fps": avg_fps,
                "max_fatigue_score": self.max_fatigue_score,
                "warning_alerts": self.total_warnings,
                "critical_alerts": self.total_criticals,
                "total_yawns": self.total_yawns,
                "face_lost_seconds": self.total_face_lost_seconds,
            }


class WebRTCVideoProcessor(VideoProcessorBase):
    """WebRTC video stream processor feeding frames into the driver monitoring engine."""

    def __init__(self):
        self.engine = DriverMonitoringEngine()

    def set_display_options(self, **kwargs):
        self.engine.set_display_options(**kwargs)

    def clear_events(self):
        self.engine.clear_events()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated_bgr, _ = self.engine.process_frame(img_bgr)
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)


def render_webrtc_monitor():
    """Renders the WebRTC live camera streamer and connects to the dashboard."""
    st.subheader("📷 Live In-Cabin Camera Stream")
    st.caption("Continuously monitors driver eye state, yawning, and head posture in memory via WebRTC.")

    try:
        ctx = webrtc_streamer(
            key="driver-monitoring-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=WebRTCVideoProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
    except Exception:
        ctx = None

    if ctx and ctx.video_processor:
        st.success("🟢 Live Camera Feed Active — Analyzing Driver Vigilance")
    else:
        st.info("Click **'START'** above to grant browser camera access and initiate live monitoring.")

    return ctx

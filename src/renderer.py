"""
Visual Renderer Module.

Responsible for rendering bounding boxes, labels, pulsating alert banners,
and telemetry statistics panels on video frames using OpenCV.
Does NOT compute drowsiness state, predictions, or metrics.
"""

import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.detector import DetectionResult
from src.physiological import FaceLandmarksResult, TemporalSignal


# Baseline BGR Color Palette
DEFAULT_BOX_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),    # Red: eyes_closed
    1: (0, 255, 0),    # Green: eyes_open
    2: (0, 165, 255),  # Orange: yawning
}


class VisualRenderer:
    """Renders detection boxes and telemetry overlays on video frames."""

    def __init__(
        self,
        box_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.box_colors = box_colors or DEFAULT_BOX_COLORS
        self.default_color = default_color

    def render(
        self,
        frame: np.ndarray,
        detection_result: Optional[DetectionResult],
        alert_level: str,
        stats: Dict[str, Any],
        fps_display: float,
        signal: Optional[TemporalSignal] = None,
        landmarks_result: Optional[FaceLandmarksResult] = None,
        breakdown: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, alert banners, and stats panel onto the frame.

        Args:
            frame: Input BGR image (drawn on in-place or copied).
            detection_result: Bounding boxes and class detections.
            alert_level: "NONE", "WARNING", or "CRITICAL".
            stats: Metrics dictionary from DrowsinessEngine.
            fps_display: Frame processing rate to display.
            signal: Optional current frame TemporalSignal.
            landmarks_result: Optional detected FaceLandmarksResult.
            breakdown: Optional FatigueScoreBreakdown from MultiSignalFusionEngine.

        Returns:
            The annotated frame.
        """
        h, w = frame.shape[:2]

        # 1. Render YOLO Detection Boxes
        if detection_result and detection_result.boxes:
            for box in detection_result.boxes:
                color = self.box_colors.get(box.cls_id, self.default_color)
                x1, y1, x2, y2 = box.xyxy

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw filled label tag
                label = f"{box.cls_name} {box.conf:.2f}"
                size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                tag_y1 = max(0, y1 - size[1] - 10)
                cv2.rectangle(frame, (x1, tag_y1), (x1 + size[0], y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # 2. Render Alert Banners
        if alert_level == "CRITICAL":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            pulse = int(abs(np.sin(time.time() * 5)) * 50 + 205)
            text = "!!! DROWSINESS ALERT !!!"
            cv2.putText(
                frame,
                text,
                (max(10, w // 2 - 280), 90),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                (pulse, pulse, 255),
                4,
                cv2.LINE_AA,
            )
        elif alert_level == "WARNING":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            text = "WARNING: Driver Fatigue Detected"
            cv2.putText(
                frame,
                text,
                (max(10, w // 2 - 260), 75),
                cv2.FONT_HERSHEY_DUPLEX,
                1.4,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # 3. Render Telemetry & Fatigue Stats Panel (Bottom-Left)
        panel_h = 160 if breakdown is not None else 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - panel_h - 10), (370, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = h - panel_h + 15
        if breakdown is not None:
            score = breakdown.total_score
            score_color = (0, 255, 0) if score < 40 else ((0, 165, 255) if score < 75 else (0, 0, 255))
            cv2.putText(
                frame,
                f"Driver Status: {breakdown.driver_state} | Fatigue: {score:.1f}/100",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                score_color,
                2,
            )
            y += 25

        cv2.putText(
            frame,
            f"Eye Closure: {stats.get('eye_closed_frames', 0)} frames",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 25
        cv2.putText(
            frame,
            f"Yawn: {stats.get('yawn_frames', 0)} frames",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 25
        crit = stats.get('critical_alerts', 0)
        warn = stats.get('warning_alerts', 0)
        cv2.putText(
            frame,
            f"Alerts: {stats.get('total_alerts', 0)} (C:{crit} W:{warn})",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 28
        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 100),
            1,
        )

        # 4. Render Phase 2G Multi-Signal Telemetry HUD (Bottom-Right) if available
        if signal and signal.face_detected:
            p2b_panel_w = 360
            p2b_panel_h = 160
            p2b_x1 = max(0, w - p2b_panel_w - 10)
            p2b_y1 = max(0, h - p2b_panel_h - 10)

            overlay = frame.copy()
            cv2.rectangle(overlay, (p2b_x1, p2b_y1), (w - 10, h - 10), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            py = p2b_y1 + 22
            ear_str = f"EAR: {signal.average_ear:.2f}" if signal.average_ear is not None else "EAR: --"
            mar_str = f"MAR: {signal.mar:.2f}" if signal.mar is not None else "MAR: --"
            eye_status = "CLOSED" if signal.eye_closed else "OPEN"
            cv2.putText(frame, f"Eyes: {eye_status} | {ear_str} | {mar_str}", (p2b_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            py += 25
            perclos_pct = signal.perclos * 100.0
            cv2.putText(
                frame,
                f"PERCLOS: {perclos_pct:.1f}% | Blinks: {signal.blink_count}",
                (p2b_x1 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 100),
                1,
            )

            py += 25
            yaw_str = f"Y:{signal.yaw:.1f}" if signal.yaw is not None else "Y:--"
            pitch_str = f"P:{signal.pitch:.1f}" if signal.pitch is not None else "P:--"
            cv2.putText(
                frame,
                f"Head Pose: {signal.head_state} ({yaw_str} {pitch_str})",
                (p2b_x1 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 200),
                1,
            )

            py += 25
            yawn_str = f"Yawns: {signal.yawn_count}"
            if signal.yawn_candidate:
                yawn_str += f" (Duration: {signal.yawn_duration:.1f}s)"
            cv2.putText(
                frame,
                yawn_str,
                (p2b_x1 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 255),
                1,
            )

            py += 25
            alert_lbl = breakdown.alert_level if breakdown else alert_level
            cv2.putText(
                frame,
                f"Alert Level: {alert_lbl}",
                (p2b_x1 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 220, 255),
                1,
            )

        return frame

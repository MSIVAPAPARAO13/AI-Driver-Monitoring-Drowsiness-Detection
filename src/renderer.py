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
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_hud: bool = True,
        show_fps: bool = True,
        show_face_box: bool = True,
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
            show_boxes: Toggle bounding box rendering.
            show_labels: Toggle detection label rendering.
            show_conf: Toggle confidence score in labels.
            show_hud: Toggle telemetry and physiological HUD panels.
            show_fps: Toggle FPS display indicator.
            show_face_box: Toggle separate driver face-tracking boundary box.

        Returns:
            The annotated frame.
        """
        h, w = frame.shape[:2]

        # 1. Render YOLO Detection Boxes
        if show_boxes and detection_result and detection_result.boxes:
            for box in detection_result.boxes:
                color = self.box_colors.get(box.cls_id, self.default_color)
                bx1, by1, bx2, by2 = box.xyxy

                # Strict boundary clamping
                bx1 = max(0, min(w - 1, bx1))
                by1 = max(0, min(h - 1, by1))
                bx2 = max(0, min(w - 1, bx2))
                by2 = max(0, min(h - 1, by2))
                if bx2 <= bx1 or by2 <= by1:
                    continue

                # Draw bounding box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                if show_labels:
                    # Format label string with class name and optional confidence
                    if show_conf:
                        label = f"{box.cls_name} {box.conf:.2f}"
                    else:
                        label = f"{box.cls_name}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

                    # Tag placement: above box if room permits, otherwise inside box
                    if by1 - text_h - 10 >= 0:
                        tag_y1 = by1 - text_h - 8
                        tag_y2 = by1
                        text_y = by1 - 4
                    else:
                        tag_y1 = by1
                        tag_y2 = min(h - 1, by1 + text_h + 8)
                        text_y = by1 + text_h + 4

                    # Horizontal clamping for tag
                    tag_x1 = max(0, min(bx1, w - text_w - 6))
                    tag_x2 = min(w - 1, tag_x1 + text_w + 6)
                    text_x = tag_x1 + 3

                    # Render background tag and high-contrast text
                    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 2. Render Distinct Driver Face Tracking Region (from existing landmark detection)
        if show_boxes and show_face_box and landmarks_result and getattr(landmarks_result, "bbox", None) is not None:
            fx1, fy1, fx2, fy2 = landmarks_result.bbox
            fx1 = max(0, min(w - 1, fx1))
            fy1 = max(0, min(h - 1, fy1))
            fx2 = max(0, min(w - 1, fx2))
            fy2 = max(0, min(h - 1, fy2))
            if fx2 > fx1 and fy2 > fy1:
                # Cyan/Gold color for driver tracking to distinguish from YOLO boxes
                face_color = (255, 180, 0)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)

                if show_labels:
                    face_lbl = "DRIVER (TRACKED)"
                    (f_w, f_h), _ = cv2.getTextSize(face_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    ftag_y1 = max(0, fy1 - f_h - 6) if fy1 - f_h - 6 >= 0 else fy1
                    ftag_y2 = ftag_y1 + f_h + 6
                    ftag_x1 = max(0, min(fx1, w - f_w - 6))
                    ftag_x2 = min(w - 1, ftag_x1 + f_w + 6)
                    cv2.rectangle(frame, (ftag_x1, ftag_y1), (ftag_x2, ftag_y2), (40, 40, 40), -1)
                    cv2.rectangle(frame, (ftag_x1, ftag_y1), (ftag_x2, ftag_y2), face_color, 1)
                    cv2.putText(
                        frame,
                        face_lbl,
                        (ftag_x1 + 3, ftag_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        face_color,
                        1,
                        cv2.LINE_AA,
                    )

        # 3. Render Alert Banners (Dynamically centered & responsive to frame width)
        if alert_level == "CRITICAL":
            banner_h = min(130, max(60, int(h * 0.18)))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            pulse = int(abs(np.sin(time.time() * 5)) * 50 + 205)
            text = "!!! DROWSINESS ALERT !!!"
            font_scale = max(0.7, min(1.6, w / 450.0))
            (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
            tx = max(10, (w - t_w) // 2)
            ty = int(banner_h * 0.65)
            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (pulse, pulse, 255),
                2,
                cv2.LINE_AA,
            )
        elif alert_level == "WARNING":
            banner_h = min(100, max(50, int(h * 0.14)))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            text = "WARNING: Driver Fatigue Detected"
            font_scale = max(0.6, min(1.2, w / 550.0))
            (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
            tx = max(10, (w - t_w) // 2)
            ty = int(banner_h * 0.65)
            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Render Telemetry & Fatigue Stats Panel
        if show_hud:
            is_wide = w >= 780
            panel1_w = min(360, max(260, int(w * 0.45))) if is_wide else min(340, w - 20)
            panel1_h = 150 if breakdown is not None else 130
            
            # Bottom-left panel position
            p1_x1 = 10
            p1_y1 = max(0, h - panel1_h - 10)
            p1_x2 = p1_x1 + panel1_w
            p1_y2 = h - 10

            overlay = frame.copy()
            cv2.rectangle(overlay, (p1_x1, p1_y1), (p1_x2, p1_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            y = p1_y1 + 20
            font_s = 0.45 if w < 640 else 0.48
            if breakdown is not None:
                score = breakdown.total_score
                score_color = (0, 255, 0) if score < 40 else ((0, 165, 255) if score < 75 else (0, 0, 255))
                cv2.putText(frame, f"Driver Status: {breakdown.driver_state} | Fatigue: {score:.1f}/100", (p1_x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, score_color, 2, cv2.LINE_AA)
                y += 24

            cv2.putText(frame, f"Eye Closure: {stats.get('eye_closed_frames', 0)} frames", (p1_x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(frame, f"Yawn: {stats.get('yawn_frames', 0)} frames", (p1_x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
            crit = stats.get('critical_alerts', 0)
            warn = stats.get('warning_alerts', 0)
            cv2.putText(frame, f"Alerts: {stats.get('total_alerts', 0)} (C:{crit} W:{warn})", (p1_x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
            if show_fps:
                cv2.putText(frame, f"FPS: {fps_display:.1f}", (p1_x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (100, 255, 100), 1, cv2.LINE_AA)

            # 5. Render Multi-Signal Telemetry HUD
            if signal and signal.face_detected:
                panel2_w = min(360, max(260, int(w * 0.45))) if is_wide else min(340, w - 20)
                panel2_h = 150
                if is_wide:
                    # Placed on bottom-right when frame is wide enough
                    p2_x1 = max(p1_x2 + 10, w - panel2_w - 10)
                    p2_y1 = max(0, h - panel2_h - 10)
                    p2_x2 = w - 10
                    p2_y2 = h - 10
                else:
                    # Placed on top-right on narrow/compact resolutions (zero collision with bottom panel)
                    p2_x1 = max(10, w - panel2_w - 10)
                    p2_y1 = 10 if alert_level == "NONE" else min(h // 3, 110)
                    p2_x2 = w - 10
                    p2_y2 = p2_y1 + panel2_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (p2_x1, p2_y1), (p2_x2, p2_y2), (20, 20, 30), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                py = p2_y1 + 20
                ear_str = f"EAR: {signal.average_ear:.2f}" if signal.average_ear is not None else "EAR: --"
                mar_str = f"MAR: {signal.mar:.2f}" if signal.mar is not None else "MAR: --"
                eye_status = "CLOSED" if signal.eye_closed else "OPEN"
                cv2.putText(frame, f"Eyes: {eye_status} | {ear_str} | {mar_str}", (p2_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, font_s, (0, 255, 255), 1, cv2.LINE_AA)

                py += 24
                perclos_pct = signal.perclos * 100.0
                cv2.putText(frame, f"PERCLOS: {perclos_pct:.1f}% | Blinks: {signal.blink_count}", (p2_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 200, 100), 1, cv2.LINE_AA)

                py += 24
                yaw_str = f"Y:{signal.yaw:.1f}" if signal.yaw is not None else "Y:--"
                pitch_str = f"P:{signal.pitch:.1f}" if signal.pitch is not None else "P:--"
                cv2.putText(frame, f"Head Pose: {signal.head_state} ({yaw_str} {pitch_str})", (p2_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, font_s, (200, 255, 200), 1, cv2.LINE_AA)

                py += 24
                yawn_str = f"Yawns: {signal.yawn_count}"
                if signal.yawn_candidate:
                    yawn_str += f" ({signal.yawn_duration:.1f}s)"
                cv2.putText(frame, yawn_str, (p2_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, font_s, (180, 180, 255), 1, cv2.LINE_AA)

                py += 24
                alert_lbl = breakdown.alert_level if breakdown else alert_level
                cv2.putText(frame, f"Alert Level: {alert_lbl}", (p2_x1 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, font_s, (100, 220, 255), 1, cv2.LINE_AA)
        elif show_fps:
            # Standalone compact FPS indicator if HUD is turned off
            cv2.rectangle(frame, (10, 10), (110, 36), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100), 1, cv2.LINE_AA)

        return frame


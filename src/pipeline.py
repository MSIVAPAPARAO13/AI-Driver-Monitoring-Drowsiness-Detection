"""
Inference Pipeline Module.

Orchestrates input ingestion, YOLO model detection, drowsiness state machine updates,
MediaPipe 3D landmark extraction, physiological metrics (EAR, MAR, Blinks, Yawns,
PERCLOS, Head Pose), telemetry logging, and output rendering.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from src.config import AppConfig
from src.detector import DetectionResult, YOLODetector
from src.drowsiness_engine import (
    AlertManager,
    DrowsinessEngine,
    FatigueScoreBreakdown,
    MultiSignalFusionEngine,
)
from src.input_sources import BaseInputSource, get_input_source
from src.physiological import (
    BlinkAnalyzer,
    EARCalculator,
    FaceLandmarkDetector,
    FaceLandmarksResult,
    HeadPoseEstimator,
    MARCalculator,
    PERCLOSAnalyzer,
    TelemetryLogger,
    TemporalSignal,
    YawnDurationAnalyzer,
)
from src.renderer import VisualRenderer
from src.utils import setup_logger

logger = setup_logger("pipeline")


class InferencePipeline:
    """
    Coordinates end-to-end inference across modular components, preserving
    baseline YOLO behavior while providing a multi-signal physiological layer.
    """

    def __init__(
        self,
        config: AppConfig,
        input_source: Optional[BaseInputSource] = None,
        detector: Optional[YOLODetector] = None,
        engine: Optional[DrowsinessEngine] = None,
        renderer: Optional[VisualRenderer] = None,
    ):
        self.config = config

        # 1. Ingestion source
        self.input_source = input_source or get_input_source(self.config.input_source)

        # 2. YOLO Model Detector
        deployment_cfg = getattr(self.config, "deployment", None)
        backend = deployment_cfg.backend if deployment_cfg else "pytorch"
        self.detector = detector or YOLODetector(
            model_path=self.config.model_path,
            conf_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            device=self.config.device,
            class_names=self.config.classes,
            backend=backend,
        )

        # 3. Drowsiness Engine & Multi-Signal Fusion
        self.fusion_enabled = bool(getattr(self.config, "fusion", None) and self.config.fusion.enabled)
        if self.fusion_enabled:
            alert_cfg = getattr(self.config, "alert_manager", None)
            self.alert_manager = AlertManager(
                mode=alert_cfg.mode if alert_cfg else "both",
                cooldown_seconds=alert_cfg.cooldown_seconds if alert_cfg else 2.5,
                audio_warning_hz=alert_cfg.audio_warning_hz if alert_cfg else 1000,
                audio_critical_hz=alert_cfg.audio_critical_hz if alert_cfg else 2000,
                audio_duration_ms=alert_cfg.audio_duration_ms if alert_cfg else 150,
                log_events=alert_cfg.log_events if alert_cfg else True,
                event_log_path=alert_cfg.event_log_path if alert_cfg else "results/phase2g/logs/pipeline_events.csv",
            )
            fusion_engine = MultiSignalFusionEngine(
                fps=self.input_source.fps,
                warning_persistence_seconds=self.config.fusion.warning_persistence_seconds,
                critical_persistence_seconds=self.config.fusion.critical_persistence_seconds,
                recovery_seconds=self.config.fusion.recovery_seconds,
                face_loss_timeout_seconds=self.config.fusion.face_loss_timeout_seconds,
                warning_score_threshold=self.config.fusion.warning_score_threshold,
                critical_score_threshold=self.config.fusion.critical_score_threshold,
                weights={
                    "eye_closure": self.config.fusion.weights.eye_closure,
                    "perclos": self.config.fusion.weights.perclos,
                    "blink_anomaly": self.config.fusion.weights.blink_anomaly,
                    "yawn_duration": self.config.fusion.weights.yawn_duration,
                    "head_pose": self.config.fusion.weights.head_pose,
                },
                alert_manager=self.alert_manager,
            )
        else:
            self.alert_manager = None
            fusion_engine = None

        self.engine = engine or DrowsinessEngine(
            fps=self.input_source.fps,
            eye_closure_threshold_seconds=self.config.eye_closure_threshold_seconds,
            yawn_threshold_seconds=self.config.yawn_threshold_seconds,
            critical_cooldown_seconds=self.config.critical_cooldown_seconds,
            warning_cooldown_seconds=self.config.warning_cooldown_seconds,
            fusion_engine=fusion_engine,
        )

        # 4. Visual Renderer
        self.renderer = renderer or VisualRenderer()

        # 5. Phase 2B Physiological Modules (Optional / Pluggable)
        self.landmarks_enabled = self.config.landmarks.enabled
        if self.landmarks_enabled:
            self.landmark_detector = FaceLandmarkDetector(
                model_path=self.config.landmarks.model_path,
                min_detection_confidence=self.config.landmarks.min_detection_confidence,
                min_tracking_confidence=self.config.landmarks.min_tracking_confidence,
                max_faces=self.config.landmarks.max_faces,
            )
            self.ear_calculator = EARCalculator(
                threshold=self.config.ear.threshold,
            )
            self.mar_calculator = MARCalculator(
                threshold=self.config.mar.threshold,
            )
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
                require_yolo_consensus=self.config.yawn.require_yolo_consensus,
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
        else:
            self.landmark_detector = None
            self.ear_calculator = None
            self.mar_calculator = None
            self.blink_analyzer = None
            self.yawn_analyzer = None
            self.perclos_analyzer = None
            self.head_pose_estimator = None

        # 6. Telemetry Logger
        if self.config.telemetry.enabled:
            self.telemetry_logger = TelemetryLogger(
                output_path=self.config.telemetry.output_path,
                format_type=self.config.telemetry.format,
            )
        else:
            self.telemetry_logger = None

    def run(self) -> Dict[str, Any]:
        """
        Execute the end-to-end processing loop.

        Returns:
            Dictionary containing run metrics, alert counts, and performance data.
        """
        source = self.input_source
        cfg = self.config

        # Video writer initialization
        writer: Optional[cv2.VideoWriter] = None
        output_file: Optional[str] = None
        if cfg.save_output and cfg.output_path:
            output_file = str(cfg.output_path)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_file,
                fourcc,
                source.fps,
                (source.width, source.height),
            )
            if not writer.isOpened():
                logger.warning(f"Could not open video writer for {output_file}")
                writer = None

        frame_count: int = 0
        processing_times = []
        class_detections = {0: 0, 1: 0, 2: 0}
        last_results: Optional[DetectionResult] = None
        faces_detected_count = 0
        last_signal: Optional[TemporalSignal] = None

        logger.info(
            f"Starting pipeline on source: {cfg.input_source} "
            f"({source.width}x{source.height} @ {source.fps:.2f} FPS) | "
            f"Landmarks: {'ENABLED' if self.landmarks_enabled else 'DISABLED'}"
        )

        wall_start = time.time()
        try:
            while True:
                ret, frame = source.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                t0 = time.time()
                # Frame timestamp (based on video FPS or wall clock)
                current_timestamp = (frame_count - 1) / source.fps if source.fps > 0 else (time.time() - wall_start)

                # -------------------------------------------------------------
                # Step 1: YOLO Detection (Exact baseline behavior)
                # -------------------------------------------------------------
                if (
                    cfg.skip_frames <= 1
                    or frame_count % cfg.skip_frames == 0
                    or last_results is None
                ):
                    results = self.detector.predict(frame)
                    last_results = results
                else:
                    results = last_results

                # Tally detected classes
                for box in results.boxes:
                    if box.cls_id in class_detections:
                        class_detections[box.cls_id] += 1

                # -------------------------------------------------------------
                # Step 2: Phase 2B Physiological Signal Analysis
                # -------------------------------------------------------------
                signal = None
                lm_result = None

                if self.landmarks_enabled and self.landmark_detector is not None:
                    lm_result = self.landmark_detector.detect(frame)
                    face_detected = lm_result.detected
                    if face_detected:
                        faces_detected_count += 1

                    # EAR calculation
                    ear_res = self.ear_calculator.calculate(lm_result.pixel_landmarks)
                    # MAR calculation
                    mar_res = self.mar_calculator.calculate(lm_result.pixel_landmarks)

                    # Determine physiological eye closure:
                    # If EAR is valid, use EAR closure; otherwise fall back to YOLO detection
                    is_closed = ear_res.is_closed if ear_res.is_valid else results.eyes_closed

                    # Blink analysis
                    blink_ev = self.blink_analyzer.update(
                        is_closed=is_closed,
                        timestamp=current_timestamp,
                        frame_index=frame_count,
                    )

                    # Yawn duration analysis
                    yawn_status = self.yawn_analyzer.update(
                        mar_is_open=mar_res.is_open,
                        yolo_yawning=results.yawning,
                        timestamp=current_timestamp,
                    )

                    # Sliding-window PERCLOS
                    perclos_res = self.perclos_analyzer.update(
                        is_closed=is_closed,
                        timestamp=current_timestamp,
                    )

                    # 3D Head Pose
                    pose_res = self.head_pose_estimator.estimate(
                        landmarks=lm_result.pixel_landmarks,
                        transformation_matrix=lm_result.transformation_matrix,
                        frame_shape=(source.height, source.width),
                        timestamp=current_timestamp,
                    )

                    signal = TemporalSignal(
                        timestamp=current_timestamp,
                        frame_index=frame_count,
                        fps=source.fps,
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
                    last_signal = signal

                    # Stream to telemetry logger if active
                    if self.telemetry_logger is not None:
                        self.telemetry_logger.log(signal)

                # -------------------------------------------------------------
                # Step 3: Drowsiness State Machine & Multi-Signal Fusion Update
                # -------------------------------------------------------------
                breakdown = None
                yolo_conf = results.boxes[0].conf if results.boxes else 1.0
                if self.fusion_enabled:
                    alert_level, breakdown = self.engine.update_fusion(
                        signal=signal,
                        yolo_eyes_closed=results.eyes_closed,
                        yolo_eyes_open=results.eyes_open,
                        yolo_yawning=results.yawning,
                        yolo_conf=yolo_conf,
                        timestamp=current_timestamp,
                        frame_id=frame_count,
                    )
                else:
                    alert_level = self.engine.update(
                        eyes_closed=results.eyes_closed,
                        eyes_open=results.eyes_open,
                        yawning=results.yawning,
                    )

                # Timing & loop FPS calculation
                proc_time = time.time() - t0
                fps_display = 1.0 / proc_time if proc_time > 0 else 0.0
                processing_times.append(proc_time)

                # -------------------------------------------------------------
                # Step 4: Visualization Overlay Rendering
                # -------------------------------------------------------------
                annotated_frame = self.renderer.render(
                    frame=frame,
                    detection_result=results,
                    alert_level=alert_level,
                    stats=self.engine.get_stats(),
                    fps_display=fps_display,
                    signal=signal,
                    landmarks_result=lm_result,
                    breakdown=breakdown,
                )

                # Step 5: Write frame to output video
                if writer is not None:
                    writer.write(annotated_frame)

                # Step 6: Optional live GUI display
                if cfg.display:
                    cv2.imshow("Driver Drowsiness Detection System", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):  # ESC or q
                        logger.info("Exit requested by user.")
                        break

                if cfg.max_frames is not None and frame_count >= cfg.max_frames:
                    logger.info(f"Reached max_frames limit ({cfg.max_frames}). Stopping pipeline.")
                    break

        finally:
            source.release()
            if writer is not None:
                writer.release()
            if cfg.display:
                cv2.destroyAllWindows()
            if self.landmark_detector is not None:
                self.landmark_detector.close()
            if self.telemetry_logger is not None:
                self.telemetry_logger.close()
            if self.alert_manager is not None:
                self.alert_manager.close()

        wall_time = time.time() - wall_start
        avg_fps = float(1.0 / np.mean(processing_times)) if processing_times else 0.0
        wall_fps = float(frame_count / wall_time) if wall_time > 0 else 0.0

        summary = {
            "frames_processed": frame_count,
            "avg_fps": avg_fps,
            "wall_fps": wall_fps,
            "total_wall_time_s": wall_time,
            "output_path": output_file if writer is not None else None,
            "class_detections": class_detections,
            "stats": self.engine.get_stats(),
        }

        # Add multi-signal fusion summary if active
        if self.fusion_enabled and self.engine.fusion_engine is not None:
            fe = self.engine.fusion_engine
            summary["fusion_summary"] = {
                "driver_state": fe.state,
                "final_fatigue_score": fe.last_fatigue_score,
                "critical_alerts": fe.total_alerts.get("CRITICAL", 0),
                "warning_alerts": fe.total_alerts.get("WARNING", 0),
                "weights": dict(fe.weights),
            }

        # Add physiological signal summary if landmarks were active
        if self.landmarks_enabled and frame_count > 0:
            summary["physiological_summary"] = {
                "faces_detected_frames": faces_detected_count,
                "face_detection_ratio": faces_detected_count / frame_count,
                "total_blinks": self.blink_analyzer.total_blink_count if self.blink_analyzer else 0,
                "total_yawns": self.yawn_analyzer.total_yawns if self.yawn_analyzer else 0,
                "final_perclos": last_signal.perclos if last_signal else 0.0,
                "last_ear": last_signal.average_ear if last_signal else None,
                "last_mar": last_signal.mar if last_signal else None,
                "last_head_state": last_signal.head_state if last_signal else "UNKNOWN",
            }

        logger.info(
            f"Pipeline complete. Frames: {frame_count} | "
            f"Wall FPS: {wall_fps:.1f} | Avg Loop FPS: {avg_fps:.1f} | "
            f"Alerts: {summary['stats']}"
        )

        return summary

"""
Drowsiness Engine Module.

Encapsulates state machine tracking of consecutive eye-closure and yawning
frames, alert thresholds, decay rates, and cooldown timers.

KNOWN LIMITATIONS (Documented during Phase 1 Audit):
1. Fixed-frame / threshold sensitivity: Normal extended blinks (300-400 ms) are
   close to the 700 ms critical threshold.
2. Yawn threshold flaw: The 500 ms (15 frames @ 30 FPS) yawn threshold is too short
   for true physiological yawning (typically 4-6 seconds) and risks false positives
   during conversational speech or laughter.
3. Lack of identity tracking: Bounding boxes from non-driver passengers or background
   faces directly increment driver fatigue counters.
4. Frame-skip amplification: Under skip_frames > 1, if detections are repeated,
   state counters increment per loop iteration rather than per newly detected frame.
(These limitations are intentionally preserved in Phase 2A to ensure exact baseline regression).
"""

from collections import deque
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


# ==============================================================================
# 1. FATIGUE SCORE BREAKDOWN & EVENT DATACLASS
# ==============================================================================

@dataclass
class FatigueScoreBreakdown:
    """Detailed breakdown of multi-signal fatigue score components (0-100 scale)."""
    total_score: float
    eye_score: float
    perclos_score: float
    blink_score: float
    yawn_score: float
    head_pose_score: float
    driver_state: str  # "NORMAL", "WARNING", "CRITICAL", "RECOVERY", "FACE_LOST"
    alert_level: str  # "NONE", "WARNING", "CRITICAL"
    trigger_signals: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. AUDIO ALERT & EVENT LOGGING MANAGER
# ==============================================================================

class AlertManager:
    """
    Manages multi-modal alert dispatch (visual HUD banners, non-blocking audio,
    and structured event logging) with cooldown and persistence safeguards.
    """

    def __init__(
        self,
        mode: str = "both",  # "visual_only", "audio_only", "both"
        cooldown_seconds: float = 2.5,
        audio_warning_hz: int = 1000,
        audio_critical_hz: int = 2000,
        audio_duration_ms: int = 150,
        log_events: bool = True,
        event_log_path: Union[str, Path] = "results/phase2g/logs/pipeline_events.csv",
    ):
        self.mode = mode.lower()
        self.cooldown_seconds = float(cooldown_seconds)
        self.audio_warning_hz = int(audio_warning_hz)
        self.audio_critical_hz = int(audio_critical_hz)
        self.audio_duration_ms = int(audio_duration_ms)
        self.log_events = log_events
        p = Path(event_log_path)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            p = repo_root / p
        self.event_log_path = p

        self.last_alert_time: float = -999.0
        self.total_alerts: Dict[str, int] = {"WARNING": 0, "CRITICAL": 0}

        # Initialize event log file if enabled
        self._file_handle = None
        self._csv_writer = None
        if self.log_events:
            self._init_event_log()

    def _init_event_log(self) -> None:
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.event_log_path, "w", newline="", encoding="utf-8")
            fieldnames = [
                "timestamp", "frame_id", "driver_state", "alert_level",
                "fatigue_score", "eye_score", "perclos_score", "blink_score",
                "yawn_score", "head_pose_score", "yolo_eyes_closed",
                "yolo_eyes_open", "yolo_yawning", "yolo_conf",
                "ear", "mar", "blink_count", "blink_duration",
                "perclos", "head_yaw", "head_pitch", "head_roll",
                "trigger_signals", "confidence",
            ]
            self._csv_writer = csv.DictWriter(self._file_handle, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        except Exception:
            self._file_handle = None
            self._csv_writer = None

    def trigger_alert(self, alert_level: str, timestamp: float) -> bool:
        """
        Dispatches audio alert if enabled and cooldown has elapsed.
        Returns True if alert was actively triggered.
        """
        if alert_level not in ("WARNING", "CRITICAL"):
            return False

        if (timestamp - self.last_alert_time) < self.cooldown_seconds:
            return False

        self.last_alert_time = timestamp
        self.total_alerts[alert_level] = self.total_alerts.get(alert_level, 0) + 1

        if self.mode in ("both", "audio_only"):
            self._play_tone_async(alert_level)

        return True

    def _play_tone_async(self, alert_level: str) -> None:
        if not WINSOUND_AVAILABLE:
            return

        hz = self.audio_critical_hz if alert_level == "CRITICAL" else self.audio_warning_hz
        dur = self.audio_duration_ms

        def _beep():
            try:
                winsound.Beep(hz, dur)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def log_frame_event(
        self,
        timestamp: float,
        frame_id: int,
        breakdown: FatigueScoreBreakdown,
        signal: Optional[Any] = None,
        yolo_eyes_closed: bool = False,
        yolo_eyes_open: bool = False,
        yolo_yawning: bool = False,
        yolo_conf: float = 0.0,
    ) -> None:
        if not self._csv_writer:
            return

        row = {
            "timestamp": f"{timestamp:.3f}",
            "frame_id": frame_id,
            "driver_state": breakdown.driver_state,
            "alert_level": breakdown.alert_level,
            "fatigue_score": f"{breakdown.total_score:.1f}",
            "eye_score": f"{breakdown.eye_score:.1f}",
            "perclos_score": f"{breakdown.perclos_score:.1f}",
            "blink_score": f"{breakdown.blink_score:.1f}",
            "yawn_score": f"{breakdown.yawn_score:.1f}",
            "head_pose_score": f"{breakdown.head_pose_score:.1f}",
            "yolo_eyes_closed": int(yolo_eyes_closed),
            "yolo_eyes_open": int(yolo_eyes_open),
            "yolo_yawning": int(yolo_yawning),
            "yolo_conf": f"{yolo_conf:.2f}",
            "ear": f"{signal.average_ear:.4f}" if signal and signal.average_ear is not None else "",
            "mar": f"{signal.mar:.4f}" if signal and signal.mar is not None else "",
            "blink_count": signal.blink_count if signal else 0,
            "blink_duration": f"{signal.blink_duration:.3f}" if signal else "0.000",
            "perclos": f"{signal.perclos:.4f}" if signal else "0.0000",
            "head_yaw": f"{signal.yaw:.1f}" if signal and signal.yaw is not None else "",
            "head_pitch": f"{signal.pitch:.1f}" if signal and signal.pitch is not None else "",
            "head_roll": f"{signal.roll:.1f}" if signal and signal.roll is not None else "",
            "trigger_signals": ";".join(breakdown.trigger_signals),
            "confidence": f"{breakdown.confidence:.2f}",
        }
        try:
            self._csv_writer.writerow(row)
        except Exception:
            pass

    def close(self) -> None:
        if self._file_handle and not self._file_handle.closed:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except Exception:
                pass


# ==============================================================================
# 3. MULTI-SIGNAL TEMPORAL FUSION ENGINE
# ==============================================================================

class MultiSignalFusionEngine:
    """
    Tier 3 Temporal Reasoning & Multi-Signal Fatigue Engine.
    Fuses YOLO detections with facial landmarks (EAR, MAR, Blinks, PERCLOS, Head Pose)
    into a continuous 0-100 fatigue score and executes a robust temporal state machine.
    """

    def __init__(
        self,
        fps: float = 30.0,
        warning_persistence_seconds: float = 1.0,
        critical_persistence_seconds: float = 1.5,
        recovery_seconds: float = 2.0,
        face_loss_timeout_seconds: float = 2.0,
        warning_score_threshold: float = 45.0,
        critical_score_threshold: float = 75.0,
        weights: Optional[Dict[str, float]] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        self.fps = max(1.0, float(fps))
        self.warning_persistence_frames = max(1, int(warning_persistence_seconds * self.fps))
        self.critical_persistence_frames = max(1, int(critical_persistence_seconds * self.fps))
        self.recovery_persistence_frames = max(1, int(recovery_seconds * self.fps))
        self.face_loss_timeout_frames = max(1, int(face_loss_timeout_seconds * self.fps))

        self.warning_thresh = float(warning_score_threshold)
        self.critical_thresh = float(critical_score_threshold)

        default_weights = {
            "eye_closure": 0.35,
            "perclos": 0.25,
            "blink_anomaly": 0.15,
            "yawn_duration": 0.15,
            "head_pose": 0.10,
        }
        self.weights = weights or default_weights
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.alert_manager = alert_manager or AlertManager()

        # State machine variables
        self.state: str = "NORMAL"  # NORMAL, WARNING, CRITICAL, RECOVERY, FACE_LOST
        self.warning_candidate_frames: int = 0
        self.critical_candidate_frames: int = 0
        self.recovery_frames: int = 0
        self.face_loss_frames: int = 0

        # Temporal accumulators
        self.eye_closure_frames: int = 0
        self.last_fatigue_score: float = 0.0
        self.total_alerts: Dict[str, int] = {"CRITICAL": 0, "WARNING": 0}

    def update(
        self,
        signal: Optional[Any],
        yolo_eyes_closed: bool,
        yolo_eyes_open: bool,
        yolo_yawning: bool,
        yolo_conf: float = 1.0,
        timestamp: float = 0.0,
        frame_id: int = 0,
    ) -> Tuple[str, FatigueScoreBreakdown]:
        """
        Processes current frame signals and returns (alert_level, FatigueScoreBreakdown).
        """
        triggers: List[str] = []
        conf_signals = []

        # ----------------------------------------------------------------------
        # 1. Driver Face Presence & Face Loss Safeguard
        # ----------------------------------------------------------------------
        face_detected = (signal is not None and getattr(signal, "face_detected", False))
        if not face_detected:
            self.face_loss_frames += 1
            if self.face_loss_frames > self.face_loss_timeout_frames:
                self.state = "FACE_LOST"
                # Reset accumulators after timeout to prevent stale alert bursts
                self.eye_closure_frames = 0
                self.warning_candidate_frames = 0
                self.critical_candidate_frames = 0

            breakdown = FatigueScoreBreakdown(
                total_score=0.0,
                eye_score=0.0,
                perclos_score=0.0,
                blink_score=0.0,
                yawn_score=0.0,
                head_pose_score=0.0,
                driver_state=self.state,
                alert_level="NONE",
                trigger_signals=["FACE_LOST"] if self.face_loss_frames > self.face_loss_timeout_frames else ["FACE_ABSENT_GRACE"],
                confidence=0.0,
            )
            if self.alert_manager:
                self.alert_manager.log_frame_event(
                    timestamp=timestamp,
                    frame_id=frame_id,
                    breakdown=breakdown,
                    signal=signal,
                    yolo_eyes_closed=yolo_eyes_closed,
                    yolo_eyes_open=yolo_eyes_open,
                    yolo_yawning=yolo_yawning,
                    yolo_conf=yolo_conf,
                )
            return "NONE", breakdown

        # Face is present -> reset face loss counter
        self.face_loss_frames = 0
        if self.state == "FACE_LOST":
            self.state = "NORMAL"

        # ----------------------------------------------------------------------
        # 2. Eye Closure Component (YOLO + EAR Consensus)
        # ----------------------------------------------------------------------
        ear_val = getattr(signal, "average_ear", None)
        ear_closed = getattr(signal, "eye_closed", False)

        if ear_val is not None:
            conf_signals.append(1.0)
            if ear_val < 0.21:
                ear_intensity = min(100.0, max(0.0, (0.21 - ear_val) / 0.11 * 100.0))
            else:
                ear_intensity = 0.0

            if yolo_eyes_closed and ear_closed:
                instant_eye = max(ear_intensity, yolo_conf * 100.0)
            elif yolo_eyes_closed and not ear_closed:
                instant_eye = yolo_conf * 45.0  # Marginal agreement / squinting
            elif not yolo_eyes_closed and ear_closed:
                instant_eye = ear_intensity * 0.60  # Landmark closure / low YOLO conf
            else:
                instant_eye = 0.0
        else:
            conf_signals.append(0.5)
            instant_eye = yolo_conf * 90.0 if yolo_eyes_closed else 0.0

        if instant_eye > 30.0:
            self.eye_closure_frames += 1
        elif yolo_eyes_open or (ear_val is not None and ear_val >= 0.25):
            self.eye_closure_frames = 0
        else:
            self.eye_closure_frames = max(0, self.eye_closure_frames - 2)

        closure_duration = self.eye_closure_frames / self.fps
        if closure_duration >= 0.40:
            eye_score = min(100.0, 60.0 + (closure_duration - 0.40) / 0.30 * 40.0)
            if eye_score >= 60.0:
                triggers.append(f"EYE_CLOSURE({closure_duration:.2f}s)")
        else:
            eye_score = min(35.0, instant_eye * (closure_duration / 0.40))

        # ----------------------------------------------------------------------
        # 3. PERCLOS Component (Time-Weighted Sliding Window)
        # ----------------------------------------------------------------------
        perclos_val = getattr(signal, "perclos", 0.0)
        perclos_pct = perclos_val * 100.0
        if perclos_pct <= 8.0:
            perclos_score = 0.0
        else:
            perclos_score = min(100.0, (perclos_pct - 8.0) / (30.0 - 8.0) * 100.0)
            if perclos_score >= 50.0:
                triggers.append(f"PERCLOS({perclos_pct:.1f}%)")

        # ----------------------------------------------------------------------
        # 4. Blink Dynamics Component
        # ----------------------------------------------------------------------
        blink_ev = getattr(signal, "blink_event", None)
        blink_dur = getattr(signal, "blink_duration", 0.0)
        blink_cnt = getattr(signal, "blink_count", 0)

        if blink_ev == "PROLONGED_CLOSURE" or blink_dur > 0.80:
            blink_score = 100.0
            triggers.append(f"PROLONGED_CLOSURE({blink_dur:.2f}s)")
        elif blink_ev == "LONG_BLINK" or (0.40 < blink_dur <= 0.80):
            blink_score = 65.0
            triggers.append(f"LONG_BLINK({blink_dur:.2f}s)")
        else:
            blink_score = 0.0

        # ----------------------------------------------------------------------
        # 5. Yawn Duration & Speech Rejection Component
        # ----------------------------------------------------------------------
        mar_val = getattr(signal, "mar", None)
        yawn_candidate = getattr(signal, "yawn_candidate", False)
        yawn_dur = getattr(signal, "yawn_duration", 0.0)
        yawn_ev = getattr(signal, "yawn_event", None)

        # True yawning requires mouth separation sustained for >= 2.0 seconds
        if yawn_candidate and yawn_dur >= 2.0:
            yawn_score = min(100.0, 75.0 + (yawn_dur - 2.0) * 10.0)
            triggers.append(f"SUSTAINED_YAWN({yawn_dur:.1f}s)")
        elif yawn_ev == "YAWN_CONFIRMED":
            yawn_score = 85.0
            triggers.append("YAWN_CONFIRMED")
        elif yolo_yawning and (mar_val is not None and mar_val > 0.55):
            # Candidate mouth opening under 2 seconds: likely speech/laughter
            yawn_score = 15.0
        else:
            yawn_score = 0.0

        # ----------------------------------------------------------------------
        # 6. 3D Head Pose Nodding & Orientation Component
        # ----------------------------------------------------------------------
        head_pitch = getattr(signal, "pitch", None)
        head_state = getattr(signal, "head_state", "HEAD_FORWARD")
        if head_pitch is not None:
            conf_signals.append(1.0)
            if head_state == "HEAD_DOWN" or head_pitch > 15.0:
                head_pose_score = 75.0
                triggers.append(f"HEAD_NOD(pitch={head_pitch:.1f}d)")
            elif head_state in ("HEAD_LEFT", "HEAD_RIGHT"):
                head_pose_score = 40.0
                triggers.append(f"HEAD_DEVIATION({head_state})")
            else:
                head_pose_score = 0.0
        else:
            conf_signals.append(0.5)
            head_pose_score = 0.0

        # ----------------------------------------------------------------------
        # 7. Total Multi-Signal Fatigue Score (0 - 100)
        # ----------------------------------------------------------------------
        total_score = (
            self.weights["eye_closure"] * eye_score
            + self.weights["perclos"] * perclos_score
            + self.weights["blink_anomaly"] * blink_score
            + self.weights["yawn_duration"] * yawn_score
            + self.weights["head_pose"] * head_pose_score
        )
        total_score = float(np.clip(total_score, 0.0, 100.0))
        self.last_fatigue_score = total_score
        mean_conf = float(np.mean(conf_signals)) if conf_signals else 1.0

        # ----------------------------------------------------------------------
        # 8. Temporal State Machine Transitions
        # ----------------------------------------------------------------------
        is_critical_evidence = (
            total_score >= self.critical_thresh
            or eye_score >= 80.0
            or (perclos_score >= 60.0 and eye_score >= 50.0)
            or blink_score >= 90.0
        )
        is_warning_evidence = (
            total_score >= self.warning_thresh
            or eye_score >= 40.0
            or perclos_score >= 40.0
            or yawn_score >= 70.0
            or blink_score >= 60.0
            or head_pose_score >= 60.0
        )

        alert_level = "NONE"

        if self.state == "NORMAL":
            if is_critical_evidence:
                self.critical_candidate_frames += 1
                if self.critical_candidate_frames >= self.critical_persistence_frames:
                    self.state = "CRITICAL"
                    self.critical_candidate_frames = 0
            elif is_warning_evidence:
                self.warning_candidate_frames += 1
                if self.warning_candidate_frames >= self.warning_persistence_frames:
                    self.state = "WARNING"
                    self.warning_candidate_frames = 0
            else:
                self.warning_candidate_frames = max(0, self.warning_candidate_frames - 1)
                self.critical_candidate_frames = max(0, self.critical_candidate_frames - 1)

        elif self.state == "WARNING":
            if is_critical_evidence:
                self.critical_candidate_frames += 1
                if self.critical_candidate_frames >= self.critical_persistence_frames:
                    self.state = "CRITICAL"
                    self.critical_candidate_frames = 0
            elif not is_warning_evidence:
                self.state = "RECOVERY"
                self.recovery_frames = 0

        elif self.state == "CRITICAL":
            if not is_critical_evidence and not is_warning_evidence:
                self.state = "RECOVERY"
                self.recovery_frames = 0

        elif self.state == "RECOVERY":
            self.recovery_frames += 1
            if is_critical_evidence:
                self.state = "CRITICAL"
            elif is_warning_evidence:
                self.state = "WARNING"
            elif self.recovery_frames >= self.recovery_persistence_frames:
                self.state = "NORMAL"
                self.recovery_frames = 0

        # Determine alert level and trigger active alert
        if self.state == "CRITICAL":
            alert_level = "CRITICAL"
            if self.alert_manager:
                if self.alert_manager.trigger_alert("CRITICAL", timestamp):
                    self.total_alerts["CRITICAL"] += 1
        elif self.state == "WARNING":
            alert_level = "WARNING"
            if self.alert_manager:
                if self.alert_manager.trigger_alert("WARNING", timestamp):
                    self.total_alerts["WARNING"] += 1

        breakdown = FatigueScoreBreakdown(
            total_score=total_score,
            eye_score=eye_score,
            perclos_score=perclos_score,
            blink_score=blink_score,
            yawn_score=yawn_score,
            head_pose_score=head_pose_score,
            driver_state=self.state,
            alert_level=alert_level,
            trigger_signals=triggers,
            confidence=mean_conf,
        )

        if self.alert_manager:
            self.alert_manager.log_frame_event(
                timestamp=timestamp,
                frame_id=frame_id,
                breakdown=breakdown,
                signal=signal,
                yolo_eyes_closed=yolo_eyes_closed,
                yolo_eyes_open=yolo_eyes_open,
                yolo_yawning=yolo_yawning,
                yolo_conf=yolo_conf,
            )

        return alert_level, breakdown

    def reset(self) -> None:
        self.state = "NORMAL"
        self.warning_candidate_frames = 0
        self.critical_candidate_frames = 0
        self.recovery_frames = 0
        self.face_loss_frames = 0
        self.eye_closure_frames = 0
        self.last_fatigue_score = 0.0
        self.total_alerts = {"CRITICAL": 0, "WARNING": 0}


# ==============================================================================
# 4. DROWSINESS ENGINE (Unified Adapter Preserving Exact Legacy Regression)
# ==============================================================================

class DrowsinessEngine:
    """
    State machine for determining drowsiness and fatigue alerts.
    Maintains 100% backward compatibility with Phase 2A baseline tests,
    while optionally hosting the Phase 2G MultiSignalFusionEngine.
    """

    def __init__(
        self,
        fps: float = 30.0,
        eye_closure_threshold_seconds: float = 0.70,
        yawn_threshold_seconds: float = 0.50,
        critical_cooldown_seconds: float = 3.00,
        warning_cooldown_seconds: Optional[float] = None,
        fusion_engine: Optional[MultiSignalFusionEngine] = None,
    ):
        self.fps = max(1.0, float(fps))
        self.eye_closed_frames: int = 0
        self.yawn_frames: int = 0
        self.alert_cooldown: int = 0

        # Frame threshold conversions matching baseline implementation
        self.eye_threshold: int = int(eye_closure_threshold_seconds * self.fps)
        self.yawn_threshold: int = int(yawn_threshold_seconds * self.fps)
        self.cooldown_duration: int = int(critical_cooldown_seconds * self.fps)

        if warning_cooldown_seconds is not None:
            self.warning_cooldown: int = int(warning_cooldown_seconds * self.fps)
        else:
            self.warning_cooldown: int = self.cooldown_duration // 2

        self.total_alerts: Dict[str, int] = {"CRITICAL": 0, "WARNING": 0}
        self.frame_history: deque = deque(maxlen=int(30 * self.fps))
        self.fusion_engine = fusion_engine

    def update(self, eyes_closed: bool, eyes_open: bool, yawning: bool) -> str:
        """
        Legacy frame-counter state machine update (exact Phase 2A baseline behavior).
        """
        # Eye closure counter update & decay
        if eyes_closed:
            self.eye_closed_frames += 1
        elif eyes_open:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 3)
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 1)

        # Yawning counter update & decay
        if yawning:
            self.yawn_frames += 1
        else:
            self.yawn_frames = max(0, self.yawn_frames - 2)

        # Decrement cooldown counter if active
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1

        # Determine alert level
        alert_level = "NONE"
        if self.alert_cooldown == 0:
            if self.eye_closed_frames > self.eye_threshold:
                alert_level = "CRITICAL"
                self.alert_cooldown = self.cooldown_duration
                self.total_alerts["CRITICAL"] += 1
            elif self.yawn_frames > self.yawn_threshold:
                alert_level = "WARNING"
                self.alert_cooldown = self.warning_cooldown
                self.total_alerts["WARNING"] += 1

        self.frame_history.append({
            "eyes_closed": eyes_closed,
            "eyes_open": eyes_open,
            "yawning": yawning,
            "alert": alert_level,
        })

        return alert_level

    def update_fusion(
        self,
        signal: Optional[Any],
        yolo_eyes_closed: bool,
        yolo_eyes_open: bool,
        yolo_yawning: bool,
        yolo_conf: float = 1.0,
        timestamp: float = 0.0,
        frame_id: int = 0,
    ) -> Tuple[str, FatigueScoreBreakdown]:
        """
        Multi-signal temporal fusion update.
        """
        # Maintain legacy counters in parallel for telemetry consistency
        self.update(yolo_eyes_closed, yolo_eyes_open, yolo_yawning)

        if self.fusion_engine is not None:
            alert_level, breakdown = self.fusion_engine.update(
                signal=signal,
                yolo_eyes_closed=yolo_eyes_closed,
                yolo_eyes_open=yolo_eyes_open,
                yolo_yawning=yolo_yawning,
                yolo_conf=yolo_conf,
                timestamp=timestamp,
                frame_id=frame_id,
            )
            return alert_level, breakdown

        # Fallback if fusion engine is not configured
        breakdown = FatigueScoreBreakdown(
            total_score=float(min(100, self.eye_closed_frames * 2)),
            eye_score=float(min(100, self.eye_closed_frames * 2)),
            perclos_score=0.0,
            blink_score=0.0,
            yawn_score=float(min(100, self.yawn_frames * 3)),
            head_pose_score=0.0,
            driver_state="NORMAL" if self.eye_closed_frames < self.eye_threshold else "CRITICAL",
            alert_level=self.frame_history[-1]["alert"] if self.frame_history else "NONE",
        )
        return breakdown.alert_level, breakdown

    def get_stats(self) -> Dict[str, Any]:
        """Return the current metrics snapshot."""
        stats = {
            "eye_closed_frames": self.eye_closed_frames,
            "yawn_frames": self.yawn_frames,
            "total_alerts": sum(self.total_alerts.values()),
            "critical_alerts": self.total_alerts["CRITICAL"],
            "warning_alerts": self.total_alerts["WARNING"],
        }
        if self.fusion_engine is not None:
            stats["fusion_state"] = self.fusion_engine.state
            stats["fatigue_score"] = self.fusion_engine.last_fatigue_score
            stats["fusion_critical_alerts"] = self.fusion_engine.total_alerts.get("CRITICAL", 0)
            stats["fusion_warning_alerts"] = self.fusion_engine.total_alerts.get("WARNING", 0)
        return stats

    def reset(self) -> None:
        """Reset all internal counters and cooldown states."""
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.alert_cooldown = 0
        self.total_alerts = {"CRITICAL": 0, "WARNING": 0}
        self.frame_history.clear()
        if self.fusion_engine is not None:
            self.fusion_engine.reset()


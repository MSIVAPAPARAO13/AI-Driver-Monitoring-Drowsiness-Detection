"""
Unit tests for Phase 2G Multi-Signal Fusion Engine, Fatigue Scoring,
Temporal State Machine, and Alert Management.
"""

from pathlib import Path
import numpy as np
import pytest

from src.config import AppConfig
from src.drowsiness_engine import (
    AlertManager,
    DrowsinessEngine,
    FatigueScoreBreakdown,
    MultiSignalFusionEngine,
)
from src.physiological import TemporalSignal


def create_dummy_signal(
    face_detected: bool = True,
    ear: float = 0.30,
    mar: float = 0.20,
    perclos: float = 0.05,
    blink_event: str = None,
    blink_duration: float = 0.20,
    yawn_candidate: bool = False,
    yawn_duration: float = 0.0,
    yawn_event: str = None,
    pitch: float = 0.0,
    head_state: str = "HEAD_FORWARD",
) -> TemporalSignal:
    return TemporalSignal(
        timestamp=0.0,
        frame_index=1,
        fps=30.0,
        face_detected=face_detected,
        left_ear=ear,
        right_ear=ear,
        average_ear=ear,
        eye_closed=(ear < 0.21),
        blink_event=blink_event,
        blink_duration=blink_duration,
        blink_count=5,
        mar=mar,
        yawn_candidate=yawn_candidate,
        yawn_duration=yawn_duration,
        yawn_event=yawn_event,
        yawn_count=1,
        perclos=perclos,
        yaw=0.0,
        pitch=pitch,
        roll=0.0,
        head_state=head_state,
    )


def test_normal_alert_driver_low_fatigue_score():
    """Verify an alert driver has near-zero fatigue score and NORMAL state."""
    engine = MultiSignalFusionEngine(fps=30.0)
    sig = create_dummy_signal(ear=0.32, mar=0.20, perclos=0.02)
    alert, breakdown = engine.update(
        signal=sig,
        yolo_eyes_closed=False,
        yolo_eyes_open=True,
        yolo_yawning=False,
        yolo_conf=0.95,
        timestamp=1.0,
        frame_id=1,
    )
    assert alert == "NONE"
    assert breakdown.driver_state == "NORMAL"
    assert breakdown.total_score < 25.0
    assert breakdown.eye_score == 0.0


def test_speech_suppression_vs_true_yawn():
    """Verify mouth opening <1.5s (speech) is suppressed while >=2.0s triggers confirmed yawn."""
    engine = MultiSignalFusionEngine(fps=30.0)

    # 1. Brief mouth opening (e.g. speech / laugh, 0.8s) -> suppressed
    speech_sig = create_dummy_signal(
        ear=0.30,
        mar=0.60,
        yawn_candidate=True,
        yawn_duration=0.8,
        yawn_event=None,
    )
    alert, breakdown = engine.update(
        signal=speech_sig,
        yolo_eyes_closed=False,
        yolo_eyes_open=True,
        yolo_yawning=True,
        yolo_conf=0.80,
        timestamp=1.0,
        frame_id=1,
    )
    assert breakdown.yawn_score <= 15.0
    assert alert == "NONE"

    # 2. Sustained mouth opening (2.5s true physiological yawn) -> triggers yawn score & warning
    yawn_sig = create_dummy_signal(
        ear=0.30,
        mar=0.65,
        yawn_candidate=True,
        yawn_duration=2.5,
        yawn_event="YAWN_CONFIRMED",
    )
    for i in range(35):  # >1.0s warning persistence
        alert, breakdown = engine.update(
            signal=yawn_sig,
            yolo_eyes_closed=False,
            yolo_eyes_open=True,
            yolo_yawning=True,
            yolo_conf=0.90,
            timestamp=2.0 + i / 30.0,
            frame_id=2 + i,
        )
    assert breakdown.yawn_score >= 75.0
    assert breakdown.driver_state in ("WARNING", "CRITICAL")
    assert alert == "WARNING"


def test_micro_sleep_prolonged_closure():
    """Verify prolonged eye closure (low EAR + YOLO eyes_closed) triggers CRITICAL."""
    engine = MultiSignalFusionEngine(fps=30.0, critical_persistence_seconds=1.0)
    closed_sig = create_dummy_signal(
        ear=0.12,
        mar=0.20,
        perclos=0.35,
        blink_event="PROLONGED_CLOSURE",
        blink_duration=1.2,
    )

    # Feed frames to satisfy critical persistence (30 frames @ 30 FPS = 1.0s)
    last_alert = "NONE"
    for i in range(50):
        last_alert, breakdown = engine.update(
            signal=closed_sig,
            yolo_eyes_closed=True,
            yolo_eyes_open=False,
            yolo_yawning=False,
            yolo_conf=0.95,
            timestamp=i / 30.0,
            frame_id=i,
        )

    assert breakdown.driver_state == "CRITICAL"
    assert breakdown.total_score >= 70.0
    assert last_alert == "CRITICAL"


def test_face_loss_grace_and_timeout():
    """Verify face disappearance enters FACE_LOST without raising false alarms."""
    engine = MultiSignalFusionEngine(fps=30.0, face_loss_timeout_seconds=1.0)

    # Normal frame first
    engine.update(create_dummy_signal(), False, True, False, 1.0, 0.0, 1)

    # Face disappears
    for i in range(25):  # <1.0s grace period
        alert, breakdown = engine.update(None, False, False, False, 0.0, i / 30.0, i + 2)
        assert alert == "NONE"
        assert breakdown.driver_state != "CRITICAL"

    # Beyond timeout (40 frames = 1.33s > 1.0s)
    for i in range(25, 45):
        alert, breakdown = engine.update(None, False, False, False, 0.0, i / 30.0, i + 2)
    assert breakdown.driver_state == "FACE_LOST"
    assert alert == "NONE"


def test_recovery_state_transition():
    """Verify state machine transitions from CRITICAL -> RECOVERY -> NORMAL when signals recover."""
    engine = MultiSignalFusionEngine(fps=30.0, critical_persistence_seconds=0.5, recovery_seconds=0.5)

    # 1. Drive into CRITICAL (requires >0.4s eye closure ramp + 15 frames persistence)
    closed_sig = create_dummy_signal(ear=0.10, perclos=0.40)
    for i in range(35):
        engine.update(closed_sig, True, False, False, 0.95, i / 30.0, i)
    assert engine.state == "CRITICAL"

    # 2. Driver recovers (eyes open, low perclos)
    open_sig = create_dummy_signal(ear=0.32, perclos=0.05)
    alert, breakdown = engine.update(open_sig, False, True, False, 0.95, 1.0, 21)
    assert breakdown.driver_state == "RECOVERY"

    # 3. Wait out recovery period (15 frames = 0.5s)
    for i in range(16):
        alert, breakdown = engine.update(open_sig, False, True, False, 0.95, 1.0 + i / 30.0, 22 + i)
    assert breakdown.driver_state == "NORMAL"


def test_alert_manager_cooldown_and_logging(tmp_path):
    """Verify AlertManager cooldown prevents continuous beeping and logs to CSV."""
    log_csv = tmp_path / "test_events.csv"
    mgr = AlertManager(
        mode="visual_only",  # Don't play audio in unit test
        cooldown_seconds=2.0,
        log_events=True,
        event_log_path=log_csv,
    )

    # First critical alert triggers
    assert mgr.trigger_alert("CRITICAL", timestamp=10.0) is True

    # Immediate second alert at t=10.5s is suppressed by cooldown
    assert mgr.trigger_alert("CRITICAL", timestamp=10.5) is False

    # Alert at t=12.1s triggers
    assert mgr.trigger_alert("CRITICAL", timestamp=12.1) is True

    # Test logging
    breakdown = FatigueScoreBreakdown(
        total_score=80.0,
        eye_score=90.0,
        perclos_score=75.0,
        blink_score=60.0,
        yawn_score=0.0,
        head_pose_score=0.0,
        driver_state="CRITICAL",
        alert_level="CRITICAL",
        trigger_signals=["EYE_CLOSURE"],
    )
    mgr.log_frame_event(12.1, 100, breakdown, create_dummy_signal())
    mgr.close()

    assert log_csv.exists()
    content = log_csv.read_text(encoding="utf-8")
    assert "CRITICAL" in content
    assert "EYE_CLOSURE" in content

"""
Unit tests for the DrowsinessEngine state machine and alert transitions.
"""

import pytest
from src.drowsiness_engine import DrowsinessEngine


def test_eye_counter_increment_and_decay():
    """Verify eye closure counter increments and decays properly."""
    engine = DrowsinessEngine(fps=30.0)

    # 1. Closed eyes increment by 1
    engine.update(eyes_closed=True, eyes_open=False, yawning=False)
    engine.update(eyes_closed=True, eyes_open=False, yawning=False)
    assert engine.eye_closed_frames == 2

    # 2. Open eyes decay by 3 (capped at 0)
    engine.update(eyes_closed=False, eyes_open=True, yawning=False)
    assert engine.eye_closed_frames == 0

    # 3. Neither detected decays by 1
    for _ in range(5):
        engine.update(eyes_closed=True, eyes_open=False, yawning=False)
    assert engine.eye_closed_frames == 5

    engine.update(eyes_closed=False, eyes_open=False, yawning=False)
    assert engine.eye_closed_frames == 4


def test_yawn_counter_increment_and_decay():
    """Verify yawning counter increments and decays by 2."""
    engine = DrowsinessEngine(fps=30.0)

    # Yawn increments by 1
    for _ in range(4):
        engine.update(eyes_closed=False, eyes_open=True, yawning=True)
    assert engine.yawn_frames == 4

    # Non-yawn decays by 2
    engine.update(eyes_closed=False, eyes_open=True, yawning=False)
    assert engine.yawn_frames == 2


def test_critical_alert_transition_and_cooldown():
    """
    At 30 FPS with 0.70s threshold:
      eye_threshold = int(0.70 * 30) = 21 frames.
      Frame 21 -> no alert (21 is not > 21)
      Frame 22 -> CRITICAL alert triggered, cooldown set to 90 frames.
    """
    engine = DrowsinessEngine(fps=30.0, eye_closure_threshold_seconds=0.70)
    assert engine.eye_threshold == 21
    assert engine.cooldown_duration == 90

    # Feed 21 frames of closed eyes
    for i in range(21):
        alert = engine.update(eyes_closed=True, eyes_open=False, yawning=False)
        assert alert == "NONE"
        assert engine.total_alerts["CRITICAL"] == 0

    # Feed 22nd frame -> triggers CRITICAL
    alert = engine.update(eyes_closed=True, eyes_open=False, yawning=False)
    assert alert == "CRITICAL"
    assert engine.total_alerts["CRITICAL"] == 1
    assert engine.alert_cooldown == 90

    # 23rd frame: cooldown is active -> alert should be NONE even if still closed
    alert = engine.update(eyes_closed=True, eyes_open=False, yawning=False)
    assert alert == "NONE"
    assert engine.alert_cooldown == 89  # Cooldown decremented by 1


def test_warning_alert_transition():
    """
    At 30 FPS with 0.50s threshold:
      yawn_threshold = int(0.50 * 30) = 15 frames.
      Frame 16 -> WARNING alert triggered.
    """
    engine = DrowsinessEngine(fps=30.0, yawn_threshold_seconds=0.50)
    assert engine.yawn_threshold == 15

    for _ in range(15):
        alert = engine.update(eyes_closed=False, eyes_open=True, yawning=True)
        assert alert == "NONE"

    alert = engine.update(eyes_closed=False, eyes_open=True, yawning=True)
    assert alert == "WARNING"
    assert engine.total_alerts["WARNING"] == 1
    assert engine.alert_cooldown == engine.warning_cooldown


def test_engine_reset():
    """Verify reset clears all metrics and counters."""
    engine = DrowsinessEngine(fps=30.0)
    for _ in range(25):
        engine.update(eyes_closed=True, eyes_open=False, yawning=False)

    assert engine.eye_closed_frames > 0
    assert engine.total_alerts["CRITICAL"] > 0

    engine.reset()
    assert engine.eye_closed_frames == 0
    assert engine.yawn_frames == 0
    assert engine.alert_cooldown == 0
    assert engine.total_alerts["CRITICAL"] == 0
    assert engine.total_alerts["WARNING"] == 0
    assert len(engine.frame_history) == 0

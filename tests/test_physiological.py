"""
Unit tests for Phase 2B Physiological Signals and Temporal Analysis.

Tests EAR, MAR, BlinkAnalyzer, YawnDurationAnalyzer, PERCLOSAnalyzer,
HeadPoseEstimator, TemporalSignal, and TelemetryLogger under normal and edge conditions.
"""

from pathlib import Path
import numpy as np
import pytest

from src.physiological import (
    BlinkAnalyzer,
    BlinkEvent,
    EARCalculator,
    EARResult,
    HeadPoseEstimator,
    HeadPoseResult,
    MARCalculator,
    MARResult,
    PERCLOSAnalyzer,
    PERCLOSResult,
    TelemetryLogger,
    TemporalSignal,
    YawnDurationAnalyzer,
    YawnStatus,
)


# -----------------------------------------------------------------------------
# 1. EAR Calculator Tests
# -----------------------------------------------------------------------------

def test_ear_calculation_synthetic():
    """Verify EAR calculation with synthetic coordinate geometry."""
    calculator = EARCalculator(threshold=0.21)

    # Construct synthetic 478-point landmark array
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Set left eye coordinates:
    # p1(33) = (0, 0), p4(133) = (10, 0) -> width = 10
    # p2(160) = (3, 3), p6(144) = (3, -3) -> d_v1 = 6
    # p3(158) = (7, 3), p5(153) = (7, -3) -> d_v2 = 6
    # EAR = (6 + 6) / (2 * 10) = 0.60 (Open eye)
    landmarks[33] = [0.0, 0.0, 0.0]
    landmarks[133] = [10.0, 0.0, 0.0]
    landmarks[160] = [3.0, 3.0, 0.0]
    landmarks[144] = [3.0, -3.0, 0.0]
    landmarks[158] = [7.0, 3.0, 0.0]
    landmarks[153] = [7.0, -3.0, 0.0]

    # Mirror for right eye
    landmarks[362] = [20.0, 0.0, 0.0]
    landmarks[263] = [30.0, 0.0, 0.0]
    landmarks[385] = [23.0, 3.0, 0.0]
    landmarks[380] = [23.0, -3.0, 0.0]
    landmarks[387] = [27.0, 3.0, 0.0]
    landmarks[373] = [27.0, -3.0, 0.0]

    res = calculator.calculate(landmarks)
    assert res.is_valid is True
    assert pytest.approx(res.left_ear, 0.01) == 0.60
    assert pytest.approx(res.right_ear, 0.01) == 0.60
    assert pytest.approx(res.average_ear, 0.01) == 0.60
    assert res.is_closed is False

    # Simulate closed eye by flattening vertical distances to 0.5
    landmarks[160] = [3.0, 0.5, 0.0]
    landmarks[144] = [3.0, -0.5, 0.0]  # d_v1 = 1.0
    landmarks[158] = [7.0, 0.5, 0.0]
    landmarks[153] = [7.0, -0.5, 0.0]  # d_v2 = 1.0
    # Left EAR = (1 + 1) / 20 = 0.10
    landmarks[385] = [23.0, 0.5, 0.0]
    landmarks[380] = [23.0, -0.5, 0.0]
    landmarks[387] = [27.0, 0.5, 0.0]
    landmarks[373] = [27.0, -0.5, 0.0]

    res_closed = calculator.calculate(landmarks)
    assert pytest.approx(res_closed.average_ear, 0.01) == 0.10
    assert res_closed.is_closed is True


def test_ear_edge_cases_no_face_or_empty():
    """Verify EAR calculation handles None or missing points gracefully."""
    calculator = EARCalculator()
    assert calculator.calculate(None).is_valid is False
    assert calculator.calculate(np.array([])).is_valid is False
    # Insufficient points (< 478)
    assert calculator.calculate(np.zeros((10, 3))).is_valid is False


# -----------------------------------------------------------------------------
# 2. MAR Calculator Tests
# -----------------------------------------------------------------------------

def test_mar_calculation_synthetic():
    """Verify MAR calculation with synthetic mouth coordinates."""
    calculator = MARCalculator(threshold=0.55)

    landmarks = np.zeros((478, 3), dtype=np.float32)
    # Corners: 78 -> (0, 0), 308 -> (20, 0), width = 20
    landmarks[78] = [0.0, 0.0, 0.0]
    landmarks[308] = [20.0, 0.0, 0.0]

    # Vertical pairs: (82, 87), (13, 14), (312, 317)
    # Closed mouth: vertical distance = 1.0 each -> sum = 3.0
    # MAR = 3.0 / (2 * 20) = 0.075 (< 0.55)
    for u, l in [(82, 87), (13, 14), (312, 317)]:
        landmarks[u] = [10.0, 0.5, 0.0]
        landmarks[l] = [10.0, -0.5, 0.0]

    res_closed = calculator.calculate(landmarks)
    assert res_closed.is_valid is True
    assert pytest.approx(res_closed.mar, 0.01) == 0.075
    assert res_closed.is_open is False

    # Wide open yawn: vertical distance = 10.0 each -> sum = 30.0
    # MAR = 30.0 / 40.0 = 0.75 (> 0.55)
    for u, l in [(82, 87), (13, 14), (312, 317)]:
        landmarks[u] = [10.0, 5.0, 0.0]
        landmarks[l] = [10.0, -5.0, 0.0]

    res_open = calculator.calculate(landmarks)
    assert pytest.approx(res_open.mar, 0.01) == 0.75
    assert res_open.is_open is True


def test_mar_edge_cases():
    """Verify MAR handles None and invalid input gracefully."""
    calculator = MARCalculator()
    assert calculator.calculate(None).is_valid is False
    assert calculator.calculate(np.zeros((50, 3))).is_valid is False


# -----------------------------------------------------------------------------
# 3. Blink Analyzer Tests
# -----------------------------------------------------------------------------

def test_blink_analyzer_normal_blink():
    """Verify recognition of a normal physiological blink (150 ms)."""
    analyzer = BlinkAnalyzer(normal_blink_min_s=0.08, normal_blink_max_s=0.40)

    # Eyes open at t = 0.0
    ev1 = analyzer.update(is_closed=False, timestamp=0.0)
    assert ev1 is None

    # Eyes close at t = 0.10
    ev2 = analyzer.update(is_closed=True, timestamp=0.10)
    assert ev2 is None
    assert analyzer.is_eye_closed is True

    # Eyes open at t = 0.25 (duration = 150 ms)
    ev3 = analyzer.update(is_closed=False, timestamp=0.25)
    assert ev3 is not None
    assert ev3.event_type == "NORMAL_BLINK"
    assert pytest.approx(ev3.duration, 0.01) == 0.15
    assert analyzer.total_blink_count == 1


def test_blink_analyzer_prolonged_closure():
    """Verify recognition of a prolonged closure exceeding long_blink_max_s."""
    analyzer = BlinkAnalyzer(long_blink_max_s=0.80)

    # Close eyes at t = 1.0
    analyzer.update(is_closed=True, timestamp=1.0)
    # Still closed at t = 1.5 (duration 0.5s -> no event yet)
    ev_mid = analyzer.update(is_closed=True, timestamp=1.5)
    assert ev_mid is None

    # Exceeds 0.80s at t = 1.9 (duration 0.9s -> PROLONGED_CLOSURE triggered)
    ev_prolonged = analyzer.update(is_closed=True, timestamp=1.9)
    assert ev_prolonged is not None
    assert ev_prolonged.event_type == "PROLONGED_CLOSURE"
    assert analyzer.prolonged_closure_flag is True


# -----------------------------------------------------------------------------
# 4. Yawn Duration Analyzer Tests
# -----------------------------------------------------------------------------

def test_yawn_duration_analyzer():
    """Verify yawn candidate initiation, sustained duration confirmation, and completion."""
    analyzer = YawnDurationAnalyzer(min_duration_s=2.0, cooldown_s=3.0)

    # t = 0.0: Mouth opens -> YAWN_START
    st1 = analyzer.update(mar_is_open=True, yolo_yawning=False, timestamp=0.0)
    assert st1.is_candidate is True
    assert st1.event == "YAWN_START"
    assert st1.is_confirmed is False

    # t = 1.0: 1.0s elapsed -> Not yet confirmed
    st2 = analyzer.update(mar_is_open=True, yolo_yawning=False, timestamp=1.0)
    assert st2.is_candidate is True
    assert st2.is_confirmed is False
    assert st2.event is None

    # t = 2.1: 2.1s elapsed -> YAWN_CONFIRMED
    st3 = analyzer.update(mar_is_open=True, yolo_yawning=False, timestamp=2.1)
    assert st3.is_confirmed is True
    assert st3.event == "YAWN_CONFIRMED"
    assert analyzer.total_yawns == 1

    # t = 3.5: Mouth closes -> YAWN_END
    st4 = analyzer.update(mar_is_open=False, yolo_yawning=False, timestamp=3.5)
    assert st4.is_candidate is False
    assert st4.event == "YAWN_END"


# -----------------------------------------------------------------------------
# 5. PERCLOS Analyzer Tests
# -----------------------------------------------------------------------------

def test_perclos_analyzer_sliding_window():
    """Verify PERCLOS time-weighted calculation and sliding window eviction."""
    analyzer = PERCLOSAnalyzer(window_seconds=10.0, minimum_samples=5)

    # Feed 10 seconds of data at 1s intervals:
    # 5 seconds closed, 5 seconds open -> PERCLOS ~ 50%
    for t in range(5):
        analyzer.update(is_closed=True, timestamp=float(t))
    for t in range(5, 11):
        analyzer.update(is_closed=False, timestamp=float(t))

    res = analyzer.update(is_closed=False, timestamp=11.0)
    assert res.sample_count > 0
    # Eyes were closed for roughly half the window
    assert 0.35 <= res.perclos <= 0.65

    # Move forward in time by 15 seconds with eyes fully open:
    # All closed samples should be evicted from the 10s window
    for t in range(12, 25):
        res = analyzer.update(is_closed=False, timestamp=float(t))

    assert res.perclos == 0.0


# -----------------------------------------------------------------------------
# 6. Head Pose Estimator Tests
# -----------------------------------------------------------------------------

def test_head_pose_classification():
    """Verify classification of head orientation states."""
    estimator = HeadPoseEstimator(yaw_threshold_deg=20.0, pitch_threshold_deg=15.0)

    # Identity rotation matrix -> HEAD_FORWARD
    I_mat = np.eye(4, dtype=np.float32)
    res_fwd = estimator.estimate(landmarks=None, transformation_matrix=I_mat, timestamp=1.0)
    assert res_fwd.is_valid is True
    assert res_fwd.state == "HEAD_FORWARD"
    assert pytest.approx(res_fwd.yaw, 0.1) == 0.0

    # Yaw right: rotate by +30 deg around Y axis
    theta = np.radians(30.0)
    R_yaw_right = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    res_right = estimator.estimate(landmarks=None, transformation_matrix=R_yaw_right, timestamp=2.0)
    assert res_right.state == "HEAD_RIGHT"


# -----------------------------------------------------------------------------
# 7. Telemetry Logger & Signal Tests
# -----------------------------------------------------------------------------

def test_telemetry_logger_csv(tmp_path):
    """Verify TelemetryLogger writes CSV headers and formatted data rows."""
    log_file = tmp_path / "test_telemetry.csv"
    logger = TelemetryLogger(output_path=log_file, format_type="csv")

    signal = TemporalSignal(
        timestamp=1.234,
        frame_index=42,
        fps=30.0,
        face_detected=True,
        left_ear=0.28,
        right_ear=0.29,
        average_ear=0.285,
        eye_closed=False,
        blink_event="NORMAL_BLINK",
        blink_duration=0.18,
        blink_count=3,
        mar=0.12,
        yawn_candidate=False,
        yawn_duration=0.0,
        yawn_event=None,
        yawn_count=1,
        perclos=0.05,
        yaw=2.5,
        pitch=-1.2,
        roll=0.4,
        head_state="HEAD_FORWARD",
    )

    logger.log(signal)
    logger.close()

    content = log_file.read_text(encoding="utf-8")
    assert "timestamp,frame,fps,face_detected" in content
    assert "1.234,42,30.00,1" in content
    assert "NORMAL_BLINK" in content
    assert "HEAD_FORWARD" in content

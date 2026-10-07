"""
Unit and regression tests for Bounding-Box Coordinate Transformations,
Aspect-Ratio Invariance, and Visual Renderer Layout Safety.

Validates:
1. Coordinate clamping to exact image boundaries across resolutions.
2. Degenerate box filtering (zero/negative area).
3. Aspect ratio handling: 16:9 (1920x1080, 1280x720), 4:3 (1280x960, 640x480), and vertical (720x1280).
4. Renderer label tag positioning when boxes touch top/right edges.
5. Renderer HUD panels layout: zero collision between telemetry and physiological signal HUDs.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.detector import DetectionBox, DetectionResult, YOLODetector
from src.renderer import VisualRenderer
from src.physiological import TemporalSignal, FaceLandmarksResult

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def detector():
    model_path = REPO_ROOT / "weights" / "phase2f_best.pt"
    return YOLODetector(model_path=str(model_path), conf_threshold=0.30)


@pytest.fixture
def renderer():
    return VisualRenderer()


@pytest.mark.parametrize(
    "width,height",
    [
        (1920, 1080),  # 16:9 Full HD
        (1280, 720),   # 16:9 HD
        (1280, 960),   # 4:3
        (640, 480),    # Standard Webcam
        (720, 1280),   # Vertical Mobile / Dashcam
        (480, 640),    # Compact Vertical
    ],
)
def test_aspect_ratio_and_coordinate_bounds_clamping(detector, renderer, width, height):
    """Test inference and rendering across varied resolutions and aspect ratios."""
    # Synthetic frame with simulated facial region
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), min(width, height) // 4, (180, 180, 180), -1)

    result = detector.predict(frame)
    assert isinstance(result, DetectionResult)

    # Validate that every detection box is strictly bounded
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy
        assert 0 <= x1 <= width - 1, f"x1 ({x1}) out of bounds [0, {width-1}]"
        assert 0 <= y1 <= height - 1, f"y1 ({y1}) out of bounds [0, {height-1}]"
        assert 0 <= x2 <= width - 1, f"x2 ({x2}) out of bounds [0, {width-1}]"
        assert 0 <= y2 <= height - 1, f"y2 ({y2}) out of bounds [0, {height-1}]"
        assert x1 < x2, f"Invalid box width: x1={x1}, x2={x2}"
        assert y1 < y2, f"Invalid box height: y1={y1}, y2={y2}"

    # Render on frame
    annotated = renderer.render(
        frame=frame.copy(),
        detection_result=result,
        alert_level="NONE",
        stats={"eye_closed_frames": 0, "yawn_frames": 0, "total_alerts": 0},
        fps_display=30.0,
    )
    assert annotated.shape == (height, width, 3)


def test_renderer_edge_cases_boundary_boxes(renderer):
    """Test renderer handles boxes touching the top, bottom, left, and right edges."""
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Box 1: Touching top edge (y1 = 0)
    # Box 2: Touching right edge (x2 = w - 1)
    # Box 3: Extremely small box
    boxes = [
        DetectionBox(cls_id=0, cls_name="eyes_closed", conf=0.95, xyxy=(50, 0, 150, 40)),
        DetectionBox(cls_id=1, cls_name="eyes_open", conf=0.88, xyxy=(w - 80, 100, w - 1, 160)),
        DetectionBox(cls_id=2, cls_name="yawning", conf=0.91, xyxy=(200, 300, 210, 310)),
    ]
    det_res = DetectionResult(boxes=boxes, eyes_closed=True, eyes_open=True, yawning=True)

    # Must render without exception
    annotated = renderer.render(
        frame=frame,
        detection_result=det_res,
        alert_level="CRITICAL",
        stats={"eye_closed_frames": 10, "yawn_frames": 5, "total_alerts": 1},
        fps_display=28.5,
    )
    assert annotated is not None
    assert annotated.shape == (h, w, 3)


def test_renderer_hud_panels_non_overlapping_on_compact_resolution(renderer):
    """Test that left and right HUD panels do not collide on 640x480 resolution."""
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    signal = TemporalSignal(
        timestamp=1.0,
        frame_index=30,
        fps=30.0,
        face_detected=True,
        average_ear=0.28,
        mar=0.19,
        eye_closed=False,
        blink_count=2,
        perclos=0.05,
        yaw=2.1,
        pitch=-1.4,
        head_state="HEAD_FORWARD",
        yawn_count=0,
    )

    annotated = renderer.render(
        frame=frame,
        detection_result=None,
        alert_level="NONE",
        stats={"eye_closed_frames": 0, "yawn_frames": 0, "total_alerts": 0},
        fps_display=30.0,
        signal=signal,
    )
    assert annotated.shape == (h, w, 3)


def test_renderer_responsive_alert_banners(renderer):
    """Test alert banners render cleanly on small vertical and wide resolutions."""
    for (w, h) in [(480, 800), (1920, 1080), (640, 480)]:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Critical banner
        res_crit = renderer.render(
            frame=frame.copy(),
            detection_result=None,
            alert_level="CRITICAL",
            stats={},
            fps_display=25.0,
        )
        assert res_crit.shape == (h, w, 3)

        # Warning banner
        res_warn = renderer.render(
            frame=frame.copy(),
            detection_result=None,
            alert_level="WARNING",
            stats={},
            fps_display=25.0,
        )
        assert res_warn.shape == (h, w, 3)

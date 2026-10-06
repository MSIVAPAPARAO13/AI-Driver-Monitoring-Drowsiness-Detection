"""
Unit tests for the InferencePipeline coordination.
"""

from typing import Optional, Tuple
import numpy as np
import pytest

from src.config import AppConfig
from src.detector import DetectionBox, DetectionResult
from src.input_sources import BaseInputSource
from src.pipeline import InferencePipeline


class MockInputSource(BaseInputSource):
    """Synthetic source producing a fixed number of black frames."""

    def __init__(self, num_frames: int = 10, fps: float = 30.0):
        self._num_frames = num_frames
        self._fps = fps
        self._current_frame = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._current_frame < self._num_frames:
            self._current_frame += 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, frame
        return False, None

    def release(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return 640

    @property
    def height(self) -> int:
        return 480

    @property
    def total_frames(self) -> int:
        return self._num_frames

    @property
    def is_opened(self) -> bool:
        return True


class MockDetector:
    """Mock detector returning predefined detection results."""

    def __init__(self):
        self.call_count = 0

    def predict(self, frame: np.ndarray) -> DetectionResult:
        self.call_count += 1
        box = DetectionBox(
            cls_id=0,
            cls_name="eyes_closed",
            conf=0.95,
            xyxy=(100, 100, 200, 200),
        )
        return DetectionResult(
            boxes=[box],
            eyes_closed=True,
            eyes_open=False,
            yawning=False,
        )


def test_pipeline_execution_and_frame_skipping():
    """Verify that the pipeline orchestrates processing and applies frame skipping."""
    mock_source = MockInputSource(num_frames=10, fps=30.0)
    mock_detector = MockDetector()

    cfg = AppConfig(
        skip_frames=2,
        save_output=False,
        display=False,
    )

    pipeline = InferencePipeline(
        config=cfg,
        input_source=mock_source,
        detector=mock_detector,
    )

    results = pipeline.run()

    assert results["frames_processed"] == 10
    # With skip_frames=2 over 10 frames:
    # Frame 1: last_results is None -> predict called (1)
    # Frame 2: 2 % 2 == 0 -> predict called (2)
    # Frame 3: skipped
    # Frame 4: predict called (3)
    # Frame 5: skipped
    # Frame 6: predict called (4)
    # Frame 7: skipped
    # Frame 8: predict called (5)
    # Frame 9: skipped
    # Frame 10: predict called (6)
    assert mock_detector.call_count == 6
    assert results["class_detections"][0] == 10  # 10 boxes counted across all 10 frames
    assert "avg_fps" in results
    assert "wall_fps" in results

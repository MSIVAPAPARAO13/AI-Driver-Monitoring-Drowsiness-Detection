"""
Input Sources Abstraction Module.

Provides a unified interface for streaming frames from video files and webcams.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class BaseInputSource(ABC):
    """Abstract base class for all frame ingestion sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the capture device or file handle."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Operating frame rate."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""
        pass

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Total number of frames available (-1 if indefinite/stream)."""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """True if the source is currently accessible."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoSource(BaseInputSource):
    """Ingestion source for local video files (.mp4, .avi, etc.)."""

    def __init__(self, file_path: Union[str, Path]):
        p = Path(file_path)
        repo_root = Path(__file__).resolve().parent.parent
        if not p.is_absolute() and not p.exists() and (repo_root / file_path).exists():
            p = repo_root / file_path
        self.file_path = str(p)
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.file_path}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps > 0 else 30.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()


class WebcamSource(BaseInputSource):
    """Ingestion source for physical webcams and USB cameras."""

    def __init__(self, device_id: int = 0, default_fps: float = 30.0):
        self.device_id = int(device_id)
        # Use DSHOW on Windows if available, fallback to default backend
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam device index {self.device_id}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps and raw_fps > 0 else default_fps
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = -1  # Indefinite live stream

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()


def get_input_source(source: Union[str, int]) -> BaseInputSource:
    """
    Factory function to instantiate the appropriate input source.

    Args:
        source: Device index (e.g. 0 or '0') or file path (e.g. 'video.mp4').
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return WebcamSource(device_id=int(source))
    return VideoSource(file_path=source)

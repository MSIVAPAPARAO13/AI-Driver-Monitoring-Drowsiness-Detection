"""
YOLO Model Detector Module.

Encapsulates object detection using Ultralytics YOLOv5nu without any
alert logic, rendering, or video I/O.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from src.config import CLASS_NAMES


@dataclass
class DetectionBox:
    """Represents a single detected bounding box."""
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


@dataclass
class DetectionResult:
    """Structured detection outcome for a single frame."""
    boxes: List[DetectionBox]
    eyes_closed: bool
    eyes_open: bool
    yawning: bool
    raw_results: Optional[object] = None


class YOLODetector:
    """Wrapper around Ultralytics YOLO model for facial state detection supporting multiple runtime backends."""

    def __init__(
        self,
        model_path: str = "best.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        max_detections: int = 3,
        device: Optional[str] = None,
        fuse: bool = True,
        class_names: Optional[Dict[int, str]] = None,
        backend: str = "pytorch",
    ):
        self.backend = backend.lower()
        repo_root = Path(__file__).resolve().parent.parent

        # Resolve model path based on chosen backend
        p = Path(model_path)
        if not p.is_absolute() and not p.exists() and (repo_root / model_path).exists():
            p = repo_root / model_path

        if self.backend == "onnx":
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                raise ImportError(
                    "ONNX Runtime backend requested, but 'onnxruntime' is not installed. "
                    "Install with: pip install onnxruntime"
                )
            if not p.suffix == ".onnx":
                # Look for corresponding .onnx file or deployment folder
                candidates = [
                    p.with_suffix(".onnx"),
                    repo_root / "weights" / "deployment" / "phase2f_best.onnx",
                    repo_root / "weights" / f"{p.stem}.onnx",
                ]
                resolved = None
                for c in candidates:
                    if c.exists():
                        resolved = c
                        break
                if resolved is not None:
                    p = resolved
                else:
                    raise FileNotFoundError(
                        f"ONNX backend requested but no .onnx model found for '{model_path}'. "
                        "Expected e.g. weights/deployment/phase2f_best.onnx"
                    )

        elif self.backend == "openvino":
            try:
                import openvino  # noqa: F401
            except ImportError:
                raise ImportError(
                    "OpenVINO backend requested, but 'openvino' is not installed. "
                    "Install with: pip install openvino"
                )
            if not (p.is_dir() or p.suffix == ".xml"):
                candidates = [
                    repo_root / "weights" / "deployment" / "phase2f_openvino_model",
                    repo_root / "weights" / f"{p.stem}_openvino_model",
                    p.parent / f"{p.stem}_openvino_model",
                ]
                resolved = None
                for c in candidates:
                    if c.exists():
                        resolved = c
                        break
                if resolved is not None:
                    p = resolved
                else:
                    raise FileNotFoundError(
                        f"OpenVINO backend requested but no OpenVINO IR model directory found for '{model_path}'. "
                        "Expected e.g. weights/deployment/phase2f_openvino_model"
                    )

        elif self.backend == "pytorch":
            if not p.exists():
                raise FileNotFoundError(f"PyTorch model file not found: {p}")

        else:
            raise ValueError(f"Unsupported backend '{backend}'. Supported backends: 'pytorch', 'onnx', 'openvino'")

        self.model_path = p
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.class_names = class_names or CLASS_NAMES

        # Load YOLO model
        if self.backend in ("onnx", "openvino"):
            self.model = YOLO(str(self.model_path), task="detect")
        else:
            self.model = YOLO(str(self.model_path))
            if fuse:
                try:
                    self.model.fuse()
                except Exception:
                    pass  # Fallback if fuse is not supported or already fused

    def reload_model(self, model_path: str) -> None:
        """Safely reload model from a new checkpoint path."""
        p = Path(model_path)
        if not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / model_path).exists():
                p = repo_root / model_path
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {p}")
        new_model = YOLO(str(p), task="detect" if self.backend in ("onnx", "openvino") else None)
        self.model = new_model
        self.model_path = p

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single BGR image/frame.
        
        Args:
            frame: Input frame as a numpy ndarray (H, W, C).
            
        Returns:
            DetectionResult containing parsed boxes and state booleans.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                boxes=[],
                eyes_closed=False,
                eyes_open=False,
                yawning=False,
                raw_results=None,
            )

        kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(source=frame, **kwargs)[0]

        h, w = frame.shape[:2]
        parsed_boxes: List[DetectionBox] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Coordinate bounds clamping & degenerate box filtering
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_name = self.class_names.get(cls_id, str(cls_id))

            parsed_boxes.append(
                DetectionBox(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=conf,
                    xyxy=(x1, y1, x2, y2),
                )
            )

        eyes_closed = any(b.cls_id == 0 for b in parsed_boxes)
        eyes_open = any(b.cls_id == 1 for b in parsed_boxes)
        yawning = any(b.cls_id == 2 for b in parsed_boxes)

        return DetectionResult(
            boxes=parsed_boxes,
            eyes_closed=eyes_closed,
            eyes_open=eyes_open,
            yawning=yawning,
            raw_results=results,
        )

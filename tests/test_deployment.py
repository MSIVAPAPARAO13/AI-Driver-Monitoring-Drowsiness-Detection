"""
Phase 2H - Deployment, Export, and Runtime Backend Unit Tests.
"""
import pytest
import numpy as np
from pathlib import Path

from src.config import AppConfig, DeploymentConfig
from src.detector import YOLODetector, DetectionResult
from src.pipeline import InferencePipeline
from src.input_sources import BaseInputSource

class MockInputSource(BaseInputSource):
    """Synthetic source producing a fixed number of black frames."""
    def __init__(self, num_frames: int = 5, fps: float = 30.0):
        self._num_frames = num_frames
        self._fps = fps
        self._current_frame = 0

    def read(self):
        if self._current_frame < self._num_frames:
            self._current_frame += 1
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        return False, None

    def release(self):
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_deployment_config_defaults():
    """Verify DeploymentConfig defaults and AppConfig integration."""
    cfg = AppConfig.from_yaml("configs/config.yaml")
    assert hasattr(cfg, "deployment")
    assert isinstance(cfg.deployment, DeploymentConfig)
    assert cfg.deployment.backend in ["pytorch", "onnx", "openvino"]
    assert cfg.deployment.imgsz == 640
    assert cfg.deployment.warmup_iterations == 10
    assert cfg.deployment.benchmark_iterations == 100


def test_onnx_model_loading_and_inference():
    """Verify that ONNX model loads and runs inference successfully."""
    onnx_path = REPO_ROOT / "weights" / "deployment" / "phase2f_best.onnx"
    assert onnx_path.exists(), "Exported ONNX model should exist under weights/deployment/"

    detector = YOLODetector(
        model_path=str(onnx_path),
        backend="onnx",
        conf_threshold=0.25,
    )
    assert detector.backend == "onnx"

    # Synthetic blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.predict(frame)
    assert isinstance(res, DetectionResult)
    assert isinstance(res.boxes, list)
    assert isinstance(res.eyes_closed, bool)
    assert isinstance(res.eyes_open, bool)
    assert isinstance(res.yawning, bool)


def test_openvino_model_loading_and_inference():
    """Verify that OpenVINO IR model loads and runs inference successfully."""
    openvino_path = REPO_ROOT / "weights" / "deployment" / "phase2f_openvino_model"
    assert openvino_path.exists(), "Exported OpenVINO model should exist under weights/deployment/"

    detector = YOLODetector(
        model_path=str(openvino_path),
        backend="openvino",
        conf_threshold=0.25,
    )
    assert detector.backend == "openvino"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.predict(frame)
    assert isinstance(res, DetectionResult)
    assert isinstance(res.boxes, list)


def test_backend_output_contract_consistency():
    """Verify that PyTorch and ONNX backends return consistent DetectionResult contracts."""
    pt_detector = YOLODetector(
        model_path="weights/phase2f_best.pt",
        backend="pytorch",
        conf_threshold=0.25,
    )
    onnx_detector = YOLODetector(
        model_path="weights/deployment/phase2f_best.onnx",
        backend="onnx",
        conf_threshold=0.25,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_pt = pt_detector.predict(frame)
    res_onnx = onnx_detector.predict(frame)

    assert type(res_pt) is type(res_onnx)
    assert hasattr(res_pt, "boxes") and hasattr(res_onnx, "boxes")
    assert hasattr(res_pt, "eyes_closed") and hasattr(res_onnx, "eyes_closed")
    assert hasattr(res_pt, "eyes_open") and hasattr(res_onnx, "eyes_open")
    assert hasattr(res_pt, "yawning") and hasattr(res_onnx, "yawning")


def test_headless_mode_execution():
    """Verify pipeline runs seamlessly in headless mode without GUI window."""
    cfg = AppConfig.from_yaml("configs/config.yaml")
    cfg.display = False
    cfg.save_output = False
    cfg.max_frames = 5

    # Use synthetic input source
    source = MockInputSource(num_frames=5, fps=30.0)
    pipeline = InferencePipeline(config=cfg, input_source=source)
    results = pipeline.run()

    assert results["frames_processed"] == 5

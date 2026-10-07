"""
Configuration module for Driver Drowsiness Detection System.

Defines the central AppConfig dataclass, physiological signal sub-configs,
and canonical class mapping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

# ==============================================================================
# CANONICAL CLASS MAPPING
# ==============================================================================
CLASS_NAMES: Dict[int, str] = {
    0: "eyes_closed",
    1: "eyes_open",
    2: "yawning",
}


@dataclass
class LandmarksConfig:
    enabled: bool = True
    model_path: str = "face_landmarker.task"
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    max_faces: int = 1


@dataclass
class EARConfig:
    enabled: bool = True
    threshold: float = 0.21
    min_closed_duration_seconds: float = 0.15


@dataclass
class BlinkConfig:
    enabled: bool = True
    normal_blink_min_seconds: float = 0.08
    normal_blink_max_seconds: float = 0.40
    long_blink_max_seconds: float = 0.80
    rolling_window_seconds: float = 60.0


@dataclass
class MARConfig:
    enabled: bool = True
    threshold: float = 0.55


@dataclass
class YawnConfig:
    enabled: bool = True
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 8.0
    cooldown_seconds: float = 3.0
    require_yolo_consensus: bool = False


@dataclass
class PERCLOSConfig:
    enabled: bool = True
    window_seconds: float = 60.0
    minimum_samples: int = 30


@dataclass
class HeadPoseConfig:
    enabled: bool = True
    yaw_threshold_degrees: float = 20.0
    pitch_threshold_degrees: float = 15.0
    roll_threshold_degrees: float = 20.0
    sustained_duration_seconds: float = 2.0


@dataclass
class TelemetryConfig:
    enabled: bool = False
    output_path: str = "logs/telemetry.csv"
    format: str = "csv"


@dataclass
class FusionWeightsConfig:
    eye_closure: float = 0.35
    perclos: float = 0.25
    blink_anomaly: float = 0.15
    yawn_duration: float = 0.15
    head_pose: float = 0.10


@dataclass
class FusionConfig:
    enabled: bool = True
    warning_persistence_seconds: float = 1.0
    critical_persistence_seconds: float = 1.5
    recovery_seconds: float = 2.0
    face_loss_timeout_seconds: float = 2.0
    warning_score_threshold: float = 45.0
    critical_score_threshold: float = 75.0
    weights: FusionWeightsConfig = field(default_factory=FusionWeightsConfig)


@dataclass
class AlertManagerConfig:
    mode: str = "both"  # "visual_only", "audio_only", "both"
    cooldown_seconds: float = 2.5
    audio_warning_hz: int = 1000
    audio_critical_hz: int = 2000
    audio_duration_ms: int = 150
    log_events: bool = True
    event_log_path: str = "results/phase2g/logs/pipeline_events.csv"


@dataclass
class DeploymentConfig:
    backend: str = "pytorch"  # "pytorch", "onnx", "openvino"
    device: str = "cpu"
    model_path: str = "weights/phase2f_best.pt"
    imgsz: int = 640
    warmup_iterations: int = 10
    benchmark_iterations: int = 100


@dataclass
class ContinualLearningConfig:
    """Configuration for safe continual learning and online adaptation."""
    enabled: bool = False  # Privacy off by default
    buffer_capacity: int = 500
    min_samples_for_training: int = 20
    replay_ratio: float = 0.5  # 50% replay from base dataset, 50% adaptation
    uncertainty_low: float = 0.35
    uncertainty_high: float = 0.65
    pseudo_label_threshold: float = 0.75
    max_drift_threshold: float = 0.25
    max_regression_tolerance: float = 0.02  # Max acceptable mAP regression on base validation
    min_adaptation_f1: float = 0.85
    registry_dir: str = "models"
    candidate_epochs: int = 5
    candidate_batch_size: int = 8
    candidate_imgsz: int = 416
    learning_rate: float = 0.001
    base_data_yaml: str = "configs/yolo_phase2f.yaml"


@dataclass
class AppConfig:
    """Central configuration for the drowsiness detection pipeline."""

    # Model settings
    model_path: str = "best.pt"
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50
    max_detections: int = 3
    device: Optional[str] = None  # None for auto-detect (CUDA/CPU)

    # Frame processing
    skip_frames: int = 2

    # Baseline Drowsiness Detection Thresholds (Legacy Frame-Counter Engine)
    eye_closure_threshold_seconds: float = 0.70
    yawn_threshold_seconds: float = 0.50
    critical_cooldown_seconds: float = 3.00
    warning_cooldown_seconds: float = 1.50

    # Input / Output
    input_source: Union[str, int] = "test_video.mp4"
    output_path: Optional[str] = "output_video.mp4"
    display: bool = False
    save_output: bool = True
    max_frames: Optional[int] = None

    # Class mappings
    classes: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))

    # Phase 2B Physiological & Temporal Modules
    landmarks: LandmarksConfig = field(default_factory=LandmarksConfig)
    ear: EARConfig = field(default_factory=EARConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    mar: MARConfig = field(default_factory=MARConfig)
    yawn: YawnConfig = field(default_factory=YawnConfig)
    perclos: PERCLOSConfig = field(default_factory=PERCLOSConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # Phase 2G Multi-Signal Fusion & Alert Management
    fusion: FusionConfig = field(default_factory=FusionConfig)
    alert_manager: AlertManagerConfig = field(default_factory=AlertManagerConfig)

    # Phase 2H Deployment & Export Optimization
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # Continual Learning & Online Adaptation
    continual_learning: ContinualLearningConfig = field(default_factory=ContinualLearningConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], **overrides) -> "AppConfig":
        """Load configuration from a YAML file with optional runtime overrides."""
        path = Path(yaml_path)
        data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    data = loaded

        # Apply classes dict key conversion if loaded as string keys
        if "classes" in data and isinstance(data["classes"], dict):
            data["classes"] = {int(k): str(v) for k, v in data["classes"].items()}

        # Instantiate sub-dataclasses if present in data
        sub_configs = {
            "landmarks": LandmarksConfig,
            "ear": EARConfig,
            "blink": BlinkConfig,
            "mar": MARConfig,
            "yawn": YawnConfig,
            "perclos": PERCLOSConfig,
            "head_pose": HeadPoseConfig,
            "telemetry": TelemetryConfig,
            "alert_manager": AlertManagerConfig,
            "deployment": DeploymentConfig,
            "continual_learning": ContinualLearningConfig,
        }
        for key, sub_cls in sub_configs.items():
            if key in data and isinstance(data[key], dict):
                data[key] = sub_cls(**data[key])

        if "fusion" in data and isinstance(data["fusion"], dict):
            f_dict = dict(data["fusion"])
            w_dict = f_dict.pop("weights", None)
            weights = FusionWeightsConfig(**w_dict) if isinstance(w_dict, dict) else FusionWeightsConfig()
            data["fusion"] = FusionConfig(weights=weights, **f_dict)

        # Merge YAML data with overrides
        merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**{k: v for k, v in merged.items() if k in cls.__dataclass_fields__})


def get_default_config(**overrides) -> AppConfig:
    """Return default configuration, optionally looking for configs/config.yaml."""
    default_yaml = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    if default_yaml.exists():
        return AppConfig.from_yaml(default_yaml, **overrides)
    return AppConfig(**overrides)

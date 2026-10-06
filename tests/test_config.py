"""
Tests for the configuration module and canonical class index safety.
"""

from pathlib import Path
import pytest
from src.config import AppConfig, CLASS_NAMES, get_default_config


def test_canonical_class_mappings():
    """
    Step 12 Safety Test:
    Verify that canonical class IDs strictly match best.pt and custom_dataset.yaml:
      0: eyes_closed
      1: eyes_open
      2: yawning
    Ensures that legacy classes.txt ordering does not corrupt inferences.
    """
    assert CLASS_NAMES[0] == "eyes_closed"
    assert CLASS_NAMES[1] == "eyes_open"
    assert CLASS_NAMES[2] == "yawning"
    assert len(CLASS_NAMES) == 3


def test_default_config_values():
    """Verify that default configuration preserves baseline behavior."""
    cfg = AppConfig()

    assert cfg.model_path == "best.pt"
    assert cfg.confidence_threshold == 0.40
    assert cfg.iou_threshold == 0.50
    assert cfg.max_detections == 3
    assert cfg.skip_frames == 2
    assert cfg.eye_closure_threshold_seconds == 0.70
    assert cfg.yawn_threshold_seconds == 0.50
    assert cfg.critical_cooldown_seconds == 3.00
    assert cfg.warning_cooldown_seconds == 1.50
    assert cfg.classes == CLASS_NAMES


def test_load_config_from_yaml(tmp_path):
    """Verify loading from a custom YAML file with overrides."""
    yaml_content = """
confidence_threshold: 0.65
skip_frames: 4
eye_closure_threshold_seconds: 1.2
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = AppConfig.from_yaml(yaml_file, skip_frames=5)
    assert cfg.confidence_threshold == 0.65
    assert cfg.eye_closure_threshold_seconds == 1.2
    assert cfg.skip_frames == 5  # Override took precedence
    assert cfg.iou_threshold == 0.50  # Fallback to default

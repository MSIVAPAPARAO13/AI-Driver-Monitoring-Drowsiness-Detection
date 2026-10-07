"""
Unit and integration tests for Continual Learning & Online Adaptation system.

Tests:
1. ModelRegistry (initialization, versioning, promotion, rejection, rollback).
2. ContinualLearningBuffer (quality filter, deduplication, capacity, label hierarchy).
3. Human review operations (accept, reject, correct, skip).
4. DriftMonitor (environmental and distribution shifts).
5. Overfitting & Underfitting detection.
6. ValidationGate (regression tolerance, class safety, adaptation thresholds).
7. Background worker training lifecycle & non-blocking execution.
8. ContinualLearningManager coordination & hot-swap callback.
"""

import os
import shutil
import time
from pathlib import Path
import numpy as np
import pytest

from src.config import ContinualLearningConfig
from src.continual_learning import (
    ModelRegistry,
    ContinualLearningBuffer,
    LearningSample,
    DriftMonitor,
    OverfittingDetector,
    ValidationGate,
    BackgroundLearner,
    ContinualLearningManager,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Fixture providing a temporary models directory for isolated testing."""
    reg_dir = tmp_path / "models_test"
    reg_dir.mkdir(parents=True, exist_ok=True)
    yield reg_dir
    if reg_dir.exists():
        shutil.rmtree(reg_dir, ignore_errors=True)


def test_model_registry_initialization_and_versioning(temp_registry_dir):
    """Test registry initializes v001 from baseline without modifying baseline."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    active_path = registry.get_active_model_path()
    assert Path(active_path).exists()

    meta = registry.get_active_metadata()
    assert meta is not None
    assert meta["model_version"] == "v001"
    assert meta["promotion_status"] == "ACTIVE"
    assert meta["validation_metrics"]["map50"] >= 0.90


def test_model_registry_candidate_registration_and_promotion(temp_registry_dir):
    """Test candidate registration, promotion, and archiving of previous model."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)

    # Create dummy candidate weights
    dummy_weights = temp_registry_dir / "candidate_dummy.pt"
    dummy_weights.write_text("dummy model weights")

    cand_ver = registry.register_candidate(
        candidate_weights_path=dummy_weights,
        parent_model="v001",
        training_samples=100,
        replay_samples=50,
        validation_metrics={"map50": 0.982, "f1": 0.930},
    )
    assert cand_ver == "v002"

    models = registry.list_models()
    versions = [m["model_version"] for m in models]
    assert "v002" in versions

    # Promote candidate
    promoted = registry.promote_candidate("v002", reason="Empirically superior mAP50.")
    assert promoted is True

    # Active model should now be v002
    active_meta = registry.get_active_metadata()
    assert active_meta["model_version"] == "v002"
    assert active_meta["promotion_status"] == "ACTIVE"

    # v001 should be ARCHIVED
    all_models = {m["model_version"]: m for m in registry.list_models()}
    assert all_models["v001"]["promotion_status"] == "ARCHIVED"


def test_model_registry_candidate_rejection(temp_registry_dir):
    """Test candidate rejection records reason and does not touch active model."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)

    dummy_weights = temp_registry_dir / "candidate_fail.pt"
    dummy_weights.write_text("dummy model weights fail")

    cand_ver = registry.register_candidate(
        candidate_weights_path=dummy_weights,
        parent_model="v001",
        training_samples=80,
        replay_samples=40,
        validation_metrics={"map50": 0.910, "f1": 0.850},
    )

    rejected = registry.reject_candidate(cand_ver, reason="Severe regression on test split.")
    assert rejected is True

    active_meta = registry.get_active_metadata()
    assert active_meta["model_version"] == "v001"
    assert active_meta["promotion_status"] == "ACTIVE"

    cand_meta = {m["model_version"]: m for m in registry.list_models()}[cand_ver]
    assert cand_meta["promotion_status"] == "REJECTED"
    assert "regression" in cand_meta["reason_for_rejection"]


def test_model_registry_rollback(temp_registry_dir):
    """Test rolling back restores the previous archived model."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)

    dummy_weights = temp_registry_dir / "candidate_dummy.pt"
    dummy_weights.write_text("dummy model weights")
    cand_ver = registry.register_candidate(
        candidate_weights_path=dummy_weights,
        parent_model="v001",
        training_samples=100,
        replay_samples=50,
        validation_metrics={"map50": 0.985, "f1": 0.940},
    )
    registry.promote_candidate(cand_ver, "Promoting v002")
    assert registry.get_active_metadata()["model_version"] == "v002"

    # Rollback to v001
    success, msg = registry.rollback("v001")
    assert success is True
    assert registry.get_active_metadata()["model_version"] == "v001"


def test_continual_learning_buffer_quality_filtering_and_privacy():
    """Test that disabled buffer rejects all frames, while enabled buffer filters informative ones."""
    cfg = ContinualLearningConfig(enabled=False, buffer_capacity=10)
    buf = ContinualLearningBuffer(config=cfg)

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.50, "xyxy": [10, 10, 50, 50]}]

    # Disabled: must reject
    added, reason = buf.add_observation(frame, boxes, ear=0.25, mar=0.20, driver_state="NORMAL")
    assert added is False
    assert "disabled" in reason.lower()

    # Enabled: uncertain confidence (0.50) triggers addition
    cfg.enabled = True
    added, sample_id = buf.add_observation(frame, boxes, ear=0.25, mar=0.20, driver_state="NORMAL")
    assert added is True
    assert sample_id is not None
    assert len(buf.samples) == 1


def test_continual_learning_buffer_deduplication():
    """Test perceptual deduplication rejects nearly identical consecutive frames."""
    cfg = ContinualLearningConfig(enabled=True, buffer_capacity=10)
    buf = ContinualLearningBuffer(config=cfg)

    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    boxes = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.45, "xyxy": [10, 10, 50, 50]}]

    added1, _ = buf.add_observation(frame, boxes, ear=0.25, mar=0.20, driver_state="NORMAL")
    assert added1 is True

    # Immediate identical frame should be rejected as near-duplicate
    added2, reason2 = buf.add_observation(frame, boxes, ear=0.25, mar=0.20, driver_state="NORMAL")
    assert added2 is False
    assert "duplicate" in reason2.lower()


def test_label_hierarchy_levels():
    """Test hierarchy assignment (Level 2 Pseudo-label vs Level 3 Needs Review)."""
    cfg = ContinualLearningConfig(enabled=True, pseudo_label_threshold=0.75, uncertainty_low=0.35, uncertainty_high=0.65)
    buf = ContinualLearningBuffer(config=cfg)

    frame1 = np.full((100, 100, 3), 40, dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 210, dtype=np.uint8)

    # 1. High confidence + physiological consensus -> Level 2
    boxes_high = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.85, "xyxy": [10, 10, 50, 50]}]
    added1, s1 = buf.add_observation(frame1, boxes_high, ear=0.15, mar=0.20, driver_state="NORMAL")
    assert added1 is True
    assert buf.samples[s1].label_level == 2

    # 2. Conflicting or uncertain signal -> Level 3 (Needs Review)
    boxes_uncertain = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.48, "xyxy": [10, 10, 50, 50]}]
    added2, s2 = buf.add_observation(frame2, boxes_uncertain, ear=0.30, mar=0.20, driver_state="NORMAL")
    assert added2 is True
    assert buf.samples[s2].label_level == 3


def test_human_review_actions():
    """Test human reviewer accept, correct, and reject actions."""
    cfg = ContinualLearningConfig(enabled=True)
    buf = ContinualLearningBuffer(config=cfg)

    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    boxes = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.45, "xyxy": [10, 10, 50, 50]}]
    buf.add_observation(frame, boxes, ear=0.25, mar=0.20, driver_state="NORMAL", force_review=True)

    samples = buf.get_review_queue()
    assert len(samples) >= 1
    sample_id = samples[0].sample_id

    # Accept
    buf.review_sample(sample_id, action="accept")
    assert buf.samples[sample_id].label_level == 1
    assert buf.samples[sample_id].reviewed is True

    # Correct
    corrected = [{"cls_id": 1, "cls_name": "eyes_open", "conf": 1.0, "xyxy": [0, 0, 100, 100]}]
    buf.review_sample(sample_id, action="correct", corrected_labels=corrected)
    assert buf.samples[sample_id].verified_labels == corrected

    # Reject (deletes from buffer)
    buf.review_sample(sample_id, action="reject")
    assert sample_id not in buf.samples


def test_drift_monitor_detection():
    """Test drift monitor identifies severe luminance and blur shifts."""
    monitor = DriftMonitor(window_size=10, max_drift_threshold=0.20)

    # Nominal frames
    nominal_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    for _ in range(10):
        res = monitor.update(nominal_frame, avg_confidence=0.75)
    assert res["drift_detected"] is False

    # Severe dark frame (shift in luminance)
    dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
    for _ in range(10):
        res = monitor.update(dark_frame, avg_confidence=0.40)
    assert res["drift_detected"] is True
    assert "POSSIBLE DATA DRIFT" in res["message"]

    status = monitor.get_status()
    assert status["drift_detected"] is True
    assert "drift_score" in status


def test_overfitting_and_underfitting_detection():
    """Test overfitting and underfitting curve analyzers."""
    # Overfitting: Train loss dropping, val loss increasing, val mAP dropping
    train_losses = [0.08, 0.05, 0.02]
    val_losses = [0.05, 0.065, 0.085]
    train_map = [0.90, 0.95, 0.98]
    val_map = [0.91, 0.88, 0.82]

    is_overfit, is_underfit, diag = OverfittingDetector.check_curves(
        train_losses, val_losses, train_map, val_map
    )
    assert is_overfit is True
    assert is_underfit is False
    assert "OVERFITTING" in diag

    # Underfitting: Both train and val mAP poor
    train_losses = [0.20, 0.19, 0.18]
    val_losses = [0.22, 0.21, 0.20]
    train_map = [0.45, 0.48, 0.52]
    val_map = [0.40, 0.44, 0.48]

    is_overfit, is_underfit, diag = OverfittingDetector.check_curves(
        train_losses, val_losses, train_map, val_map
    )
    assert is_overfit is False
    assert is_underfit is True
    assert "UNDERFITTING" in diag


def test_validation_gate_pass_and_regression_rejection():
    """Test validation gate promotes superior models and blocks regressed candidates."""
    cfg = ContinualLearningConfig(max_regression_tolerance=0.02, min_adaptation_f1=0.85)
    gate = ValidationGate(config=cfg)

    base_metrics = {"map50": 0.9777, "f1": 0.9189, "recall": 0.9382}

    # Case 1: Successful candidate
    cand_good = {"map50": 0.9810, "f1": 0.9250, "recall": 0.9400, "eyes_closed_recall": 0.950}
    adapt_good = {"f1": 0.920}
    res_pass = gate.evaluate_candidate(cand_good, base_metrics, adapt_good)
    assert res_pass.passed is True
    assert "PASSED" in res_pass.reason

    # Case 2: Regression on baseline mAP (drop > 0.02)
    cand_regressed = {"map50": 0.9400, "f1": 0.8800, "recall": 0.9000, "eyes_closed_recall": 0.920}
    res_fail = gate.evaluate_candidate(cand_regressed, base_metrics, adapt_good)
    assert res_fail.passed is False
    assert res_fail.regression_detected is True
    assert "Unacceptable regression" in res_fail.reason

    # Case 3: Degraded drowsiness recall (< 0.85)
    cand_safety_fail = {"map50": 0.9780, "f1": 0.9190, "recall": 0.9300, "eyes_closed_recall": 0.810}
    res_safety = gate.evaluate_candidate(cand_safety_fail, base_metrics, adapt_good)
    assert res_safety.passed is False
    assert "eyes_closed recall" in res_safety.reason


def test_background_worker_lifecycle(temp_registry_dir):
    """Test background worker pauses, resumes, and executes training cycle asynchronously."""
    cfg = ContinualLearningConfig(
        enabled=True,
        registry_dir=str(temp_registry_dir),
        candidate_epochs=2,
        min_samples_for_training=2,
    )
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    buffer = ContinualLearningBuffer(config=cfg)

    # Populate buffer with 2 high-confidence samples
    for i in range(2):
        f = np.full((100, 100, 3), (i + 1) * 70, dtype=np.uint8)
        boxes = [{"cls_id": 0, "cls_name": "eyes_closed", "conf": 0.85, "xyxy": [10, 10, 50, 50]}]
        buffer.add_observation(f, boxes, ear=0.15, mar=0.20, driver_state="NORMAL")

    worker = BackgroundLearner(registry=registry, buffer=buffer, config=cfg)
    assert worker.status == "COLLECTING"

    # Start training
    started, msg = worker.trigger_training()
    assert started is True
    assert worker.status == "TRAINING"

    # Pause and Resume
    worker.pause()
    assert worker.status == "PAUSED"
    worker.resume()
    assert worker.status == "COLLECTING" or worker.status == "TRAINING"

    # Wait for completion of small 2-epoch job
    worker._thread.join(timeout=5.0)
    status_info = worker.get_status()
    assert status_info["status"] in ("READY_FOR_PROMOTION", "PROMOTED", "REJECTED")
    assert status_info["last_candidate_version"] is not None


def test_continual_learning_manager_frame_processing(temp_registry_dir):
    """Test manager coordinates frame observation, drift detection, and promotion callback."""
    cfg = ContinualLearningConfig(
        enabled=True,
        min_samples_for_training=5,
        registry_dir=str(temp_registry_dir),
    )
    manager = ContinualLearningManager(config=cfg)

    # Frame observation
    frame = np.full((100, 100, 3), 125, dtype=np.uint8)
    boxes = [{"cls_id": 1, "cls_name": "eyes_open", "conf": 0.80, "xyxy": [10, 10, 50, 50]}]
    res = manager.process_frame_observation(frame, boxes, ear=0.30, mar=0.18, driver_state="NORMAL")

    assert "drift_info" in res
    assert "buffer_counts" in res
    assert res["continual_learning_enabled"] is True

    # Test promotion callback
    promoted_path = None
    def on_promote(path):
        nonlocal promoted_path
        promoted_path = path

    manager.on_model_promoted_callback = on_promote
    active_ver = manager.registry.get_active_metadata()["model_version"]
    # Register dummy candidate
    cand_file = Path(manager.registry.get_active_model_path())
    cand_id = manager.registry.register_candidate(
        candidate_weights_path=cand_file,
        parent_model=active_ver,
        training_samples=50,
        replay_samples=25,
        validation_metrics={"map50": 0.985, "f1": 0.930},
    )
    success, msg = manager.promote_candidate(cand_id, "Passed unit test verification")
    assert success is True
    assert promoted_path is not None

"""
Consolidated Physiological Signals & Temporal Analysis Module.

Contains the lightweight, production-ready components required by InferencePipeline:
- FaceLandmarkDetector (MediaPipe FaceLandmarker 478 3D Mesh)
- EARCalculator (Eye Aspect Ratio)
- MARCalculator (Mouth Aspect Ratio)
- BlinkAnalyzer (Normal/Long/Prolonged Blink Tracking & BPM)
- YawnDurationAnalyzer (Sustained Yawn Duration vs Speech)
- PERCLOSAnalyzer (Time-Weighted Sliding Window Eye Closure)
- HeadPoseEstimator (Pitch, Yaw, Roll & Head Orientation)
- TemporalSignal (Unified Typed Telemetry Dataclass)
- TelemetryLogger (Streaming CSV/JSONL Logger)

NOTE: Deep ML/DL experiments, exploratory analysis, visualizations, and
threshold calibrations are conducted in the companion Jupyter notebooks:
  notebooks/06_physiological_signals.ipynb
  notebooks/07_temporal_analysis.ipynb
  notebooks/08_head_pose_analysis.ipynb
  notebooks/09_phase2b_evaluation.ipynb
"""

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import urllib.request
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ==============================================================================
# 1. TEMPORAL SIGNAL STRUCTURE
# ==============================================================================

@dataclass
class TemporalSignal:
    """Unified telemetry snapshot for a single processed frame."""
    timestamp: float
    frame_index: int
    fps: float
    face_detected: bool = False

    # Eye Aspect Ratio (EAR)
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    average_ear: Optional[float] = None
    eye_closed: bool = False

    # Blink Analysis
    blink_event: Optional[str] = None  # "NORMAL_BLINK", "LONG_BLINK", "PROLONGED_CLOSURE", None
    blink_duration: float = 0.0
    blink_count: int = 0

    # Mouth Aspect Ratio (MAR) & Yawn Analysis
    mar: Optional[float] = None
    yawn_candidate: bool = False
    yawn_duration: float = 0.0
    yawn_event: Optional[str] = None  # "YAWN_START", "YAWN_CONFIRMED", "YAWN_END", None
    yawn_count: int = 0

    # PERCLOS (Percentage of Eye Closure over window)
    perclos: float = 0.0

    # Head Pose (degrees)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    head_state: str = "HEAD_FORWARD"  # "HEAD_FORWARD", "HEAD_LEFT", "HEAD_RIGHT", "HEAD_UP", "HEAD_DOWN", "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. FACIAL LANDMARK DETECTION
# ==============================================================================

@dataclass
class FaceLandmarksResult:
    """Structured representation of detected face landmarks."""
    detected: bool
    pixel_landmarks: Optional[np.ndarray] = None  # Shape (N, 3), coords in pixel space (x, y, z)
    normalized_landmarks: Optional[np.ndarray] = None  # Shape (N, 3), normalized [0, 1]
    transformation_matrix: Optional[np.ndarray] = None  # 4x4 rigid pose matrix
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    face_count: int = 0


DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceLandmarkDetector:
    """Detects 3D facial landmarks using MediaPipe FaceLandmarker with multi-face driver selection."""

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        max_faces: int = 2,
    ):
        self.model_path = Path(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_faces = max_faces
        self.detector = None

        if not MEDIAPIPE_AVAILABLE:
            return

        self._ensure_model_exists()
        self._initialize_detector()

    def _ensure_model_exists(self) -> None:
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(self.model_path))
            except Exception:
                pass

    def _initialize_detector(self) -> None:
        if not self.model_path.exists():
            return
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                output_facial_transformation_matrixes=True,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception:
            self.detector = None

    def detect(self, frame: np.ndarray) -> FaceLandmarksResult:
        if self.detector is None or frame is None or frame.size == 0:
            return FaceLandmarksResult(detected=False, face_count=0)

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return FaceLandmarksResult(detected=False, face_count=0)

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)
        except Exception:
            return FaceLandmarksResult(detected=False, face_count=0)

        faces = detection_result.face_landmarks
        if not faces:
            return FaceLandmarksResult(detected=False, face_count=0)

        # Multi-face selection: deterministic driver selection strategy combining
        # bounding box area (dominant foreground), center proximity, and temporal continuity
        selected_idx = 0
        best_score = -1.0
        selected_pixel_pts = None
        selected_norm_pts = None
        selected_bbox = None

        frame_cx, frame_cy = w / 2.0, h / 2.0
        for idx, face_pts in enumerate(faces):
            norm_pts = np.array([(p.x, p.y, p.z) for p in face_pts], dtype=np.float32)
            pixel_pts = np.zeros_like(norm_pts)
            pixel_pts[:, 0] = norm_pts[:, 0] * w
            pixel_pts[:, 1] = norm_pts[:, 1] * h
            pixel_pts[:, 2] = norm_pts[:, 2] * w

            x1 = max(0, int(np.min(pixel_pts[:, 0])))
            y1 = max(0, int(np.min(pixel_pts[:, 1])))
            x2 = min(w - 1, int(np.max(pixel_pts[:, 0])))
            y2 = min(h - 1, int(np.max(pixel_pts[:, 1])))
            area = max(0.0, float((x2 - x1) * (y2 - y1)))
            face_cx, face_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Deterministic selection score:
            # 1. Area is the primary signal (closest face to the camera)
            # 2. Centroid continuity: penalty if far from previous detected driver
            continuity_penalty = 1.0
            if getattr(self, "last_driver_center", None) is not None:
                prev_cx, prev_cy = self.last_driver_center
                dist_prev = float(np.hypot(face_cx - prev_cx, face_cy - prev_cy))
                continuity_penalty = 1.0 + 0.3 * (dist_prev / max(1.0, float(w)))

            score = area / continuity_penalty

            if score > best_score:
                best_score = score
                selected_idx = idx
                selected_pixel_pts = pixel_pts
                selected_norm_pts = norm_pts
                selected_bbox = (x1, y1, x2, y2)

        if selected_bbox is not None:
            self.last_driver_center = ((selected_bbox[0] + selected_bbox[2]) / 2.0, (selected_bbox[1] + selected_bbox[3]) / 2.0)

        trans_matrix = None
        if (
            hasattr(detection_result, "facial_transformation_matrixes")
            and detection_result.facial_transformation_matrixes
            and len(detection_result.facial_transformation_matrixes) > selected_idx
        ):
            trans_matrix = np.array(
                detection_result.facial_transformation_matrixes[selected_idx],
                dtype=np.float32,
            )

        return FaceLandmarksResult(
            detected=True,
            pixel_landmarks=selected_pixel_pts,
            normalized_landmarks=selected_norm_pts,
            transformation_matrix=trans_matrix,
            bbox=selected_bbox,
            face_count=len(faces),
        )

    def close(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None


# ==============================================================================
# 3. EYE METRICS (EAR)
# ==============================================================================

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


@dataclass
class EARResult:
    left_ear: Optional[float]
    right_ear: Optional[float]
    average_ear: Optional[float]
    is_closed: bool
    is_valid: bool


class EARCalculator:
    """Computes Left, Right, and Average Eye Aspect Ratio (EAR) from facial landmarks."""

    def __init__(
        self,
        threshold: float = 0.21,
        left_indices: Optional[list] = None,
        right_indices: Optional[list] = None,
    ):
        self.threshold = float(threshold)
        self.left_indices = left_indices or LEFT_EYE_INDICES
        self.right_indices = right_indices or RIGHT_EYE_INDICES

    def _compute_single_ear(self, landmarks: np.ndarray, indices: list) -> Optional[float]:
        if landmarks is None or len(landmarks) <= max(indices):
            return None
        pts = landmarks[indices, :2]
        d_v1 = float(np.linalg.norm(pts[1] - pts[5]))
        d_v2 = float(np.linalg.norm(pts[2] - pts[4]))
        d_h = float(np.linalg.norm(pts[0] - pts[3]))
        if d_h < 1e-6:
            return None
        return float((d_v1 + d_v2) / (2.0 * d_h))

    def calculate(self, landmarks: Optional[np.ndarray]) -> EARResult:
        if landmarks is None:
            return EARResult(None, None, None, False, False)

        left_ear = self._compute_single_ear(landmarks, self.left_indices)
        right_ear = self._compute_single_ear(landmarks, self.right_indices)

        if left_ear is None and right_ear is None:
            return EARResult(None, None, None, False, False)

        if left_ear is not None and right_ear is not None:
            avg_ear = (left_ear + right_ear) / 2.0
        elif left_ear is not None:
            avg_ear = left_ear
        else:
            avg_ear = right_ear

        return EARResult(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=avg_ear,
            is_closed=avg_ear < self.threshold,
            is_valid=True,
        )


# ==============================================================================
# 4. MOUTH METRICS (MAR)
# ==============================================================================

MOUTH_CORNER_INDICES = (78, 308)
MOUTH_VERTICAL_PAIRS = [(82, 87), (13, 14), (312, 317)]


@dataclass
class MARResult:
    mar: Optional[float]
    is_open: bool
    is_valid: bool


class MARCalculator:
    """Computes Mouth Aspect Ratio (MAR) to evaluate lip separation."""

    def __init__(
        self,
        threshold: float = 0.55,
        corner_indices: tuple = MOUTH_CORNER_INDICES,
        vertical_pairs: list = MOUTH_VERTICAL_PAIRS,
    ):
        self.threshold = float(threshold)
        self.corner_indices = corner_indices
        self.vertical_pairs = vertical_pairs

    def calculate(self, landmarks: Optional[np.ndarray]) -> MARResult:
        if landmarks is None:
            return MARResult(None, False, False)

        max_idx = max(
            self.corner_indices[0],
            self.corner_indices[1],
            *(max(p[0], p[1]) for p in self.vertical_pairs),
        )
        if len(landmarks) <= max_idx:
            return MARResult(None, False, False)

        pts = landmarks[:, :2]
        d_h = float(np.linalg.norm(pts[self.corner_indices[0]] - pts[self.corner_indices[1]]))
        if d_h < 1e-6:
            return MARResult(None, False, False)

        d_v_total = sum(float(np.linalg.norm(pts[u] - pts[l])) for u, l in self.vertical_pairs)
        mar = d_v_total / (2.0 * d_h)
        return MARResult(mar=float(mar), is_open=mar > self.threshold, is_valid=True)


# ==============================================================================
# 5. BLINK ANALYZER
# ==============================================================================

@dataclass
class BlinkEvent:
    event_type: str  # "NORMAL_BLINK", "LONG_BLINK", "PROLONGED_CLOSURE"
    duration: float
    start_time: float
    end_time: float
    frame_index: int


class BlinkAnalyzer:
    """Tracks temporal eye closures to detect blinks and prolonged micro-sleep closures."""

    def __init__(
        self,
        normal_blink_min_s: float = 0.08,
        normal_blink_max_s: float = 0.40,
        long_blink_max_s: float = 0.80,
        rolling_window_s: float = 60.0,
    ):
        self.normal_blink_min_s = float(normal_blink_min_s)
        self.normal_blink_max_s = float(normal_blink_max_s)
        self.long_blink_max_s = float(long_blink_max_s)
        self.rolling_window_s = float(rolling_window_s)

        self.is_eye_closed: bool = False
        self.closure_start_time: Optional[float] = None
        self.closure_start_frame: int = 0
        self.last_blink_duration: float = 0.0
        self.total_blink_count: int = 0
        self.prolonged_closure_flag: bool = False
        self.blink_timestamps: Deque[float] = deque()

    def update(
        self,
        is_closed: bool,
        timestamp: float,
        frame_index: int = 0,
    ) -> Optional[BlinkEvent]:
        event: Optional[BlinkEvent] = None

        if is_closed:
            if not self.is_eye_closed:
                self.is_eye_closed = True
                self.closure_start_time = timestamp
                self.closure_start_frame = frame_index
                self.prolonged_closure_flag = False
            else:
                start_t = self.closure_start_time if self.closure_start_time is not None else timestamp
                current_duration = timestamp - start_t
                if current_duration > self.long_blink_max_s and not self.prolonged_closure_flag:
                    self.prolonged_closure_flag = True
                    event = BlinkEvent(
                        event_type="PROLONGED_CLOSURE",
                        duration=current_duration,
                        start_time=start_t,
                        end_time=timestamp,
                        frame_index=frame_index,
                    )
        else:
            if self.is_eye_closed:
                start_t = self.closure_start_time if self.closure_start_time is not None else timestamp
                closure_duration = timestamp - start_t
                self.is_eye_closed = False
                self.last_blink_duration = closure_duration

                if not self.prolonged_closure_flag:
                    if self.normal_blink_min_s <= closure_duration <= self.normal_blink_max_s:
                        self.total_blink_count += 1
                        self.blink_timestamps.append(timestamp)
                        event = BlinkEvent("NORMAL_BLINK", closure_duration, start_t, timestamp, frame_index)
                    elif self.normal_blink_max_s < closure_duration <= self.long_blink_max_s:
                        self.total_blink_count += 1
                        self.blink_timestamps.append(timestamp)
                        event = BlinkEvent("LONG_BLINK", closure_duration, start_t, timestamp, frame_index)

                self.closure_start_time = None

        while self.blink_timestamps and (timestamp - self.blink_timestamps[0] > self.rolling_window_s):
            self.blink_timestamps.popleft()

        return event

    def get_blink_rate(self) -> float:
        if not self.blink_timestamps:
            return 0.0
        window_span = max(1.0, min(self.rolling_window_s, self.blink_timestamps[-1] - self.blink_timestamps[0]))
        return float(len(self.blink_timestamps)) * (60.0 / window_span)

    def reset(self) -> None:
        self.is_eye_closed = False
        self.closure_start_time = None
        self.closure_start_frame = 0
        self.last_blink_duration = 0.0
        self.total_blink_count = 0
        self.prolonged_closure_flag = False
        self.blink_timestamps.clear()


# ==============================================================================
# 6. YAWN DURATION ANALYZER
# ==============================================================================

@dataclass
class YawnStatus:
    is_candidate: bool
    current_duration: float
    is_confirmed: bool
    event: Optional[str]
    total_yawn_count: int


class YawnDurationAnalyzer:
    """Tracks sustained mouth opening to confirm genuine yawns and reject speech."""

    def __init__(
        self,
        min_duration_s: float = 2.0,
        max_duration_s: float = 8.0,
        cooldown_s: float = 3.0,
        require_yolo_consensus: bool = False,
    ):
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        self.cooldown_s = float(cooldown_s)
        self.require_yolo_consensus = require_yolo_consensus

        self.in_candidate: bool = False
        self.candidate_start_time: Optional[float] = None
        self.confirmed_flag: bool = False
        self.last_yawn_end_time: float = -999.0
        self.total_yawns: int = 0

    def update(
        self,
        mar_is_open: bool,
        yolo_yawning: bool,
        timestamp: float,
    ) -> YawnStatus:
        is_active = (mar_is_open and yolo_yawning) if self.require_yolo_consensus else (mar_is_open or yolo_yawning)
        in_cooldown = (timestamp - self.last_yawn_end_time) < self.cooldown_s
        event = None
        duration = 0.0

        if is_active and not in_cooldown:
            if not self.in_candidate:
                self.in_candidate = True
                self.candidate_start_time = timestamp
                self.confirmed_flag = False
                event = "YAWN_START"
                duration = 0.0
            else:
                start_t = self.candidate_start_time if self.candidate_start_time is not None else timestamp
                duration = timestamp - start_t
                if duration >= self.min_duration_s and not self.confirmed_flag:
                    self.confirmed_flag = True
                    self.total_yawns += 1
                    event = "YAWN_CONFIRMED"
        else:
            if self.in_candidate:
                start_t = self.candidate_start_time if self.candidate_start_time is not None else timestamp
                duration = timestamp - start_t
                self.in_candidate = False
                self.candidate_start_time = None
                self.last_yawn_end_time = timestamp
                event = "YAWN_END"
                self.confirmed_flag = False

        return YawnStatus(
            is_candidate=self.in_candidate,
            current_duration=duration,
            is_confirmed=self.confirmed_flag,
            event=event,
            total_yawn_count=self.total_yawns,
        )

    def reset(self) -> None:
        self.in_candidate = False
        self.candidate_start_time = None
        self.confirmed_flag = False
        self.last_yawn_end_time = -999.0
        self.total_yawns = 0


# ==============================================================================
# 7. PERCLOS ANALYZER
# ==============================================================================

@dataclass
class PERCLOSResult:
    perclos: float
    percentage: float
    sample_count: int
    window_duration: float
    is_warmed_up: bool


class PERCLOSAnalyzer:
    """Sliding-window time-weighted PERCLOS evaluator."""

    def __init__(
        self,
        window_seconds: float = 60.0,
        minimum_samples: int = 30,
        warmup_fraction: float = 0.5,
    ):
        self.window_seconds = float(window_seconds)
        self.minimum_samples = int(minimum_samples)
        self.warmup_fraction = float(warmup_fraction)
        self.history: Deque[Tuple[float, bool]] = deque()

    def update(self, is_closed: bool, timestamp: float) -> PERCLOSResult:
        self.history.append((timestamp, is_closed))

        cutoff = timestamp - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        total_samples = len(self.history)
        if total_samples == 0:
            return PERCLOSResult(0.0, 0.0, 0, 0.0, False)

        window_span = self.history[-1][0] - self.history[0][0]
        is_warmed_up = (
            total_samples >= self.minimum_samples
            and window_span >= (self.window_seconds * self.warmup_fraction)
        )

        if total_samples > 1 and window_span > 1e-3:
            closed_time = 0.0
            for i in range(1, total_samples):
                dt = min(self.history[i][0] - self.history[i - 1][0], 2.0)
                if self.history[i - 1][1]:
                    closed_time += dt
            perclos = min(1.0, max(0.0, closed_time / window_span))
        else:
            closed_samples = sum(1 for _, closed in self.history if closed)
            perclos = closed_samples / float(total_samples)

        return PERCLOSResult(
            perclos=float(perclos),
            percentage=float(perclos * 100.0),
            sample_count=total_samples,
            window_duration=float(window_span),
            is_warmed_up=is_warmed_up,
        )

    def reset(self) -> None:
        self.history.clear()


# ==============================================================================
# 8. HEAD POSE ESTIMATOR
# ==============================================================================

CANONICAL_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (idx 1)
    (0.0, -63.6, -12.5),         # Chin (idx 199)
    (-43.3, 32.7, -26.0),        # Left eye outer corner (idx 33)
    (43.3, 32.7, -26.0),         # Right eye outer corner (idx 263)
    (-28.9, -28.9, -24.1),       # Left mouth corner (idx 61)
    (28.9, -28.9, -24.1),        # Right mouth corner (idx 291)
], dtype=np.float64)

LANDMARK_KEYPOINT_INDICES = [1, 199, 33, 263, 61, 291]


@dataclass
class HeadPoseResult:
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    state: str
    sustained_duration: float
    is_valid: bool


class HeadPoseEstimator:
    """Estimates 3D head pose and tracks sustained distraction or nodding."""

    def __init__(
        self,
        yaw_threshold_deg: float = 20.0,
        pitch_threshold_deg: float = 15.0,
        roll_threshold_deg: float = 20.0,
        sustained_duration_s: float = 2.0,
    ):
        self.yaw_threshold_deg = float(yaw_threshold_deg)
        self.pitch_threshold_deg = float(pitch_threshold_deg)
        self.roll_threshold_deg = float(roll_threshold_deg)
        self.sustained_duration_s = float(sustained_duration_s)

        self.current_state: str = "HEAD_FORWARD"
        self.state_start_time: Optional[float] = None
        self.sustained_duration: float = 0.0

    def estimate(
        self,
        landmarks: Optional[np.ndarray],
        transformation_matrix: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
        timestamp: float = 0.0,
    ) -> HeadPoseResult:
        pitch, yaw, roll = None, None, None

        if transformation_matrix is not None and transformation_matrix.shape == (4, 4):
            try:
                R = transformation_matrix[:3, :3]
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
                pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
            except Exception:
                pitch, yaw, roll = None, None, None

        if pitch is None and landmarks is not None and len(landmarks) > max(LANDMARK_KEYPOINT_INDICES):
            try:
                h, w = frame_shape
                image_points = landmarks[LANDMARK_KEYPOINT_INDICES, :2].astype(np.float64)
                focal_length = float(w)
                center = (w / 2.0, h / 2.0)
                camera_matrix = np.array([
                    [focal_length, 0.0, center[0]],
                    [0.0, focal_length, center[1]],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)

                success, rvec, _ = cv2.solvePnP(
                    CANONICAL_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
                    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
            except Exception:
                pitch, yaw, roll = None, None, None

        if pitch is None or yaw is None or roll is None:
            return HeadPoseResult(None, None, None, "UNKNOWN", 0.0, False)

        new_state = "HEAD_FORWARD"
        if yaw > self.yaw_threshold_deg:
            new_state = "HEAD_RIGHT"
        elif yaw < -self.yaw_threshold_deg:
            new_state = "HEAD_LEFT"
        elif pitch > self.pitch_threshold_deg:
            new_state = "HEAD_DOWN"
        elif pitch < -self.pitch_threshold_deg:
            new_state = "HEAD_UP"

        if new_state == self.current_state and new_state != "HEAD_FORWARD":
            self.sustained_duration = timestamp - (self.state_start_time or timestamp)
        else:
            self.current_state = new_state
            self.state_start_time = timestamp
            self.sustained_duration = 0.0

        return HeadPoseResult(pitch, yaw, roll, new_state, self.sustained_duration, True)

    def reset(self) -> None:
        self.current_state = "HEAD_FORWARD"
        self.state_start_time = None
        self.sustained_duration = 0.0


# ==============================================================================
# 9. TELEMETRY LOGGER
# ==============================================================================

CSV_COLUMNS = [
    "timestamp", "frame", "fps", "face_detected",
    "left_ear", "right_ear", "average_ear", "mar", "eye_closed",
    "blink_event", "blink_duration", "blink_count",
    "yawn_candidate", "yawn_duration", "yawn_event", "yawn_count",
    "perclos", "yaw", "pitch", "roll", "head_state",
]


class TelemetryLogger:
    """Streams TemporalSignal records to CSV/JSONL."""

    def __init__(
        self,
        output_path: Union[str, Path] = "logs/telemetry.csv",
        format_type: str = "csv",
        flush_interval: int = 30,
    ):
        self.output_path = Path(output_path)
        self.format_type = format_type.lower()
        self.flush_interval = flush_interval

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.output_path, "w", newline="", encoding="utf-8")
        self.record_count = 0

        if self.format_type == "csv":
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=CSV_COLUMNS)
            self.csv_writer.writeheader()
        else:
            self.csv_writer = None

    def log(self, signal: TemporalSignal) -> None:
        self.record_count += 1
        if self.format_type == "csv":
            row = {
                "timestamp": f"{signal.timestamp:.3f}",
                "frame": signal.frame_index,
                "fps": f"{signal.fps:.2f}",
                "face_detected": int(signal.face_detected),
                "left_ear": f"{signal.left_ear:.4f}" if signal.left_ear is not None else "",
                "right_ear": f"{signal.right_ear:.4f}" if signal.right_ear is not None else "",
                "average_ear": f"{signal.average_ear:.4f}" if signal.average_ear is not None else "",
                "mar": f"{signal.mar:.4f}" if signal.mar is not None else "",
                "eye_closed": int(signal.eye_closed),
                "blink_event": signal.blink_event or "",
                "blink_duration": f"{signal.blink_duration:.3f}",
                "blink_count": signal.blink_count,
                "yawn_candidate": int(signal.yawn_candidate),
                "yawn_duration": f"{signal.yawn_duration:.3f}",
                "yawn_event": signal.yawn_event or "",
                "yawn_count": signal.yawn_count,
                "perclos": f"{signal.perclos:.4f}",
                "yaw": f"{signal.yaw:.2f}" if signal.yaw is not None else "",
                "pitch": f"{signal.pitch:.2f}" if signal.pitch is not None else "",
                "roll": f"{signal.roll:.2f}" if signal.roll is not None else "",
                "head_state": signal.head_state,
            }
            self.csv_writer.writerow(row)
        else:
            self.file_handle.write(json.dumps(signal.to_dict()) + "\n")

        if self.record_count % self.flush_interval == 0:
            self.file_handle.flush()

    def close(self) -> None:
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.flush()
            self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

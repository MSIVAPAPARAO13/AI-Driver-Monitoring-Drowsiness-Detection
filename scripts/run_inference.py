"""
CLI Script for running Driver Drowsiness Detection System inference.

Usage examples:
    # Run on video file with default settings
    python scripts/run_inference.py --source test_video.mp4

    # Run with landmarks explicitly enabled/disabled
    python scripts/run_inference.py --source test_video.mp4 --enable-landmarks
    python scripts/run_inference.py --source test_video.mp4 --disable-landmarks

    # Run with telemetry logging
    python scripts/run_inference.py --source test_video.mp4 --telemetry --telemetry-output logs/telemetry.csv

    # Run on live physical webcam
    python scripts/run_inference.py --source 0 --display
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, get_default_config
from src.pipeline import InferencePipeline
from src.utils import setup_logger

logger = setup_logger("cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Driver Drowsiness Detection System inference pipeline."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="test_video.mp4",
        help="Input video file path or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model checkpoint (overrides config).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold (overrides config).",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Number of frames between YOLO inferences (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (overrides config).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=None,
        help="Display the video output in an OpenCV window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before stopping.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the rendered video to disk.",
    )

    # Phase 2B Physiological & Telemetry CLI flags
    parser.add_argument(
        "--enable-landmarks",
        dest="landmarks_enabled",
        action="store_true",
        default=None,
        help="Explicitly enable MediaPipe facial landmarks and physiological signals.",
    )
    parser.add_argument(
        "--disable-landmarks",
        dest="landmarks_enabled",
        action="store_false",
        default=None,
        help="Explicitly disable MediaPipe facial landmarks and physiological signals.",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        default=False,
        help="Enable streaming telemetry logging (CSV).",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=None,
        help="Path to save the telemetry CSV/JSONL output file.",
    )
    # Phase 2H Deployment & Optimization CLI flags
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "onnx", "openvino"],
        default=None,
        help="Inference runtime backend: 'pytorch', 'onnx', or 'openvino'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI window / display (overrides --display).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run in benchmark mode with detailed latency and throughput metrics.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum duration in seconds to run before terminating.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config from YAML if provided, else defaults
    config_path = Path(args.config)
    overrides = {}

    if args.source is not None:
        overrides["input_source"] = args.source
    if args.model is not None:
        overrides["model_path"] = args.model
    if args.conf is not None:
        overrides["confidence_threshold"] = args.conf
    if args.iou is not None:
        overrides["iou_threshold"] = args.iou
    if args.skip_frames is not None:
        overrides["skip_frames"] = args.skip_frames
    if args.output is not None:
        overrides["output_path"] = args.output
    if args.display is not None:
        overrides["display"] = args.display
    if args.headless:
        overrides["display"] = False
    if args.max_frames is not None:
        overrides["max_frames"] = args.max_frames
    if args.no_save:
        overrides["save_output"] = False

    if config_path.exists():
        config = AppConfig.from_yaml(config_path, **overrides)
    else:
        config = get_default_config(**overrides)

    # Apply deployment backend override
    if args.backend is not None:
        config.deployment.backend = args.backend

    # Apply physiological CLI overrides
    if args.landmarks_enabled is not None:
        config.landmarks.enabled = args.landmarks_enabled
    if args.telemetry:
        config.telemetry.enabled = True
    if args.telemetry_output is not None:
        config.telemetry.output_path = args.telemetry_output
        config.telemetry.enabled = True

    # If duration specified and max_frames not explicitly set, calculate max_frames
    if args.duration is not None and config.max_frames is None:
        # Assume standard 30 FPS estimate if not yet queried
        config.max_frames = int(args.duration * 30.0)

    logger.info(
        f"Loaded config: model={config.model_path}, backend={config.deployment.backend}, "
        f"source={config.input_source}, display={config.display}, "
        f"landmarks={config.landmarks.enabled}, telemetry={config.telemetry.enabled}"
    )

    pipeline = InferencePipeline(config=config)
    results = pipeline.run()

    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY:")
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Average Loop FPS: {results['avg_fps']:.2f}")
    print(f"Wall Clock FPS:   {results['wall_fps']:.2f}")
    print(f"Total Time (s):   {results['total_wall_time_s']:.2f}")
    print(f"Output Video:     {results['output_path']}")
    print(f"Detections:       {results['class_detections']}")
    print(f"Detector Stats:   {results['stats']}")
    if "physiological_summary" in results:
        print(f"Physiological:    {results['physiological_summary']}")
    print("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    main()

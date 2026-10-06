# Final Clean Repository Architecture & Directory Structure
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Canonical Repository Structure  
**Location:** `docs/FINAL_PROJECT_STRUCTURE.md`  
**Standard:** Clean, modular, production-oriented repository layout separating source code, tests, configs, weights, results, documentation, and notebooks.  

```text
Drowsiness-Detection-using-YOLOv5/
│
├── README.md                           # Main GitHub documentation & project overview
├── requirements.txt                    # Production environment dependencies
├── pytest.ini                          # Automated test discovery configuration
├── Dockerfile                          # Production CPU deployment container specification
├── .dockerignore                       # Container build exclusion rules
├── .gitignore                          # Git tracking exclusions (caches, environments, raw data)
│
├── src/                                # Core Modular Production Library
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Dataclass & YAML hierarchical application configuration
│   ├── detector.py                     # Multi-backend YOLO detector (PyTorch, ONNX, OpenVINO)
│   ├── physiological.py                # MediaPipe 3D face mesh, EAR, MAR, PERCLOS, head pose
│   ├── drowsiness_engine.py            # Tier 3 multi-signal fusion, speech suppression, state machine
│   ├── pipeline.py                     # End-to-end ingestion, frame skipping, and processing loop
│   ├── renderer.py                     # OpenCV telemetry dashboard & HUD visualization renderer
│   ├── input_sources.py                # Video file & physical webcam streaming abstractions
│   └── utils.py                        # Structured logging, timing utilities, and helpers
│
├── tests/                              # Automated Pytest Test Suite (39/39 Passing)
│   ├── __init__.py                     # Test package marker
│   ├── test_config.py                  # Configuration validation and defaults
│   ├── test_deployment.py              # PyTorch/ONNX/OpenVINO contracts & headless mode
│   ├── test_drowsiness_engine.py       # Counter decay, alert states, and cooldowns
│   ├── test_edge_cases.py              # No-face, multi-face, low-light, camera disconnect
│   ├── test_fusion.py                  # Speech suppression vs yawn, micro-sleep, face loss
│   ├── test_physiological.py           # EAR, MAR, blink analyzer, PERCLOS, head pose math
│   └── test_pipeline.py                # Full pipeline loop and frame-skipping integrity
│
├── configs/                            # System Configuration Profiles
│   ├── config.yaml                     # Primary application configuration (thresholds, backends)
│   ├── yolo_phase2d.yaml               # Phase 2D baseline training dataset configuration
│   └── yolo_phase2f.yaml               # Phase 2F verified dataset training configuration
│
├── scripts/                            # Operational CLI Execution Scripts
│   └── run_inference.py                # Primary CLI inference, video/webcam runner & benchmark
│
├── weights/                            # Trained Model Checkpoints & Exports
│   ├── phase2f_best.pt                 # Canonical fine-tuned YOLOv5nu PyTorch checkpoint (4.99 MB)
│   ├── phase2f_best.onnx               # Portable ONNX Runtime FP32 export (10.26 MB)
│   ├── phase2f_best_openvino_model/    # Intel OpenVINO IR model directory (XML / BIN)
│   └── best.pt                         # Baseline model checkpoint (historical reference)
│
├── notebooks/                          # Notebook-First Research & Verification Notebooks (01–18)
│   ├── 01_dataset_audit.ipynb          # Baseline dataset audit and leakage identification
│   ├── 02_dataset_preparation.ipynb    # Video frame extraction and annotation formatting
│   ├── 03_person_level_split.ipynb     # Subject-disjoint train/val/test partitioning
│   ├── 04_annotation_expansion.ipynb   # Phase 2D initial annotation expansion
│   ├── 05_yolo_training.ipynb          # Phase 2D initial model training
│   ├── 06_model_evaluation.ipynb       # Phase 2D model evaluation on unseen subjects
│   ├── 07_annotation_quality_control.ipynb # Annotation quality verification
│   ├── 08_phase2e_dataset_analysis.ipynb   # Phase 2E expanded dataset distribution analysis
│   ├── 09_phase2f_yolo_training.ipynb      # Phase 2F YOLOv5nu retraining
│   ├── 10_phase2f_model_evaluation.ipynb   # Phase 2F quantitative test set evaluation
│   ├── 11_phase2g_fusion_experiments.ipynb # Multi-signal temporal fusion experiments
│   ├── 12_phase2g_temporal_benchmark.ipynb # UTA-RLDD 5-fold temporal evaluation
│   ├── 13_phase2g_error_analysis.ipynb     # Fusion false alarm & failure mode analysis
│   ├── 14_phase2h_optimization_benchmark.ipynb # Latency profiling & stage breakdown
│   ├── 15_phase2h_deployment_validation.ipynb  # Multi-backend deployment verification
│   ├── 16_phase2i_final_validation.ipynb       # End-to-end pipeline validation
│   ├── 17_phase2i_reproducibility_audit.ipynb  # Verification against single source of truth
│   └── 18_phase2i_error_analysis.ipynb         # Edge-case robustness analysis
│
├── results/                            # Empirical Experimental Outputs & Plots
│   ├── final_resume_metrics.json       # Canonical single source of truth for all metrics
│   ├── final_project_results.csv       # Unified comparative experimental results table
│   ├── phase2f/                        # Phase 2F evaluation metrics and PR curves
│   ├── phase2g/                        # Phase 2G ablation and UTA-RLDD benchmark results
│   │   ├── ablation_comparison.png     # Stepwise ablation F1 and false alarm curves
│   │   ├── fatigue_score_timeline.png  # Continuous fatigue score tracking plot
│   │   └── uta_rldd_fold_metrics.png   # 5-fold cross-subject temporal performance plot
│   ├── phase2h/                        # Profiling breakdowns and stress test logs
│   └── phase2i/                        # Final validation confusion matrix and PR curves
│       ├── confusion_matrix.png        # Evaluated test split confusion matrix
│       └── confusion_matrix_normalized.png # Normalized test split confusion matrix
│
├── docs/                               # Architecture, Guides, and Documentation
│   ├── FINAL_ARCHITECTURE.md           # Comprehensive 4-tier architectural specification
│   ├── final_architecture.png          # High-resolution (300 DPI) system architecture diagram
│   ├── DEMO_GUIDE.md                   # Operational CLI demonstration and execution guide
│   ├── PROJECT_DESCRIPTION.md          # Standardized project descriptions (short/med/long)
│   ├── RESUME_BULLETS.md               # Verified, metric-grounded CV bullet points
│   ├── FINAL_INTERVIEW_GUIDE.md        # 30-question technical interview preparation guide
│   ├── RESEARCH_CONTRIBUTION.md        # System engineering contributions and methodology
│   ├── FINAL_PROJECT_STRUCTURE.md      # Repository directory structure specification
│   ├── DATASET_ACKNOWLEDGMENTS.md      # Academic dataset sources, licensing, and citations
│   ├── INTERVIEW_READY_NOTES.md        # Technical architectural notes and Q&A
│   ├── README_CONTENT_PLAN.md          # 18-section specification plan for release README
│   └── demo/                           # Demonstration Assets
│       ├── architecture.png            # Embedded architecture diagram asset
│       ├── benchmark_result.png        # Embedded ablation benchmark asset
│       ├── confusion_matrix.png        # Embedded normalized confusion matrix asset
│       ├── fatigue_score_timeline.png  # Embedded fatigue timeline asset
│       └── DEMO_SCREENSHOTS_REQUIRED.md # Guide for capturing operational HUD frames
│
└── data/                               # Data Manifests & Structured Splits (Raw data git-ignored)
    ├── manifests/                      # Audited dataset manifests and split indices
    ├── annotated/                      # Frame-level verified ground-truth annotations
    └── processed/                      # Preprocessed YOLO images and labels
```

# Dataset Workspace & Acquisition Guide
## AI Driver Monitoring & Drowsiness Detection System

This directory manages the local data workspace for training, validation, and benchmarking.

> [!IMPORTANT]
> **Dataset Protection & Licensing Notice:**
> In accordance with academic dataset distribution licenses and ethical research practices, **raw video and image dataset files are NOT committed to this repository**.
> Users must acquire third-party academic datasets directly from their respective authors and accept their license terms.
> For complete licensing, attribution, and citation details, see [docs/DATASET_ACKNOWLEDGMENTS.md](../docs/DATASET_ACKNOWLEDGMENTS.md).

---

## 1. Directory Structure

```text
data/
├── README.md               # This dataset workspace guide
├── manifests/              # Audited split manifests and ground-truth metadata (CSV)
├── raw/                    # Downloaded third-party raw datasets (git-ignored)
├── interim/                # Preprocessed frame caches (git-ignored)
└── processed/              # Formatted YOLO training splits (git-ignored)
```

---

## 2. Dataset Roles & Boundaries

| Dataset | Primary Role | YOLO Bounding Box Training? | Temporal Evaluation? |
|:---|:---|:---:|:---:|
| **YawDD** | Visual Bounding Box Training & Validation | **Yes** (clean frames) | No |
| **DMD** | Visual Diversity Expansion | **Yes** (clean frames) | No |
| **UTA-RLDD** | Sequence-Level Temporal Evaluation | **NO (Strict Boundary)** | **Yes (5-Fold Cross-Validation)** |
| **NTHU-DDD** | Out-of-Distribution (OOD) Robustness | **NO** | No (OOD Auditing) |
| **MRL Eye** | Ocular Calibration & Unit Tests | **NO** | No |

- **UTA-RLDD Usage Policy:** UTA-RLDD consists of sequence-level drowsiness recordings across 60 subjects. It was **NOT used for YOLO bounding-box training** to maintain clean evaluation boundaries. It was used exclusively for 5-fold cross-subject temporal evaluation.
- **Subject-Disjoint Rule:** All visual training and validation partitions strictly enforce subject-level separation. Zero subject overlap exists between training, validation, and testing partitions.

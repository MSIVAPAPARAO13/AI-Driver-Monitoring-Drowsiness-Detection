# Dataset Acknowledgments, Licensing & Governance
## AI Driver Monitoring & Drowsiness Detection System

**Document:** Comprehensive Dataset Provenance & Academic Acknowledgments  
**Location:** `docs/DATASET_ACKNOWLEDGMENTS.md`  
**Compliance Standard:** Ethical Academic Use, Boundary Separation & Licensing Integrity  

---

## 1. Governance Policy & Redistribution Notice

This project utilizes several publicly available academic datasets for research, algorithmic benchmarking, and validation. In accordance with ethical machine learning research practices and dataset access terms:

> [!IMPORTANT]
> **Redistribution Policy:** Raw third-party video and image datasets are **NOT** bundled or redistributed within this repository. All raw datasets are excluded via `.gitignore`. Users wishing to reproduce raw data ingestion must acquire the datasets directly from their respective academic authors and accept their license terms. Verify dataset-specific license before redistribution.

---

## 2. Dataset Attribution & Usage Boundaries

### 1. Original Baseline Dataset
- **Role:** Baseline comparison and historical audit.
- **Source:** Original open-source repository commit history.
- **Access Restrictions:** Open-source repository sample.
- **Redistribution:** Verify dataset-specific license before redistribution.
- **How It Was Used:** Extensively audited in Phase 1 and Phase 2B. Identified severe random-split identity leakage; filtered down to 24 clean frames and subsequently augmented with verified multi-source subjects.

### 2. YawDD (Yawning Detection Dataset)
- **Role:** Visual training and validation ground-truth expansion.
- **Source:** University of Ottawa (Abtahi et al., 2014, *ACM Multimedia Systems*).
- **Access Restrictions:** Released for academic, non-commercial research purposes.
- **Redistribution:** Non-commercial academic research. Verify dataset-specific license before redistribution.
- **How It Was Used:** Used to extract verified facial frames containing naturalistic yawn apertures, eye closures, and open eyes across male and female drivers with and without eyeglasses for Phase 2E/2F YOLO training and validation splits.

### 3. DMD (Driver Monitoring Dataset)
- **Role:** Multi-subject intra-class variation expansion.
- **Source:** Ortega et al., 2020 (*IEEE Intelligent Vehicles Symposium* / *IEEE Access*).
- **Access Restrictions:** Academic research access upon formal institutional request.
- **Redistribution:** Academic research only. Verify dataset-specific license before redistribution.
- **How It Was Used:** Provided diverse in-cabin driver frames under natural daylight and varied head poses to expand the verified training pool to 26 subjects.

### 4. UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)
- **Role:** Independent sequence-level temporal evaluation benchmark.
- **Source:** Ghoddoosian et al., 2019 (*IEEE/CVF CVPR Workshops*).
- **Access Restrictions:** Freely available for non-commercial academic research upon request.
- **Redistribution:** Academic use only. Verify dataset-specific license before redistribution.
- **How It Was Used (Strict Boundary):** **Explicitly NOT used for YOLO bounding box training.** UTA-RLDD was exclusively reserved as a held-out temporal evaluation benchmark, conducting 5-fold cross-subject validation across 180 multi-stage video sessions from 60 subjects to evaluate sequence-level drowsiness detection delay and temporal F1 scores.

### 5. NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection Dataset)
- **Role:** External out-of-distribution (OOD) robustness audit.
- **Source:** Weng et al., 2016 (*IEEE/CVF CVPR Workshops*).
- **Access Restrictions:** Academic research via signed license agreement.
- **Redistribution:** Strictly non-redistributable without explicit institutional agreement. Verify dataset-specific license before redistribution.
- **How It Was Used (Strict Boundary):** Exclusively utilized for out-of-distribution boundary testing (night driving, wearing sunglasses) to evaluate system failure modes. Not included in YOLO bounding box training.

### 6. MRL Eye Dataset
- **Role:** Ocular threshold calibration and synthetic landmark verification.
- **Source:** Media Research Lab, University of Maribor.
- **Access Restrictions:** Freely available for non-commercial research.
- **Redistribution:** Academic research. Verify dataset-specific license before redistribution.
- **How It Was Used (Strict Boundary):** Used solely for statistical calibration of EAR distribution percentiles and unit test synthetic edge cases. Not mixed into YOLO training or evaluation splits.

---

## 3. Academic Citations

```bibtex
@inproceedings{abtahi2014yawdd,
  title={YawDD: a yawning detection dataset},
  author={Abtahi, Shabnam and Omidyeganeh, Mona and Shirmohammadi, Shervin and Hariri, Behnoosh},
  booktitle={Proceedings of the 5th ACM Multimedia Systems Conference},
  pages={24--28},
  year={2014}
}

@inproceedings{ghoddoosian2019realistic,
  title={A realistic dataset and baseline temporal model for detecting driver drowsiness},
  author={Ghoddoosian, Reza and Galib, Mohammad Bashir and Athitsos, Vassilis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}

@article{ortega2020dmd,
  title={DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Drowsiness Analysis},
  author={Ortega, Javier and Kose, Neslihan and Ca{\~n}as, Pau and Chao, Marcos and Toupas, Alexandros and others},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2020}
}

@inproceedings{weng2016driver,
  title={Driver drowsiness detection via a hierarchical temporal deep model},
  author={Weng, Ching-Hua and Lai, Ying-Hsiu and Lai, Shang-Hong},
  booktitle={Asian Conference on Computer Vision Workshops},
  pages={117--133},
  year={2016}
}
```

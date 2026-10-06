# Technical Interview Preparation Guide
## AI Driver Monitoring & Safety System

**Project:** AI Driver Monitoring & Drowsiness Detection System  
**Document:** 20 Rigorous Interview Questions & Deep-Dive Technical Answers  
**Author:** AI Driver Monitoring Engineering Team  
**Date:** October 2026  

---

### 1. Why use YOLO for Driver Monitoring?
**Answer:** YOLO (specifically YOLOv5nu) offers lightweight (2.50M parameters, 4.99 MB) visual object detection with fast single-stage forward pass latency (~21 ms on CPU). It provides immediate spatial grounding for open eyes, closed eyes, and yawning mouths without requiring initial facial landmark alignment.

### 2. Why combine YOLO with EAR (Eye Aspect Ratio)?
**Answer:** YOLO operates on bounding boxes and cannot capture continuous sub-pixel eyelid geometric trajectories. Normal human blinks last 80–400 ms. If YOLO inferences every 3rd frame (~100 ms), it can easily alias a regular blink as a prolonged closure. EAR computes 6-point Euclidean landmark ratios on every frame at 60+ FPS, providing continuous microsecond tracking of eyelid opening.

### 3. Why use MAR (Mouth Aspect Ratio)?
**Answer:** YOLO detects yawning based on visual appearance, but visual mouth aperture alone cannot distinguish between yawning, singing, shouting, or regular speech. MAR calculates the vertical-to-horizontal lip ratio in 3D metric space, enabling duration tracking to filter out speech apertures.

### 4. Why use PERCLOS (Percentage of Eye Closure)?
**Answer:** In scientific automotive human factors research (Wierwille & Ellsworth, FHWA), PERCLOS is the gold-standard physiological metric for driver fatigue. We implement P80 PERCLOS over a rolling 60-second temporal window, measuring the proportion of time the driver's eyes are $\ge 80\%$ closed. This captures cumulative micro-drowsiness that instantaneous classifiers miss.

### 5. Why incorporate Head Pose estimation?
**Answer:** Drowsiness manifests not only ocularly but through vestibular microsleep nodding ("head drops") and sustained distraction away from the forward roadway. We estimate 3D Euler angles (Pitch, Yaw, Roll) using Perspective-n-Point (PnP) to detect downward head tilting and off-road gaze.

### 6. Why implement Temporal Multi-Signal Fusion?
**Answer:** Isolated instantaneous signals generate high false alarm rates. Our Tier 3 engine fuses visual YOLO confidence (35%), PERCLOS (25%), blink anomaly duration (15%), sustained yawn duration (15%), and head pose tilt (10%) into a 0–100 fatigue score governed by a finite state machine (`NORMAL` -> `WARNING` -> `CRITICAL` -> `RECOVERY`).

### 7. Why is a subject-level split mandatory?
**Answer:** Random frame-level train/test splitting causes catastrophic data leakage (~95%+ overlap), where identical facial structures, lighting, glasses, and backgrounds appear in both splits. Evaluating on the same person measures facial memorization, not generalization. We enforced strict subject-disjoint splits: 19 subjects in train, 4 in val, 3 in test, with 0 subject overlap.

### 8. Why was the original repository's dataset insufficient?
**Answer:** The original dataset had only 24 clean training frames, and its reported ~99% accuracy was an artifact of random frame-level splitting where contiguous video frames leaked across train and test. When evaluated on unseen subjects in Phase 2D, the baseline collapsed to F1 = 0.3111.

### 9. How was data leakage discovered and audited?
**Answer:** In Phase 2B/2C, we audited frame hashes, subject ID metadata, and temporal contiguousness across video frames. We discovered that adjacent frames from identical video segments were randomly scattered into train and validation directories.

### 10. Why did Phase 2D fail?
**Answer:** Phase 2D trained on only 24 original frames with heavy synthetic augmentation. Neural networks cannot learn rich intra-class intra-subject diversity (different eye shapes, facial hair, skin tones, lighting) from 24 images, resulting in AP50 = 0.00 for `eyes_open` and `yawning`.

### 11. Why did Phase 2E/2F dramatically improve model performance?
**Answer:** Phase 2E expanded verified ground-truth data to 333 clean frames and 341 bounding boxes across 25 subjects from YawDD, DMD, and verified baseline data, combined with a 50% train-only photometric augmentation pool. Retraining in Phase 2F lifted test mAP@0.5 from 0.2963 to 0.9777 and F1 from 0.3111 to 0.9189 on unseen subjects.

### 12. Why did Phase 2G multi-signal fusion increase F1 to 0.983?
**Answer:** The YOLO-only detector had 7 false alerts on benign video actions (speech, normal blinks). The Phase 2G fusion layer added speech suppression, 60s PERCLOS, and temporal persistence checks, eliminating all 7 false alerts while maintaining 100% micro-sleep recall.

### 13. Why is PyTorch faster than ONNX/OpenVINO on this Windows CPU host?
**Answer:** PyTorch 2.x includes fused CPU kernels (TorchScript / oneDNN) optimized for batch=1 single-frame low-latency dispatch (21.05 ms). ONNX Runtime (27.20 ms) and OpenVINO (54.88 ms) incurred higher single-frame dispatch and C++ wrapper memory copy overhead in Python 3.13 on this host.

### 14. Why export and support ONNX Runtime?
**Answer:** In production edge deployments (e.g. Raspberry Pi, Jetson, embedded Linux), deploying a full 2+ GB PyTorch stack is impractical. ONNX Runtime provides a lightweight (~50 MB) runtime without PyTorch dependencies, producing identical detection accuracy (mAP@0.5 = 0.9877) at 22.3 FPS.

### 15. Why was TensorRT not benchmarked?
**Answer:** The host environment is an Intel Core i7-10700 CPU workstation without an NVIDIA GPU. Under engineering integrity principles, we never fabricate benchmark numbers for unavailable hardware.

### 16. What was the biggest profiling bottleneck?
**Answer:** Profiling revealed that MediaPipe 3D face mesh extraction takes 16.28 ms per frame (40.6% of loop latency), while YOLO takes 14.14 ms amortized. In contrast, all physiological math (EAR, MAR, blink, PERCLOS, head pose) takes <0.40 ms combined.

### 17. How is false alert suppression handled?
**Answer:** We implemented a two-stage filter:
1. **Speech Suppression:** Yawn events require sustained mouth opening for $\ge 2.0\text{ s}$; apertures $<1.5\text{ s}$ are classified as speech and discarded.
2. **State Persistence Guard:** A warning requires $\ge 1.0\text{ s}$ of persistent elevated fatigue, preventing momentary blinks or yawns from immediately blaring critical alarms.

### 18. How is face loss handled?
**Answer:** If the driver turns completely around or leaves the camera FOV, a 1.0-second grace period is granted. If the face remains unobserved for $>2.0\text{ s}$, the system transitions to `FACE_LOST` and triggers a distracted-driver alert.

### 19. How is driver privacy protected?
**Answer:** Video frames are processed strictly in volatile RAM and immediately discarded unless video recording is explicitly toggled. No facial recognition or driver biometric identity embeddings are generated. Telemetry logs contain only anonymized numerical fatigue scores and timestamps.

### 20. What would you improve next in Phase 3?
**Answer:**
1. Integrate near-infrared (NIR) camera support for night driving.
2. Quantize ONNX to INT8 using Post-Training Quantization (PTQ) for embedded ARM microcontrollers.
3. Incorporate steering wheel capacitive touch sensor fusion (CAN bus integration).

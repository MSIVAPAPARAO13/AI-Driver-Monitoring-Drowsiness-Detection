# External Video Failure Analysis & Edge-Case Audit

## 1. Edge-Case Matrix

| Scenario | Challenge | Pipeline Response | Mitigation in Codebase |
| :--- | :--- | :--- | :--- |
| **Eyeglasses Reflection** | Specular glare on eye lenses | YOLOv5nu detects eye region; EAR fallback | Multi-signal fusion balances YOLO confidence with EAR |
| **Speech vs Yawn** | Fast conversational mouth opening | MAR spikes briefly (< 0.8s) | `YawnDurationAnalyzer` requires sustained >= 1.5s dilation |
| **Night Driving (< 10 lux)** | Low-contrast facial features | Laplacian variance drops; drift flag | Drift monitor logs environmental shift; YOLO maintains bounding box |
| **Rapid Head Turn (> 60 deg)** | Partial face landmark occlusion | Landmark confidence decreases | Head pose estimator triggers `HEAD_LEFT`/`HEAD_RIGHT` flag |
| **Camera Disconnect / Exit** | Driver exits frame temporarily | `face_detected = False` | Temporal fusion applies 1.0s grace period before `FACE_LOST` state |
| **Vertical Mobile (720x1280)** | Unusual aspect ratio | Box coordinates scale to display | Coordinate bounds clamping prevents border bleeding |

## 2. Bounding Box Boundary Audit
In prior iterations, bounding boxes near frame edges risk text truncation if the label tag was drawn above `y1 = 0`. This was eliminated by dynamic tag placement:
- When `y1 - text_h - 10 < 0`, the label tag renders *inside* the box below `y1`.
- When `x1 + text_w > w`, the label tag is clamped to `w - text_w - 6`.

## 3. HUD Layout Collision Audit
On compact or standard 640x480 webcam streams, the bottom-left telemetry panel (`x: [10, 370]`) previously collided with the bottom-right physiological panel (`x: [270, 630]`).
- **Fix Implemented**: Responsive layout logic. For `w < 780`, the telemetry panel stays bottom-left while the physiological signal HUD is placed at top-right, eliminating collision entirely.

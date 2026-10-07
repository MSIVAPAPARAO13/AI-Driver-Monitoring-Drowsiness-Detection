# Live Webcam Manual Testing & User Experience Audit

## Test Setup
- **App**: `streamlit run app/app.py` on `http://localhost:8501`
- **Streamer**: `streamlit-webrtc` (SENDRECV mode, 640x480 resolution)
- **Subjects Tested**: Multiple driving postures, varied distances (0.4m to 1.5m), ambient daylight & indoor illumination.

## Functional Step-by-Step Verification

1. **Camera Initiation**:
   - Clicking **'START'** requests browser media permissions cleanly.
   - Stream starts within ~350ms of permission grant.
2. **Face Acquisition**:
   - Immediate bounding box placement around facial regions.
   - MediaPipe 478-point mesh tracks eyes and lips smoothly.
3. **Blink Analysis**:
   - Normal blinks (100–300ms) increment `blink_count` without false alerts.
   - Prolonged eye closure (> 1.2s) transitions state: `NORMAL -> WARNING -> CRITICAL`.
4. **Yawn Differentiation**:
   - Talking and laughter are suppressed.
   - Prolonged mouth opening (> 1.5s) correctly triggers `YAWN DETECTED` event.
5. **Head Posture**:
   - Looking forward: `HEAD_FORWARD`.
   - Looking left/right (> 28 deg): `HEAD_LEFT` / `HEAD_RIGHT`.
6. **Recommendations**:
   - `NORMAL`: "You appear alert. Continue monitoring."
   - `WARNING`: "Signs of fatigue detected. Stay focused and consider taking a safe break if this continues."
   - `CRITICAL`: "High fatigue indicators detected. Stop at a safe location and rest before continuing."
   - `FACE_LOST`: "Face not clearly visible. Adjust camera position."
   - Strictly non-diagnostic; zero medical claims.
7. **Session Termination**:
   - Clicking **'STOP'** releases camera device cleanly without thread hang.
   - Session Summary card displays total duration, frames processed, average FPS, peak fatigue score, alerts, and face-lost seconds.

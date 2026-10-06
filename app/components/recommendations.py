"""
Driver Safety Feedback & Recommendation Engine.

Translates real-time physiological signals and state machine events
into non-diagnostic, safety-oriented driver advisories.
"""

from typing import Dict, Any, Tuple


def get_driver_recommendation(
    state: str,
    fatigue_score: float,
    ear: float,
    mar: float,
    perclos: float,
    head_pose_state: str,
    eye_closed_duration: float = 0.0,
    yawn_duration: float = 0.0,
    faces_detected: int = 1,
) -> Tuple[str, str, str, list[str]]:
    """
    Generate safety-oriented driver recommendations based strictly on runtime telemetry.

    Returns:
        title (str): Concise advisory headline.
        message (str): Detailed action guidance.
        severity (str): Alert severity level ('success', 'info', 'warning', 'danger').
        tips (list[str]): Contextual bullet points.
    """
    state_upper = state.upper() if state else "NORMAL"
    tips = []

    # 1. State-level core guidance
    if state_upper == "CRITICAL":
        title = "CRITICAL FATIGUE DETECTED"
        message = (
            "High fatigue indicators detected. Please stop driving at a safe location "
            "and rest before continuing your journey."
        )
        severity = "danger"
        tips.append("Pull over safely at the next rest area or service station.")
        tips.append("Engage in a 15–20 minute power nap or switch drivers.")

    elif state_upper == "WARNING":
        title = "FATIGUE ONSET DETECTED"
        message = (
            "Attention: prolonged signs of fatigue were detected. Stay focused and "
            "consider taking a break if the condition continues."
        )
        severity = "warning"
        tips.append("Increase cabin airflow or adjust climate control.")
        tips.append("Plan a brief rest stop within the next 15–30 minutes.")

    elif state_upper == "FACE_LOST" or faces_detected == 0:
        title = "DRIVER FACE NOT DETECTED"
        message = (
            "Driver face is not clearly visible. Please adjust camera position, angle, "
            "and ambient lighting to maintain continuous monitoring."
        )
        severity = "warning"
        tips.append("Ensure your face is centered inside the camera frame.")
        tips.append("Check that cabin lighting illuminates the face without direct glare.")

    elif state_upper == "RECOVERY":
        title = "VIGILANCE RECOVERING"
        message = (
            "Alertness indicators have improved. Continue monitoring and maintain "
            "safe driving habits."
        )
        severity = "info"
        tips.append("Maintain forward gaze on the roadway.")
        tips.append("Ensure sustained alertness before increasing travel speed.")

    else:  # NORMAL
        title = "DRIVER ALERT & ATTENTIVE"
        message = (
            "You appear alert. Continue monitoring and maintain safe driving habits."
        )
        severity = "success"
        tips.append("Maintain consistent speed and safe following distances.")
        tips.append("Recommended rule: take a 15-minute break every 2 hours of driving.")

    # 2. Contextual signal-specific advisories
    if eye_closed_duration >= 0.5:
        tips.append(
            f"Ocular alert: Eyes closed for {eye_closed_duration:.1f}s. "
            "Refocus immediately on the forward roadway."
        )
    elif ear < 0.20 and state_upper != "CRITICAL":
        tips.append("Eye aperture is narrowed. Ensure eyes remain open and vigilant.")

    if yawn_duration >= 2.0:
        tips.append(
            f"Oral alert: Sustained yawn detected ({yawn_duration:.1f}s). "
            "Frequent yawning is an early physiological sign of fatigue."
        )

    if perclos >= 0.20:
        tips.append(
            f"Temporal alert: Elevated PERCLOS ({perclos*100:.1f}%). "
            "Cumulative eye-closure activity indicates declining vigilance."
        )

    if head_pose_state and head_pose_state not in ["HEAD_FORWARD", "FORWARD", "NORMAL"]:
        if "DOWN" in head_pose_state:
            tips.append("Head orientation: Downward pitch detected. Keep head upright.")
        elif "LEFT" in head_pose_state or "RIGHT" in head_pose_state:
            tips.append("Attention orientation: Head turned off-road. Direct gaze forward.")

    return title, message, severity, tips

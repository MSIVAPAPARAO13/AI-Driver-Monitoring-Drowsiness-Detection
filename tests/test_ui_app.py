"""
UI and Widget Key Integrity Tests using Streamlit AppTest.
Tests application startup, 5 workspaces, widget uniqueness, and session state.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = str(REPO_ROOT / "app" / "app.py")


def test_streamlit_app_startup_and_tabs():
    """Verify application starts with 0 uncaught exceptions and renders all 5 tabs."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"
    assert len(at.tabs) == 5
    expected_tabs = [
        "📺 Live Monitor",
        "📁 Video Analysis",
        "🧠 Continual Learning",
        "📊 Model Performance",
        "🏗️ Architecture & Privacy",
    ]
    for i, exp in enumerate(expected_tabs):
        assert at.tabs[i].label == exp


def test_widget_keys_uniqueness_and_stability():
    """Verify all interactive widgets across all tabs have unique and stable keys."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    all_keys = []
    # Collect button keys
    for b in at.button:
        assert b.key is not None, f"Button {b.label} missing explicit key"
        all_keys.append(b.key)

    # Collect checkbox keys
    for c in at.checkbox:
        assert c.key is not None, f"Checkbox {c.label} missing explicit key"
        all_keys.append(c.key)

    # Collect radio keys
    for r in at.radio:
        if r.key:
            all_keys.append(r.key)

    # Check for duplicate keys
    duplicates = [k for k in all_keys if all_keys.count(k) > 1]
    assert len(duplicates) == 0, f"Found duplicate widget keys: {set(duplicates)}"


def test_sidebar_toggles_and_cl_state():
    """Verify toggling display options and continual learning operates cleanly."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Toggle continual learning on
    cl_cb = next((c for c in at.checkbox if c.key == "sidebar_cl_enabled"), None)
    if cl_cb:
        cl_cb.check().run()
        assert len(at.exception) == 0

    # Toggle bounding boxes off then on
    box_cb = next((c for c in at.checkbox if c.key == "sidebar_show_boxes"), None)
    if box_cb:
        box_cb.uncheck().run()
        assert len(at.exception) == 0
        box_cb.check().run()
        assert len(at.exception) == 0


def test_live_tab_buttons_and_session_state():
    """Verify Live Monitor action buttons operate without exceptions."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Click Reset Session Telemetry
    reset_btn = next((b for b in at.button if b.key == "live_reset_telemetry_btn"), None)
    if reset_btn:
        reset_btn.click().run()
        assert len(at.exception) == 0

    # Click Clear Alert History
    clear_btn = next((b for b in at.button if b.key == "live_clear_alert_history_btn"), None)
    if clear_btn:
        clear_btn.click().run()
        assert len(at.exception) == 0


def test_continual_learning_tab_actions():
    """Verify Continual Learning workspace controls run without error."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Click clear learning buffer button
    buf_btn = next((b for b in at.button if b.key == "cl_clear_buffer_btn"), None)
    if buf_btn:
        buf_btn.click().run()
        assert len(at.exception) == 0

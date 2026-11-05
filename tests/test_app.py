"""
Tests for app.py and user-interface contracts.
Validates code integrity, cached pipeline generation, and strict typography rules.
"""

from pathlib import Path
import py_compile
import pytest

from surf_camera.pipeline import PipelineConfig
from surf_camera.config import resolve_model_path


def test_app_compiles_cleanly():
    """Ensures app.py compiles without syntax or indentation errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists(), "app.py does not exist"
    # py_compile raises an exception if compilation fails
    py_compile.compile(str(app_path), doraise=True)


def test_no_double_dash_guardrail():
    """Permanent guardrail: neither app.py nor README.md should contain double dashes."""
    root = Path(__file__).resolve().parent.parent
    app_text = (root / "app.py").read_text(encoding="utf-8")
    readme_text = (root / "README.md").read_text(encoding="utf-8")

    assert "--" not in app_text, "Found forbidden double dash in app.py"
    assert "--" not in readme_text, "Found forbidden double dash in README.md"


def test_no_deprecated_residues_in_app():
    """Ensures obsolete VAE, pose figures, and split-screen logic are removed from app.py."""
    root = Path(__file__).resolve().parent.parent
    app_text = (root / "app.py").read_text(encoding="utf-8")

    # Forbidden legacy concepts
    assert "JointScaleVAE" not in app_text
    assert "maneuver_classifier" not in app_text
    assert "split_screen" not in app_text.lower() or "save_split_screen=False" in app_text
    assert "yolov8n-pose.pt" not in app_text


def test_cached_pipeline_construction():
    """Validates that get_cached_pipeline constructs a working pipeline instance."""
    from app import get_cached_pipeline

    pipe = get_cached_pipeline(
        model_type="surf_v2",
        aspect_ratio="9:16",
        surfer_height_m=1.75,
        min_ride_duration_sec=1.5,
    )
    assert pipe is not None
    assert pipe.config.aspect_ratio == "9:16"
    assert pipe.config.surfer_height_m == 1.75
    assert pipe.config.min_ride_duration_sec == 1.5
    assert pipe.config.model_type == "surf_v2"
    assert pipe.config.enable_pose_estimation is False

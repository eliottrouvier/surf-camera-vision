"""
Unit tests for wave face segmentation, physical height calibration,
and morphological behavior qualification (Droite qui ouvre, Gauche qui ouvre, Fermeture/Close-out).
"""

import pytest
import numpy as np

from surf_camera.wave_analyzer import (
    WaveSegmentationEngine,
    WaveMetrics,
    classify_wave_size,
    classify_wave_behavior,
)


def test_classify_wave_size_scale():
    """Validates categorization against standard WSL surfing scale."""
    assert classify_wave_size(0.50) == "Knee High (<2ft)"
    assert classify_wave_size(0.85) == "Waist High (2-3ft)"
    assert classify_wave_size(1.20) == "Chest High (3-4ft)"
    assert classify_wave_size(1.60) == "Head High (4.5-6ft)"
    assert classify_wave_size(2.10) == "Overhead (6-8ft)"
    assert classify_wave_size(2.80) == "Double Overhead (>8ft)"


def test_classify_wave_behavior_direct():
    """Validates rule-based wave behavior classification."""
    assert classify_wave_behavior(direction="RIGHT") == "Droite qui ouvre"
    assert classify_wave_behavior(direction="LEFT") == "Gauche qui ouvre"
    assert classify_wave_behavior(direction="RIGHT", is_closeout_hint=True) == "Fermeture (Close-out)"
    assert classify_wave_behavior(
        direction="RIGHT",
        foam_uniformity=0.90,
        lateral_speed_ratio=0.15,
    ) == "Fermeture (Close-out)"


def test_wave_height_estimation_synthetic_1m():
    """
    Validates that on a synthetic wave calibrated to 1.0 m,
    the measured physical height is strictly between 0.85 m and 1.15 m.
    """
    engine = WaveSegmentationEngine(smoothing_alpha=1.0)

    # Frame resolution: 720x1280
    frame = np.full((720, 1280, 3), (130, 80, 25), dtype=np.uint8)

    # Surfer standing from y=240 to y=320 (height = 80 px)
    # Surfer scale: 1.75 m / 80 px = 0.021875 meters/px
    meters_per_px = 1.75 / 80.0
    surfer_bbox = (500.0, 240.0, 530.0, 320.0)

    # Wave crest: horizontal foam band located ~46 px above surfer base (320 - 46 = 274)
    # Target height: ~46 px * 0.021875 m/px = ~1.00 m
    frame[274:295, 350:750] = (250, 250, 250)

    metrics = engine.analyze(
        frame=frame,
        surfer_bbox=surfer_bbox,
        direction="RIGHT",
        meters_per_pixel=meters_per_px,
    )

    assert isinstance(metrics, WaveMetrics)
    assert 0.85 <= metrics.height_meters <= 1.15
    assert metrics.category in ("Waist High (2-3ft)", "Chest High (3-4ft)")


def test_morphological_behavior_droite_qui_ouvre():
    """
    Wave with breaking foam on the left (the pocket) and clean unbroken water on the right.
    Should classify as 'Droite qui ouvre'.
    """
    engine = WaveSegmentationEngine()
    frame = np.full((400, 600, 3), (120, 80, 20), dtype=np.uint8)

    # Whitewater concentrated in left sector only (pocket curling right)
    frame[100:250, :200] = (245, 245, 250)

    behavior = engine.evaluate_wave_behavior(frame, direction="RIGHT")
    assert behavior == "Droite qui ouvre"


def test_morphological_behavior_gauche_qui_ouvre():
    """
    Wave with breaking foam on the right (the pocket) and clean unbroken water on the left.
    Should classify as 'Gauche qui ouvre'.
    """
    engine = WaveSegmentationEngine()
    frame = np.full((400, 600, 3), (120, 80, 20), dtype=np.uint8)

    # Whitewater concentrated in right sector only (pocket curling left)
    frame[100:250, 400:] = (245, 245, 250)

    behavior = engine.evaluate_wave_behavior(frame, direction="LEFT")
    assert behavior == "Gauche qui ouvre"


def test_morphological_behavior_closeout():
    """
    Wave where the entire crest collapses simultaneously across the whole width of the section.
    Should classify as 'Fermeture (Close-out)'.
    """
    engine = WaveSegmentationEngine()
    frame = np.full((400, 600, 3), (120, 80, 20), dtype=np.uint8)

    # Thick foam curtain spanning across left, center, and right sectors simultaneously
    frame[80:260, :] = (250, 250, 250)

    behavior = engine.evaluate_wave_behavior(frame, direction="RIGHT")
    assert behavior == "Fermeture (Close-out)"

"""
Unit tests for JointScaleVAE, MetricScaleEstimator, and WaveSegmentationEngine.
Validates metric scale estimation, physical wave height computation, and badge rendering.
"""

import pytest
import numpy as np
import torch

from surf_camera.scale_estimator import JointScaleVAE, MetricScaleEstimator, ScaleEstimate
from surf_camera.wave_analyzer import WaveSegmentationEngine, classify_wave_size, WaveMetrics
from surf_camera.utils import draw_minimalist_badge
from surf_camera.state_machine import RideEvent


def test_joint_scale_vae_forward():
    vae = JointScaleVAE(input_dim=6, latent_dim=4)
    x = torch.randn(2, 6)
    recon, mu, logvar, pred_scale = vae(x)

    assert recon.shape == (2, 6)
    assert mu.shape == (2, 4)
    assert logvar.shape == (2, 4)
    assert pred_scale.shape == (2, 2)


def test_metric_scale_estimator():
    estimator = MetricScaleEstimator(nominal_surfer_height_m=1.75, nominal_board_length_m=1.83)
    
    # Surfer standing (height=80px, width=28px)
    surfer_bbox = (200.0, 300.0, 228.0, 380.0)
    board_bbox = (190.0, 375.0, 245.0, 395.0)

    est = estimator.update(surfer_bbox=surfer_bbox, board_bbox=board_bbox, is_standing_hint=True)
    
    assert isinstance(est, ScaleEstimate)
    assert est.meters_per_pixel > 0.005
    assert est.meters_per_pixel < 0.10
    assert est.is_upright is True
    assert est.latent_z is not None


def test_classify_wave_size():
    assert "Knee" in classify_wave_size(0.4)
    assert "Waist" in classify_wave_size(0.8)
    assert "Chest" in classify_wave_size(1.2)
    assert "Head High" in classify_wave_size(1.6)
    assert "Overhead" in classify_wave_size(2.1)
    assert "Double Overhead" in classify_wave_size(3.0)


def test_wave_segmentation_engine():
    engine = WaveSegmentationEngine()
    
    # Create synthetic ocean frame with water (blue/green) and foam crest (white)
    frame = np.full((720, 1280, 3), (120, 80, 20), dtype=np.uint8)
    # White foam crest band
    frame[250:280, 400:800] = (245, 245, 245)

    surfer_bbox = (500.0, 320.0, 530.0, 400.0)
    meters_per_px = 0.02

    metrics = engine.analyze(
        frame=frame,
        surfer_bbox=surfer_bbox,
        direction="RIGHT",
        meters_per_pixel=meters_per_px,
    )

    assert isinstance(metrics, WaveMetrics)
    assert metrics.height_meters > 0.5
    assert metrics.height_feet > 1.5
    assert metrics.crest_y < metrics.trough_y


def test_minimalist_badge_rendering():
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    badged = draw_minimalist_badge(
        frame,
        speed_kmh=27.4,
        wave_height_m=1.65,
        distance_m=42.0,
        ride_time_sec=3.2,
    )

    assert badged.shape == frame.shape
    assert not np.array_equal(badged, frame)


def test_ride_event_with_metric_fields():
    ride = RideEvent(
        ride_id=1,
        surfer_id=1,
        start_frame=30,
        takeoff_frame=45,
        end_frame=150,
        duration_frames=120,
        duration_sec=4.0,
        max_speed_px=8.5,
        direction="RIGHT",
        trajectory=[(100.0, 200.0, 30), (200.0, 250.0, 150)],
        max_speed_kmh=28.5,
        avg_speed_kmh=21.2,
        distance_meters=45.6,
        wave_height_meters=1.65,
        wave_height_feet=5.4,
        wave_category="Head High (4.5-6ft)",
    )

    assert ride.max_speed_kmh == 28.5
    assert ride.distance_meters == 45.6
    assert ride.wave_height_meters == 1.65
    assert ride.wave_category == "Head High (4.5-6ft)"

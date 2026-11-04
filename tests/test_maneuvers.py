"""
Unit tests for Surf AI v2 Pose Estimation and Maneuver Classifier.
Tests geometric formulas, angle calculations, and maneuver thresholds.
"""

import numpy as np
import pytest
from surf_camera.pose_estimator import compute_angle_3pts, SurferPose
from surf_camera.maneuver_classifier import (
    ManeuverClassifier,
    ManeuverType,
    ManeuverDetection,
    compute_wsl_score,
)
from surf_camera.wave_analyzer import WaveMetrics
from surf_camera.utils import draw_neon_skeleton, draw_biomechanical_lab


def test_compute_angle_3pts_geometry():
    # 90 degrees right angle: (0, 1) -> (0, 0) -> (1, 0)
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    angle = compute_angle_3pts(a, b, c)
    assert abs(angle - 90.0) < 1e-3

    # 180 degrees straight line: (-1, 0) -> (0, 0) -> (1, 0)
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    angle = compute_angle_3pts(a, b, c)
    assert abs(angle - 180.0) < 1e-3

    # Degenerate points (zero length)
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    angle = compute_angle_3pts(a, b, c)
    assert angle == 180.0


def test_surfer_pose_dataclass():
    kpts = np.zeros((17, 3), dtype=np.float32)
    crop_kpts = np.zeros((17, 3), dtype=np.float32)
    pose = SurferPose(
        keypoints=kpts,
        crop_keypoints=crop_kpts,
        crop_bbox=(10, 10, 100, 100),
        confidence=0.85,
        knee_angle_deg=115.0,
        torso_lean_deg=-32.5,
        compression_ratio=0.55,
        shoulder_tilt_deg=12.0,
    )
    assert pose.confidence == 0.85
    assert pose.knee_angle_deg == 115.0
    assert pose.torso_lean_deg == -32.5
    assert pose.compression_ratio == 0.55


def test_maneuver_classifier_bottom_turn():
    classifier = ManeuverClassifier(fps=30.0)
    # Surfer at bottom of wave (norm_wave_pos > 0.55), high lean (35 deg), compressed knees (110 deg)
    kpts = np.zeros((17, 3), dtype=np.float32)
    pose = SurferPose(
        keypoints=kpts,
        crop_keypoints=kpts,
        crop_bbox=(0, 0, 50, 50),
        confidence=0.9,
        knee_angle_deg=110.0,
        torso_lean_deg=35.0,
        compression_ratio=0.52,
        shoulder_tilt_deg=10.0,
    )
    wave = WaveMetrics(
        height_meters=2.0,
        height_feet=6.5,
        category="Overhead (6-8ft)",
        crest_y=100.0,
        trough_y=300.0,
    )
    # Surfer at y = 260 -> (260 - 100) / 200 = 0.80 (deep in wave trough)
    det = classifier.update(
        pose=pose,
        wave=wave,
        surfer_center=(400.0, 260.0),
        velocity=(8.0, -2.0),
        speed_kmh=28.0,
    )
    assert det.maneuver_type == ManeuverType.BOTTOM_TURN
    assert det.is_active is True
    assert "Bottom Turn" in det.name
    assert det.score_out_of_10 >= 7.0


def test_maneuver_classifier_roller_snap():
    classifier = ManeuverClassifier(fps=30.0)
    kpts = np.zeros((17, 3), dtype=np.float32)

    wave = WaveMetrics(
        height_meters=2.0,
        height_feet=6.5,
        category="Overhead (6-8ft)",
        crest_y=100.0,
        trough_y=300.0,
    )

    # Feed history of poses with rapid shoulder rotation
    for tilt in [0.0, 5.0, 12.0, 25.0, 32.0]:
        pose = SurferPose(
            keypoints=kpts,
            crop_keypoints=kpts,
            crop_bbox=(0, 0, 50, 50),
            confidence=0.88,
            knee_angle_deg=120.0,
            torso_lean_deg=-28.0,
            compression_ratio=0.60,
            shoulder_tilt_deg=tilt,
        )
        # Surfer at y = 130 -> (130 - 100) / 200 = 0.15 (at the crest!)
        det = classifier.update(
            pose=pose,
            wave=wave,
            surfer_center=(400.0, 130.0),
            velocity=(5.0, 3.0),
            speed_kmh=26.0,
        )

    assert det.maneuver_type == ManeuverType.ROLLER
    assert "Roller" in det.name
    assert det.is_active is True


def test_maneuver_classifier_tube_barrel():
    classifier = ManeuverClassifier(fps=30.0)
    kpts = np.zeros((17, 3), dtype=np.float32)

    pose = SurferPose(
        keypoints=kpts,
        crop_keypoints=kpts,
        crop_bbox=(0, 0, 50, 50),
        confidence=0.85,
        knee_angle_deg=72.0,       # Hyper-flexion < 85°
        torso_lean_deg=10.0,
        compression_ratio=0.38,    # Compression < 0.48
        shoulder_tilt_deg=5.0,
    )
    wave = WaveMetrics(
        height_meters=2.5,
        height_feet=8.2,
        category="Double Overhead (>8ft)",
        crest_y=100.0,
        trough_y=300.0,
    )

    # Tube requires continuous frames of deep compression
    det = None
    for _ in range(8):
        det = classifier.update(
            pose=pose,
            wave=wave,
            surfer_center=(400.0, 180.0), # Under crest
            velocity=(12.0, 0.0),
            speed_kmh=32.0,
        )

    assert det is not None
    assert det.maneuver_type == ManeuverType.TUBE
    assert "Tube" in det.name
    assert det.is_active is True


def test_draw_neon_skeleton():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Set keypoint coordinates with high confidence
    for i in range(17):
        kpts[i] = [100.0 + i * 5, 100.0 + i * 5, 0.9]

    out = draw_neon_skeleton(frame, kpts, color=(0, 255, 200))
    assert out.shape == frame.shape
    assert np.any(out > 0)


def test_draw_biomechanical_lab():
    kpts = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        kpts[i] = [50.0 + i * 2, 50.0 + i * 2, 0.8]

    pose = SurferPose(
        keypoints=kpts,
        crop_keypoints=kpts,
        crop_bbox=(0, 0, 100, 100),
        confidence=0.85,
        knee_angle_deg=92.0,
        torso_lean_deg=34.0,
        compression_ratio=0.45,
        shoulder_tilt_deg=18.0,
    )
    det = ManeuverDetection(
        maneuver_type=ManeuverType.BOTTOM_TURN,
        name="🔥 Bottom Turn",
        confidence=0.92,
        score_out_of_10=8.8,
        g_force=2.7,
        knee_flexion_deg=92.0,
        torso_lean_deg=34.0,
        description="Bottom turn puissant",
        is_active=True,
    )

    lab_pane = draw_biomechanical_lab(
        canvas_w=920,
        canvas_h=1040,
        pose=pose,
        maneuver=det,
        speed_kmh=29.4,
        wave_height_m=1.85,
    )
    assert lab_pane.shape == (1040, 920, 3)
    assert np.any(lab_pane > 13)


def test_compute_wsl_score_poor_neutral_glide():
    # A surfer surfing straight with stiff legs (> 150 deg) on a 1.0m wave
    score, cat, desc = compute_wsl_score(
        maneuvers_detected=[],
        duration_sec=7.0,
        peak_speed_kmh=11.0,
        wave_height_m=1.0,
        mean_knee_angle_deg=171.0,
    )
    # WSL Poor range is strictly [0.0, 1.9]
    assert 0.4 <= score <= 1.5
    assert "Poor" in cat
    assert "posture" in desc or "Glisse" in desc


def test_compute_wsl_score_committed_combo():
    # A surfer linking a bottom turn and a snap
    bt = ManeuverDetection(
        maneuver_type=ManeuverType.BOTTOM_TURN,
        name="🔥 Bottom Turn",
        confidence=0.9,
        score_out_of_10=7.5,
        g_force=2.2,
        knee_flexion_deg=105.0,
        torso_lean_deg=32.0,
        description="Bottom turn",
        is_active=True,
    )
    snap = ManeuverDetection(
        maneuver_type=ManeuverType.ROLLER,
        name="⚡ Roller / Snap",
        confidence=0.88,
        score_out_of_10=8.0,
        g_force=2.5,
        knee_flexion_deg=112.0,
        torso_lean_deg=-26.0,
        description="Snap",
        is_active=True,
    )
    score, cat, desc = compute_wsl_score(
        maneuvers_detected=[bt, snap],
        duration_sec=8.5,
        peak_speed_kmh=24.5,
        wave_height_m=1.8,
        mean_knee_angle_deg=115.0,
    )
    assert score >= 6.5
    assert ("Very Good" in cat or "Good" in cat or "Excellent" in cat)


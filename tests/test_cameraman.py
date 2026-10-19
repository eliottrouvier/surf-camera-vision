"""
Unit tests for VirtualCameraman auto-framing and lead-room calculations.
"""

import pytest
from surf_camera.cameraman import VirtualCameraman, CameraBox


def test_cameraman_initialization():
    cam = VirtualCameraman(frame_width=1920, frame_height=1080, aspect_ratio="16:9")
    assert cam.target_ar == pytest.approx(16.0 / 9.0)

    cam_vertical = VirtualCameraman(frame_width=1920, frame_height=1080, aspect_ratio="9:16")
    assert cam_vertical.target_ar == pytest.approx(9.0 / 16.0)


def test_lead_room_rightward():
    """Surfer moving rightwards should offset the camera to the right (surfer in left 1/3)."""
    cam = VirtualCameraman(frame_width=1920, frame_height=1080, aspect_ratio="16:9", lead_room_factor=0.35)
    surfer_bbox = (500.0, 400.0, 560.0, 520.0)
    surfer_cx = (500.0 + 560.0) / 2.0  # 530.0

    # Moving right with vx = 10.0 px/frame
    box = cam.update(surfer_bbox, surfer_velocity=(10.0, 0.0))
    cam_cx, _ = box.center

    # Camera center should be shifted to the right of the surfer
    assert cam_cx > surfer_cx


def test_lead_room_leftward():
    """Surfer moving leftwards should offset the camera to the left (surfer in right 1/3)."""
    cam = VirtualCameraman(frame_width=1920, frame_height=1080, aspect_ratio="16:9", lead_room_factor=0.35)
    surfer_bbox = (1200.0, 400.0, 1260.0, 520.0)
    surfer_cx = (1200.0 + 1260.0) / 2.0  # 1230.0

    # Moving left with vx = -10.0 px/frame
    box = cam.update(surfer_bbox, surfer_velocity=(-10.0, 0.0))
    cam_cx, _ = box.center

    # Camera center should be shifted to the left of the surfer
    assert cam_cx < surfer_cx


def test_boundary_clamping():
    """Camera crop window should never exceed the original frame boundaries."""
    cam = VirtualCameraman(frame_width=1280, frame_height=720, aspect_ratio="16:9")
    # Surfer at the extreme top-left corner
    surfer_bbox = (10.0, 10.0, 50.0, 90.0)
    box = cam.update(surfer_bbox, surfer_velocity=(-5.0, -5.0))

    assert box.x1 >= 0
    assert box.y1 >= 0
    assert box.x2 <= 1280
    assert box.y2 <= 720


def test_smoothing_damping():
    """Consecutive updates should smoothly interpolate without abrupt jumps."""
    cam = VirtualCameraman(frame_width=1920, frame_height=1080, smoothing_factor=0.10)
    box1 = cam.update((500.0, 500.0, 550.0, 600.0))
    # Jump surfer abruptly by 200 pixels
    box2 = cam.update((700.0, 500.0, 750.0, 600.0))

    c1 = box1.center[0]
    c2 = box2.center[0]
    # Center shift should be smoothed (around 10% of 200px = ~20px, not 200px)
    assert abs(c2 - c1) < 40.0

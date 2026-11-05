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


def test_tube_safety_zoom_out():
    """
    Validates the tube safety mode:
    1. Surfer rides rightward and is smoothly framed.
    2. Surfer disappears into the barrel (surfer_bbox=None, is_riding=True).
    3. Cameraman maintains forward motion in glide direction and widens frame by 25% to 35%.
    4. Surfer re-emerges at spit: cameraman tightens frame back smoothly.
    """
    cam = VirtualCameraman(
        frame_width=1920,
        frame_height=1080,
        aspect_ratio="16:9",
        zoom_scale=4.5,
        smoothing_factor=0.15,
        tube_safety_zoom_out=0.30,  # 30% enlargement
    )

    # 1. Visible surfer riding rightward
    start_x, start_y = 500.0, 500.0
    vx, vy = 8.0, 0.5
    for f in range(15):
        box_init = cam.update_with_tube_safety(
            surfer_bbox=(start_x + f * vx, start_y, start_x + f * vx + 30.0, start_y + 70.0),
            last_velocity=(vx, vy),
            is_riding=True,
        )

    w_initial = box_init.width
    cx_initial = box_init.center[0]
    assert cam.is_in_tube_safety is False

    # 2. Surfer enters the tube (occlusion under lip)
    for f in range(25):
        box_tube = cam.update_with_tube_safety(
            surfer_bbox=None,
            last_velocity=(vx, vy),
            is_riding=True,
        )

    # Verify tube mode was engaged
    assert cam.is_in_tube_safety is True
    assert cam.tube_frames_counter == 25
    assert cam.tube_zoom_factor > 1.20

    # Verify smooth zoom-out: crop box expanded by 20% to 35%
    assert box_tube.width >= w_initial * 1.15
    # Verify forward tracking: camera continued moving rightwards in the glide direction
    assert box_tube.center[0] > cx_initial

    # 3. Surfer re-emerges at spit
    exit_x = start_x + 40 * vx
    box_exit = cam.update_with_tube_safety(
        surfer_bbox=(exit_x, start_y, exit_x + 30.0, start_y + 70.0),
        last_velocity=(vx, vy),
        is_riding=True,
    )
    assert cam.is_in_tube_safety is False

    # Continue riding for 15 frames: frame should smoothly tighten back
    for f in range(15):
        box_tight = cam.update_with_tube_safety(
            surfer_bbox=(exit_x + f * vx, start_y, exit_x + f * vx + 30.0, start_y + 70.0),
            last_velocity=(vx, vy),
            is_riding=True,
        )

    assert box_tight.width < box_tube.width

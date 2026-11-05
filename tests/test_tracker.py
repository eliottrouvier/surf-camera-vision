"""
Unit tests for SurferTracker, tube occlusion memory, and baseline flotation kinematics.
"""

import pytest
import numpy as np

from surf_camera.tracker import SurferTracker, Track
from surf_camera.detector import Detection


def create_detection(cx: float, cy: float, w: float, h: float, conf: float = 0.90) -> Detection:
    """Creates a Detection instance centered at (cx, cy)."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return Detection(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        class_id=0,
        class_name="surfer",
    )


def test_tracker_initialization_and_confirmation():
    """Validates that a track is created and confirmed after min_hits_to_confirm hits."""
    tracker = SurferTracker(min_hits_to_confirm=3)

    # Frame 1: Detection appears
    d1 = create_detection(100.0, 200.0, 30.0, 70.0)
    active1 = tracker.update([d1], frame_idx=1)
    assert len(tracker.tracks) == 1
    assert len(active1) == 0  # not yet confirmed (hits=1 < 3)

    # Frame 2: Hit 2
    d2 = create_detection(105.0, 201.0, 30.0, 70.0)
    tracker.update([d2], frame_idx=2)
    assert len(tracker.tracks) == 1

    # Frame 3: Hit 3 -> Confirmed
    d3 = create_detection(110.0, 202.0, 30.0, 70.0)
    active3 = tracker.update([d3], frame_idx=3)
    assert len(active3) == 1
    assert active3[0].confirmed is True
    assert active3[0].track_id == 1


def test_tube_occlusion_and_recovery_1_8s():
    """
    Validates tube inertia memory under the breaking lip:
    A surfer riding rightward disappears under spray/lip for 1.8 seconds (55 frames at 30 fps),
    then re-emerges at the tube exit and is re-associated to the exact same track ID.
    """
    tracker = SurferTracker(
        max_lost_frames=25,
        max_tube_lost_frames=75,  # 2.5 seconds at 30 fps
        min_hits_to_confirm=3,
    )

    # 1. Establish confirmed riding track moving rightward at ~6 px/frame
    start_x, start_y = 200.0, 300.0
    vx = 6.0
    for f in range(1, 8):
        det = create_detection(start_x + f * vx, start_y, 30.0, 70.0)
        tracks = tracker.update([det], frame_idx=f, riding_track_ids={1})

    assert len(tracks) == 1
    initial_track_id = tracks[0].track_id
    assert tracks[0].is_riding is True
    assert pytest.approx(tracks[0].current_velocity[0], abs=1.0) == vx

    last_seen_frame = 7
    last_cx = start_x + 7 * vx

    # 2. Tube occlusion: Surfer disappears for 55 frames (1.83 seconds at 30 fps)
    for f in range(last_seen_frame + 1, last_seen_frame + 56):
        active = tracker.update([], frame_idx=f)
        assert len(active) == 0  # No direct detections returned
        # But track remains alive in tracker memory
        assert initial_track_id in tracker.tracks
        t = tracker.tracks[initial_track_id]
        assert t.is_riding is True
        assert t.is_in_tube is True
        assert t.lost_frames == (f - last_seen_frame)
        # Bounding box projected forward along wave direction
        assert t.center[0] > last_cx

    # 3. Surfer re-emerges at the tube exit (spit) at frame 63
    reappear_frame = last_seen_frame + 56
    predicted_exit_box = tracker.tracks[initial_track_id].bbox
    exit_cx = (predicted_exit_box[0] + predicted_exit_box[2]) / 2.0
    exit_cy = (predicted_exit_box[1] + predicted_exit_box[3]) / 2.0

    # Detection near projected spit position
    spit_det = create_detection(exit_cx + 5.0, exit_cy, 32.0, 68.0)
    recovered_tracks = tracker.update([spit_det], frame_idx=reappear_frame)

    assert len(recovered_tracks) == 1
    recovered = recovered_tracks[0]
    # Critical assertion: Identity persistence preserved across the 1.8s tube
    assert recovered.track_id == initial_track_id
    assert recovered.lost_frames == 0
    assert recovered.is_in_tube is False


def test_non_riding_track_dies_after_standard_lost_frames():
    """Validates that a non-riding track (e.g. paddler) expires after max_lost_frames (25)."""
    tracker = SurferTracker(max_lost_frames=25, max_tube_lost_frames=75, min_hits_to_confirm=2)

    tracker.update([create_detection(100.0, 200.0, 50.0, 20.0)], frame_idx=1)
    tracker.update([create_detection(102.0, 201.0, 50.0, 20.0)], frame_idx=2)
    assert 1 in tracker.tracks
    assert tracker.tracks[1].is_riding is False

    # Disappears for 26 frames
    for f in range(3, 30):
        tracker.update([], frame_idx=f)

    # Should be pruned
    assert 1 not in tracker.tracks


def test_vertical_velocity_pop_up_flottaison():
    """
    Validates that bottom-baseline tracking eliminates vertical velocity spikes
    when the surfer abruptly stands upright during pop-up.
    """
    tracker = SurferTracker(min_hits_to_confirm=1)

    # Frame 1: Prone paddling (surfer height = 24, feet on waterline at y = 324)
    d_prone = Detection(
        bbox=(200.0, 300.0, 260.0, 324.0),
        confidence=0.90,
        class_id=0,
        class_name="surfer",
    )
    tracker.update([d_prone], frame_idx=1)

    # Frame 2: Upright standing (surfer height = 72, head moves from 300 to 252, but feet stay at y = 324)
    d_standing = Detection(
        bbox=(205.0, 252.0, 235.0, 324.5),
        confidence=0.92,
        class_id=0,
        class_name="surfer",
    )
    tracks = tracker.update([d_standing], frame_idx=2)
    assert len(tracks) == 1

    t = tracks[0]
    # Check vertical velocity component
    vx, vy = t.velocities[-1]
    # The vertical velocity on the waterline baseline must NOT exceed 3.0 px/frame
    assert abs(vy) < 3.0

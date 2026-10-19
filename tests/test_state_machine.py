"""
Unit tests for SurferStateMachine and automated ride detection.
"""

from surf_camera.tracker import Track
from surf_camera.state_machine import SurferStateMachine, SurferState


def create_mock_track(
    track_id: int,
    center: tuple,
    width: float,
    height: float,
    velocity: tuple,
    frame_idx: int = 10,
) -> Track:
    cx, cy = center
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    return Track(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        confidence=0.85,
        start_frame=0,
        last_frame=frame_idx,
        history=[center],
        frame_indices=[frame_idx],
        aspect_ratios=[width / height],
        velocities=[velocity, velocity, velocity],
        hits=5,
        lost_frames=0,
        confirmed=True,
    )


def test_initial_state_paddling():
    sm = SurferStateMachine(fps=30.0)
    # Slow moving, horizontal prone posture (aspect ratio > 1.2)
    track = create_mock_track(track_id=1, center=(500, 300), width=60, height=25, velocity=(0.5, 0.2))
    states = sm.update([track], current_frame=1)
    assert states[1] == SurferState.PADDLING


def test_takeoff_and_riding_transition():
    sm = SurferStateMachine(fps=30.0, speed_takeoff_thresh=3.0, min_ride_duration_sec=0.5)

    # 1. Paddling
    t1 = create_mock_track(track_id=1, center=(500, 300), width=60, height=25, velocity=(0.5, 0.2))
    states1 = sm.update([t1], current_frame=1)
    assert states1[1] == SurferState.PADDLING

    # 2. Acceleration and pop-up (standing posture: width=30, height=70, speed=5.0)
    t2 = create_mock_track(track_id=1, center=(520, 310), width=30, height=70, velocity=(5.0, 1.0))
    states2 = sm.update([t2], current_frame=10)
    assert states2[1] in (SurferState.TAKE_OFF, SurferState.RIDING)

    # 3. Sustained riding across 20 frames
    for f in range(11, 40):
        t_ride = create_mock_track(track_id=1, center=(520 + (f-10)*5, 310), width=28, height=72, velocity=(5.0, 0.5))
        states = sm.update([t_ride], current_frame=f)
        assert states[1] == SurferState.RIDING

    # 4. Surfer exits the wave (track disappears past occlusion grace period)
    sm.update([], current_frame=90)

    # Validate that a RideEvent was generated
    assert len(sm.completed_rides) == 1
    ride = sm.completed_rides[0]
    assert ride.surfer_id == 1
    assert ride.direction == "RIGHT"
    assert ride.duration_sec > 0.5


def test_ride_consolidation_merges_fragmented_segments():
    """Validates that fragmented rides on the same wave are fused into one continuous clip."""
    sm = SurferStateMachine(fps=30.0)
    from surf_camera.state_machine import RideEvent

    # Simulate 2 fragmented cuts of the same wave separated by 10 frames
    ride_a = RideEvent(
        ride_id=1, surfer_id=1, start_frame=0, takeoff_frame=15, end_frame=75,
        duration_frames=75, duration_sec=2.5, max_speed_px=15.0, direction="LEFT", trajectory=[]
    )
    ride_b = RideEvent(
        ride_id=2, surfer_id=1, start_frame=85, takeoff_frame=85, end_frame=150,
        duration_frames=65, duration_sec=2.1, max_speed_px=18.0, direction="LEFT", trajectory=[]
    )
    sm.completed_rides = [ride_a, ride_b]

    consolidated = sm.get_consolidated_rides(max_gap_sec=2.0)
    assert len(consolidated) == 1
    assert consolidated[0].ride_id == 1
    assert consolidated[0].start_frame == 0
    assert consolidated[0].end_frame == 150
    assert consolidated[0].duration_frames == 150
    assert consolidated[0].max_speed_px == 18.0

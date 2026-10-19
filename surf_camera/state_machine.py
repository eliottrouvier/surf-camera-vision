"""
Surfer State Machine and Automated Wave Ride Detector.
Classifies kinematic phases (Paddling -> Take-off -> Riding -> Exit/Wipeout)
and segments discrete surfing rides for automated video clipping.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tracker import Track


class SurferState(Enum):
    PADDLING = "Paddling"
    TAKE_OFF = "Take-Off"
    RIDING = "Riding"
    WIPEOUT = "Wipeout"
    EXIT = "Exit"


@dataclass
class RideEvent:
    """Represents a validated surfing ride event extracted from footage."""
    ride_id: int
    surfer_id: int
    start_frame: int          # Frame where paddling/take-off initiated
    takeoff_frame: int        # Exact frame where surfer stood up
    end_frame: int            # Frame where ride terminated
    duration_frames: int
    duration_sec: float
    max_speed_px: float
    direction: str            # 'RIGHT' (frontside/backside right) or 'LEFT'
    trajectory: List[Tuple[float, float, int]]  # [(cx, cy, frame_idx), ...]


class SurferStateMachine:
    """
    Analyzes track kinematics (aspect ratio shifts and velocity vectors)
    to detect when a surfer catches a wave, rides it, and finishes.
    """
    def __init__(
        self,
        fps: float = 30.0,
        min_ride_duration_sec: float = 1.5,
        speed_takeoff_thresh: float = 3.5,       # Pixels per frame
        aspect_ratio_standing_thresh: float = 0.90, # Width / Height < 0.90 indicates upright posture
        pre_roll_sec: float = 2.0,              # Seconds of context before take-off
        post_roll_sec: float = 1.5,             # Seconds of context after ride end
    ):
        self.fps = fps
        self.min_ride_frames = int(min_ride_duration_sec * fps)
        self.speed_takeoff_thresh = speed_takeoff_thresh
        self.aspect_ratio_standing_thresh = aspect_ratio_standing_thresh
        self.pre_roll_frames = int(pre_roll_sec * fps)
        self.post_roll_frames = int(post_roll_sec * fps)

        # Internal state tracking per surfer ID
        self.current_states: Dict[int, SurferState] = {}
        self.active_rides: Dict[int, Dict] = {}
        self.completed_rides: List[RideEvent] = []
        self._next_ride_id: int = 1

    def update(self, tracks: List[Track], current_frame: int) -> Dict[int, SurferState]:
        """
        Processes active tracks at current_frame, updates states, and triggers rides.
        Returns a mapping from track_id to current SurferState.
        """
        active_track_ids = {t.track_id for t in tracks}

        # Handle tracks that disappeared while riding (wipeout / kickout behind crest)
        riding_ids = list(self.active_rides.keys())
        for tid in riding_ids:
            if tid not in active_track_ids:
                ride_data = self.active_rides[tid]
                # If track was missing, close the ride
                self._finalize_ride(tid, current_frame, end_state=SurferState.EXIT)

        for track in tracks:
            tid = track.track_id
            speed = track.current_speed
            ar = track.current_aspect_ratio
            prev_state = self.current_states.get(tid, SurferState.PADDLING)

            # 1. State transition logic
            new_state = prev_state

            if prev_state == SurferState.PADDLING:
                # Transition to TAKE_OFF: acceleration spike and/or posture becoming upright
                if speed >= self.speed_takeoff_thresh or ar < self.aspect_ratio_standing_thresh:
                    new_state = SurferState.TAKE_OFF
                    # Initialize active ride record
                    self.active_rides[tid] = {
                        "start_frame": max(0, current_frame - self.pre_roll_frames),
                        "takeoff_frame": current_frame,
                        "trajectory": [(track.center[0], track.center[1], current_frame)],
                        "speeds": [speed],
                        "dx": [],
                    }

            elif prev_state == SurferState.TAKE_OFF:
                # Confirm to RIDING if posture is standing and moving
                if ar < self.aspect_ratio_standing_thresh or speed >= self.speed_takeoff_thresh:
                    new_state = SurferState.RIDING
                elif speed < self.speed_takeoff_thresh * 0.5 and ar > 1.2:
                    # Aborted wave (missed take-off)
                    new_state = SurferState.PADDLING
                    if tid in self.active_rides:
                        del self.active_rides[tid]

            elif prev_state == SurferState.RIDING:
                # Check for ride termination: speed drops to near zero, or posture collapses
                if speed < self.speed_takeoff_thresh * 0.35 and ar > 1.3:
                    new_state = SurferState.WIPEOUT
                    self._finalize_ride(tid, current_frame, end_state=SurferState.WIPEOUT)
                else:
                    new_state = SurferState.RIDING

            elif prev_state in (SurferState.WIPEOUT, SurferState.EXIT):
                # Reset back to paddling after a ride finishes
                new_state = SurferState.PADDLING

            # Update trajectory & metrics if actively riding
            if tid in self.active_rides:
                self.active_rides[tid]["trajectory"].append((track.center[0], track.center[1], current_frame))
                self.active_rides[tid]["speeds"].append(speed)
                vx, _ = track.current_velocity
                self.active_rides[tid]["dx"].append(vx)

            self.current_states[tid] = new_state

        return {t.track_id: self.current_states.get(t.track_id, SurferState.PADDLING) for t in tracks}

    def _finalize_ride(self, track_id: int, current_frame: int, end_state: SurferState):
        """Concludes an active ride and saves it if it meets the minimum duration."""
        if track_id not in self.active_rides:
            return

        ride_data = self.active_rides.pop(track_id)
        start_frame = ride_data["start_frame"]
        takeoff_frame = ride_data["takeoff_frame"]
        end_frame = current_frame + self.post_roll_frames

        duration_frames = end_frame - start_frame
        duration_sec = duration_frames / self.fps

        if duration_frames >= self.min_ride_frames:
            # Determine direction: positive dx = RIGHT (moving rightwards), negative = LEFT
            avg_dx = float(np.mean(ride_data["dx"])) if len(ride_data["dx"]) > 0 else 0.0
            direction = "RIGHT" if avg_dx >= 0 else "LEFT"
            max_speed = float(np.max(ride_data["speeds"])) if len(ride_data["speeds"]) > 0 else 0.0

            event = RideEvent(
                ride_id=self._next_ride_id,
                surfer_id=track_id,
                start_frame=start_frame,
                takeoff_frame=takeoff_frame,
                end_frame=end_frame,
                duration_frames=duration_frames,
                duration_sec=duration_sec,
                max_speed_px=max_speed,
                direction=direction,
                trajectory=ride_data["trajectory"],
            )
            self._next_ride_id += 1
            self.completed_rides.append(event)

        self.current_states[track_id] = end_state

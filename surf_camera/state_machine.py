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
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    distance_meters: float = 0.0
    wave_height_meters: float = 0.0
    wave_height_feet: float = 0.0
    wave_category: str = "Waist High (2-3ft)"
    speed_series_kmh: List[float] = field(default_factory=list)
    wave_height_series_m: List[float] = field(default_factory=list)


class SurferStateMachine:
    """
    Analyzes track kinematics (aspect ratio shifts and velocity vectors)
    to detect when a surfer catches a wave, rides it, and finishes.
    """
    def __init__(
        self,
        fps: float = 30.0,
        min_ride_duration_sec: float = 1.2,
        speed_takeoff_thresh: float = 3.5,       # Pixels per frame
        aspect_ratio_standing_thresh: float = 0.90, # Width / Height < 0.90 indicates upright posture
        pre_roll_sec: float = 2.0,              # Seconds of context before take-off
        post_roll_sec: float = 1.5,             # Seconds of context after ride end
        occlusion_grace_sec: float = 1.5,       # Seconds to tolerate spray/occlusion before ending ride
        cooldown_sec: float = 3.0,              # Cooldown period before another ride can start
    ):
        self.fps = fps
        self.min_active_frames = int(min_ride_duration_sec * fps)
        self.min_ride_frames = self.min_active_frames
        self.speed_takeoff_thresh = speed_takeoff_thresh
        self.aspect_ratio_standing_thresh = aspect_ratio_standing_thresh
        self.pre_roll_frames = int(pre_roll_sec * fps)
        self.post_roll_frames = int(post_roll_sec * fps)
        self.occlusion_grace_frames = int(occlusion_grace_sec * fps)
        self.cooldown_frames = int(cooldown_sec * fps)

        # Internal state tracking per surfer ID
        self.current_states: Dict[int, SurferState] = {}
        self.active_rides: Dict[int, Dict] = {}
        self.completed_rides: List[RideEvent] = []
        self.cooldowns: Dict[int, int] = {}
        self._next_ride_id: int = 1

    def update(self, tracks: List[Track], current_frame: int) -> Dict[int, SurferState]:
        """
        Processes active tracks at current_frame, updates states, and triggers rides.
        Returns a mapping from track_id to current SurferState.
        """
        active_track_ids = {t.track_id for t in tracks}

        # 1. Handle tracks that disappeared while riding (wipeout / kickout behind crest)
        # Give grace period before finalizing to absorb spray and duck-dives
        riding_ids = list(self.active_rides.keys())
        for tid in riding_ids:
            if tid not in active_track_ids:
                ride_data = self.active_rides[tid]
                last_seen = ride_data.get("last_seen_frame", ride_data["takeoff_frame"])
                if (current_frame - last_seen) > self.occlusion_grace_frames:
                    self._finalize_ride(tid, last_seen, end_state=SurferState.EXIT)

        for track in tracks:
            tid = track.track_id
            speed = track.current_speed
            ar = track.current_aspect_ratio
            prev_state = self.current_states.get(tid, SurferState.PADDLING)

            # In cooldown? Keep in paddling
            if current_frame < self.cooldowns.get(tid, 0):
                self.current_states[tid] = SurferState.PADDLING
                continue

            # State transition logic
            new_state = prev_state

            if prev_state == SurferState.PADDLING:
                # Transition to TAKE_OFF: acceleration spike and/or posture becoming upright
                if speed >= self.speed_takeoff_thresh or ar < self.aspect_ratio_standing_thresh:
                    new_state = SurferState.TAKE_OFF
                    self.active_rides[tid] = {
                        "start_frame": max(0, current_frame - self.pre_roll_frames),
                        "takeoff_frame": current_frame,
                        "last_seen_frame": current_frame,
                        "trajectory": [(track.center[0], track.center[1], current_frame)],
                        "speeds": [speed],
                        "dx": [],
                    }

            elif prev_state == SurferState.TAKE_OFF:
                # Confirm to RIDING if posture is standing and moving
                if ar < self.aspect_ratio_standing_thresh or speed >= self.speed_takeoff_thresh:
                    new_state = SurferState.RIDING
                    if tid in self.active_rides:
                        self.active_rides[tid]["last_seen_frame"] = current_frame
                elif speed < self.speed_takeoff_thresh * 0.4 and ar > 1.25:
                    # Aborted wave (missed take-off)
                    new_state = SurferState.PADDLING
                    if tid in self.active_rides:
                        del self.active_rides[tid]

            elif prev_state == SurferState.RIDING:
                # Check for ride termination: speed drops to near zero, or posture collapses
                if speed < self.speed_takeoff_thresh * 0.30 and ar > 1.35:
                    new_state = SurferState.WIPEOUT
                    last_seen = self.active_rides[tid].get("last_seen_frame", current_frame) if tid in self.active_rides else current_frame
                    self._finalize_ride(tid, last_seen, end_state=SurferState.WIPEOUT)
                else:
                    new_state = SurferState.RIDING
                    if tid in self.active_rides:
                        self.active_rides[tid]["last_seen_frame"] = current_frame

            elif prev_state in (SurferState.WIPEOUT, SurferState.EXIT):
                new_state = SurferState.PADDLING

            # Update trajectory & metrics if actively riding
            if tid in self.active_rides:
                self.active_rides[tid]["trajectory"].append((track.center[0], track.center[1], current_frame))
                self.active_rides[tid]["speeds"].append(speed)
                vx, _ = track.current_velocity
                self.active_rides[tid]["dx"].append(vx)
                self.active_rides[tid]["last_seen_frame"] = current_frame

            self.current_states[tid] = new_state

        return {t.track_id: self.current_states.get(t.track_id, SurferState.PADDLING) for t in tracks}

    def _finalize_ride(self, track_id: int, current_frame: int, end_state: SurferState):
        """Concludes an active ride and saves it only if actual standing duration is met."""
        if track_id not in self.active_rides:
            return

        ride_data = self.active_rides.pop(track_id)
        takeoff_frame = ride_data["takeoff_frame"]
        start_frame = ride_data["start_frame"]
        end_frame = current_frame + self.post_roll_frames

        # Crucial: Check active riding duration (excluding pre/post roll)
        active_duration_frames = max(0, current_frame - takeoff_frame)
        if active_duration_frames >= self.min_active_frames:
            avg_dx = float(np.mean(ride_data["dx"])) if len(ride_data["dx"]) > 0 else 0.0
            direction = "RIGHT" if avg_dx >= 0 else "LEFT"
            max_speed = float(np.max(ride_data["speeds"])) if len(ride_data["speeds"]) > 0 else 0.0
            duration_frames = end_frame - start_frame
            duration_sec = duration_frames / self.fps

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

            # Apply cooldown to prevent immediate false re-triggers on the same surfer
            self.cooldowns[track_id] = current_frame + self.cooldown_frames

        self.current_states[track_id] = end_state

    def get_consolidated_rides(self, max_gap_sec: float = 2.0) -> List[RideEvent]:
        """
        Fuses fragmented or overlapping ride events into unified continuous rides.
        Eliminates duplicate cuts caused by spray, duck-dives, or temporary track ID switches.
        """
        if not self.completed_rides:
            return []

        sorted_rides = sorted(self.completed_rides, key=lambda r: r.start_frame)
        merged: List[RideEvent] = []
        max_gap_frames = int(max_gap_sec * self.fps)

        current = sorted_rides[0]

        for next_ride in sorted_rides[1:]:
            # If rides overlap or are within max_gap_frames, merge them!
            if next_ride.start_frame <= (current.end_frame + max_gap_frames):
                merged_start = min(current.start_frame, next_ride.start_frame)
                merged_takeoff = min(current.takeoff_frame, next_ride.takeoff_frame)
                merged_end = max(current.end_frame, next_ride.end_frame)
                merged_dur_frames = merged_end - merged_start
                merged_dur_sec = merged_dur_frames / self.fps
                merged_max_speed = max(current.max_speed_px, next_ride.max_speed_px)
                # Keep direction of the longer segment
                direction = current.direction if current.duration_frames >= next_ride.duration_frames else next_ride.direction

                merged_max_kmh = max(current.max_speed_kmh, next_ride.max_speed_kmh)
                merged_dist = current.distance_meters + next_ride.distance_meters
                merged_wave_h = max(current.wave_height_meters, next_ride.wave_height_meters)
                merged_wave_cat = current.wave_category if current.wave_height_meters >= next_ride.wave_height_meters else next_ride.wave_category
                combined_speeds = current.speed_series_kmh + next_ride.speed_series_kmh
                combined_wave_hs = current.wave_height_series_m + next_ride.wave_height_series_m
                merged_avg_kmh = float(np.mean(combined_speeds)) if combined_speeds else 0.0

                current = RideEvent(
                    ride_id=current.ride_id,
                    surfer_id=current.surfer_id,
                    start_frame=merged_start,
                    takeoff_frame=merged_takeoff,
                    end_frame=merged_end,
                    duration_frames=merged_dur_frames,
                    duration_sec=merged_dur_sec,
                    max_speed_px=merged_max_speed,
                    direction=direction,
                    trajectory=current.trajectory + next_ride.trajectory,
                    max_speed_kmh=round(merged_max_kmh, 1),
                    avg_speed_kmh=round(merged_avg_kmh, 1),
                    distance_meters=round(merged_dist, 1),
                    wave_height_meters=round(merged_wave_h, 2),
                    wave_height_feet=round(merged_wave_h * 3.28084, 1),
                    wave_category=merged_wave_cat,
                    speed_series_kmh=combined_speeds,
                    wave_height_series_m=combined_wave_hs,
                )
            else:
                merged.append(current)
                current = next_ride

        merged.append(current)

        # Renumber cleanly
        for idx, r in enumerate(merged, start=1):
            r.ride_id = idx

        return merged

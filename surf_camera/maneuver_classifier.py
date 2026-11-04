"""
Surfing Maneuver Classifier & Kinetic Pose Evaluator.
Detects, timestamps, and scores high-performance surfing maneuvers:
- 🔥 Bottom Turn (deep compression & carve angle in wave trough)
- ⚡ Roller / Snap (critical shoulder rotation & redirection at the crest)
- 🌊 Tube / Barrel (hyper-flexion & sustained crest occlusion)
- 🏄 Carve / Cutback (directional rail engagement)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict
import numpy as np

from .pose_estimator import SurferPose
from .wave_analyzer import WaveMetrics


class ManeuverType(Enum):
    NONE = "Glisse Neutre"
    BOTTOM_TURN = "Bottom Turn"
    ROLLER = "Roller / Snap"
    TUBE = "Tube / Barrel"
    CARVE = "Cutback / Carve"


@dataclass
class ManeuverDetection:
    """Represents a detected surfing maneuver at a specific frame."""
    maneuver_type: ManeuverType
    name: str
    confidence: float
    score_out_of_10: float
    g_force: float
    knee_flexion_deg: float
    torso_lean_deg: float
    description: str
    is_active: bool = True


class ManeuverClassifier:
    """
    Evaluates kinematic state and body articulation over a temporal sliding window
    to identify discrete surfing figures with physical plausibility.
    """
    def __init__(self, fps: float = 30.0, window_size: int = 15):
        self.fps = fps
        self.window_size = window_size

        # Rolling history buffers
        self.pose_history: List[SurferPose] = []
        self.speed_history: List[float] = []
        self.pos_y_history: List[float] = []
        self.vel_x_history: List[float] = []
        self.tube_frames_counter: int = 0

    def reset(self):
        """Resets temporal buffers between separate rides."""
        self.pose_history.clear()
        self.speed_history.clear()
        self.pos_y_history.clear()
        self.vel_x_history.clear()
        self.tube_frames_counter = 0

    def update(
        self,
        pose: Optional[SurferPose],
        wave: Optional[WaveMetrics],
        surfer_center: Tuple[float, float],
        velocity: Tuple[float, float],
        speed_kmh: float,
    ) -> ManeuverDetection:
        """
        Analyzes the current frame kinematics and body pose to classify maneuvers.
        """
        cx, cy = surfer_center
        vx, vy = velocity

        # Update buffers
        self.speed_history.append(speed_kmh)
        self.pos_y_history.append(cy)
        self.vel_x_history.append(vx)
        if pose is not None:
            self.pose_history.append(pose)

        # Keep rolling window
        if len(self.speed_history) > self.window_size:
            self.speed_history.pop(0)
            self.pos_y_history.pop(0)
            self.vel_x_history.pop(0)
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)

        # Default fallback
        if pose is None:
            return ManeuverDetection(
                maneuver_type=ManeuverType.NONE,
                name="Glisse Neutre",
                confidence=0.5,
                score_out_of_10=5.0,
                g_force=1.0,
                knee_flexion_deg=140.0,
                torso_lean_deg=0.0,
                description="Suivi standard sans articulations visibles",
                is_active=False,
            )

        # Extract biomechanical signals
        knee_deg = pose.knee_angle_deg
        lean_deg = pose.torso_lean_deg
        compression = pose.compression_ratio
        shoulder_tilt = pose.shoulder_tilt_deg

        # Determine relative height on wave face (0.0 = crest, 1.0 = trough)
        if wave is not None and (wave.trough_y - wave.crest_y) > 15:
            wave_h_span = wave.trough_y - wave.crest_y
            norm_wave_pos = np.clip((cy - wave.crest_y) / wave_h_span, 0.0, 1.0)
        else:
            norm_wave_pos = 0.50

        # Estimate centrifugal G-force from turning rate
        if len(self.vel_x_history) >= 4:
            dx_diff = abs(self.vel_x_history[-1] - self.vel_x_history[0])
            g_force = round(float(1.0 + (dx_diff * speed_kmh) / 120.0), 1)
        else:
            g_force = 1.0

        # ---------------------------------------------------------------------
        # 1. TUBE / BARREL DETECTION
        # Hyper-compression + position under wave crest + upper-body occlusion
        # ---------------------------------------------------------------------
        if compression < 0.48 and knee_deg < 85.0 and norm_wave_pos < 0.65:
            self.tube_frames_counter += 1
        else:
            self.tube_frames_counter = max(0, self.tube_frames_counter - 1)

        if self.tube_frames_counter >= 6:  # ~0.20s continuous in barrel
            tube_dur_sec = self.tube_frames_counter / self.fps
            score = round(min(10.0, 7.5 + tube_dur_sec * 1.2), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.TUBE,
                name="🌊 Tube / Barrel",
                confidence=0.88,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Tube profond ({tube_dur_sec:.1f}s) • Flexion {knee_deg:.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # 2. ROLLER / SNAP DETECTION
        # High on wave (crest) + rapid shoulder rotation + speed redirect
        # ---------------------------------------------------------------------
        if norm_wave_pos <= 0.35 and len(self.pose_history) >= 4:
            # Check shoulder angular velocity
            past_tilt = self.pose_history[0].shoulder_tilt_deg
            d_tilt = abs(shoulder_tilt - past_tilt)

            if d_tilt >= 15.0 or (abs(lean_deg) > 25.0 and vy > 0):
                score = round(min(10.0, 6.5 + (d_tilt / 10.0) + (g_force * 0.4)), 1)
                return ManeuverDetection(
                    maneuver_type=ManeuverType.ROLLER,
                    name="⚡ Roller / Snap",
                    confidence=0.85,
                    score_out_of_10=score,
                    g_force=g_force,
                    knee_flexion_deg=knee_deg,
                    torso_lean_deg=lean_deg,
                    description=f"Attaque de lèvre • Rotation {d_tilt:.0f}° • G-Force {g_force}G",
                    is_active=True,
                )

        # ---------------------------------------------------------------------
        # 3. BOTTOM TURN DETECTION
        # Deep in wave trough + strong inward torso lean + compression
        # ---------------------------------------------------------------------
        if norm_wave_pos >= 0.55 and abs(lean_deg) >= 24.0 and knee_deg < 125.0:
            score = round(min(10.0, 6.0 + (abs(lean_deg) / 10.0) + (g_force * 0.5)), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.BOTTOM_TURN,
                name="🔥 Bottom Turn",
                confidence=0.90,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Prise de carre en bas de vague • Inclinaison {abs(lean_deg):.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # 4. CARVE / CUTBACK DETECTION
        # Rail engagement midway up the wave with moderate lean
        # ---------------------------------------------------------------------
        if abs(lean_deg) >= 18.0 and knee_deg < 135.0:
            score = round(min(10.0, 5.5 + (abs(lean_deg) / 12.0)), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.CARVE,
                name="🏄 Carve",
                confidence=0.80,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Courbe fluide • Inclinaison {abs(lean_deg):.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # Default Neutral Cruising
        # ---------------------------------------------------------------------
        return ManeuverDetection(
            maneuver_type=ManeuverType.NONE,
            name="Glisse Neutre",
            confidence=0.75,
            score_out_of_10=5.0,
            g_force=1.0,
            knee_flexion_deg=knee_deg,
            torso_lean_deg=lean_deg,
            description="Ligne droite d'accélération",
            is_active=False,
        )

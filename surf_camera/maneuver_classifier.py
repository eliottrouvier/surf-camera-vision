"""
Surfing Maneuver Classifier & WSL Judging Score Engine.
Evaluates biomechanical posture and kinematics to detect genuine surfing maneuvers:
- 🔥 Bottom Turn (deep knee compression & rail lean in wave trough)
- ⚡ Roller / Snap (critical rotation and redirection at the crest with flexed knees)
- 🌊 Tube / Barrel (hyper-compression & sustained position under crest)
- 🏄 Carve / Cutback (sustained rail engagement on the open face)

Enforces strict physical constraints (e.g. impossible to snap with stiff legs)
and applies authentic WSL (World Surf League Article 133) scoring criteria:
- 0.0 - 1.9 : Poor (simple straight-line glide, stiff posture, no maneuvers)
- 2.0 - 3.9 : Fair (controlled glide, basic functional carving)
- 4.0 - 5.9 : Good (committed maneuvers with speed in the critical section)
- 6.0 - 7.9 : Very Good (power, speed, flow, combination of major maneuvers)
- 8.0 - 10.0 : Excellent (extreme commitment, deep tube, radical critical figures)
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


def angle_diff(a1: float, a2: float) -> float:
    """Computes minimum circular angular difference in degrees [0, 180]."""
    return float(abs((a1 - a2 + 180.0) % 360.0 - 180.0))


def is_shoulder_swap(tilt1: float, tilt2: float) -> bool:
    """
    Detects if 2D pose estimation swapped left and right shoulders (~180° inversion).
    Occurs when viewing a surfer from an oblique or side perspective.
    """
    diff = angle_diff(tilt1, tilt2)
    return 155.0 <= diff <= 205.0


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
        self.vel_y_history: List[float] = []

        # Sustained temporal frame counters (prevent single-frame noise spikes)
        self.tube_frames_counter: int = 0
        self.roller_frames_counter: int = 0
        self.bottom_turn_frames_counter: int = 0
        self.carve_frames_counter: int = 0

    def reset(self):
        """Resets temporal buffers between separate rides."""
        self.pose_history.clear()
        self.speed_history.clear()
        self.pos_y_history.clear()
        self.vel_x_history.clear()
        self.vel_y_history.clear()
        self.tube_frames_counter = 0
        self.roller_frames_counter = 0
        self.bottom_turn_frames_counter = 0
        self.carve_frames_counter = 0

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
        self.vel_y_history.append(vy)
        if pose is not None:
            self.pose_history.append(pose)

        # Keep rolling window
        if len(self.speed_history) > self.window_size:
            self.speed_history.pop(0)
            self.pos_y_history.pop(0)
            self.vel_x_history.pop(0)
            self.vel_y_history.pop(0)
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)

        # Default fallback (Neutral straight glide)
        if pose is None:
            return ManeuverDetection(
                maneuver_type=ManeuverType.NONE,
                name="Glisse Neutre",
                confidence=0.5,
                score_out_of_10=1.0,
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

        if self.tube_frames_counter >= 8:  # ~0.26s continuous in barrel
            tube_dur_sec = self.tube_frames_counter / self.fps
            score = round(min(9.5, 7.0 + tube_dur_sec * 1.0), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.TUBE,
                name="🌊 Tube / Barrel",
                confidence=0.90,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Tube ({tube_dur_sec:.1f}s) • Flexion {knee_deg:.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # 2. ROLLER / SNAP DETECTION
        # Prerequisites:
        # - High on wave (norm_wave_pos <= 0.40)
        # - BIOMECHANICAL REQUIREMENT: Athletic knee flexion (knee_deg < 125.0)
        #   (A surfer standing stiff with knee_deg > 130 cannot physically snap!)
        # - Genuine shoulder rotation (excluding 180° left/right shoulder swap)
        # ---------------------------------------------------------------------
        is_roller_candidate = False
        d_tilt = 0.0

        if norm_wave_pos <= 0.40 and knee_deg < 125.0 and len(self.pose_history) >= 3:
            past_tilt = self.pose_history[0].shoulder_tilt_deg
            # Reject left/right shoulder keypoint swap (155° - 205°)
            if not is_shoulder_swap(shoulder_tilt, past_tilt):
                d_tilt = angle_diff(shoulder_tilt, past_tilt)
                if (d_tilt >= 14.0) or (abs(lean_deg) > 22.0 and vy > 0):
                    is_roller_candidate = True

        if is_roller_candidate:
            self.roller_frames_counter += 1
        else:
            self.roller_frames_counter = max(0, self.roller_frames_counter - 1)

        # Trigger roller if rotation is confirmed across the temporal pose window
        if self.roller_frames_counter >= 1 and is_roller_candidate:
            score = round(min(8.5, 4.5 + (d_tilt / 15.0) + (g_force * 0.4)), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.ROLLER,
                name="⚡ Roller / Snap",
                confidence=0.85,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Attaque de lèvre • Rotation {d_tilt:.0f}° • Flexion {knee_deg:.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # 3. BOTTOM TURN DETECTION
        # Deep in wave trough + strong torso lean + deep knee compression
        # ---------------------------------------------------------------------
        is_bt_candidate = (
            norm_wave_pos >= 0.50 and
            abs(lean_deg) >= 20.0 and
            knee_deg < 125.0
        )

        if is_bt_candidate:
            self.bottom_turn_frames_counter += 1
        else:
            self.bottom_turn_frames_counter = max(0, self.bottom_turn_frames_counter - 1)

        if self.bottom_turn_frames_counter >= 1: # Immediate trigger if sustained or initial criteria met
            score = round(min(7.5, 4.0 + (abs(lean_deg) / 12.0) + (g_force * 0.4)), 1)
            return ManeuverDetection(
                maneuver_type=ManeuverType.BOTTOM_TURN,
                name="🔥 Bottom Turn",
                confidence=0.88,
                score_out_of_10=score,
                g_force=g_force,
                knee_flexion_deg=knee_deg,
                torso_lean_deg=lean_deg,
                description=f"Prise de carre en creux de vague • Inclinaison {abs(lean_deg):.0f}°",
                is_active=True,
            )

        # ---------------------------------------------------------------------
        # 4. CARVE / CUTBACK DETECTION
        # Rail engagement midway up the wave with moderate lean & flexed knees
        # ---------------------------------------------------------------------
        is_carve_candidate = (
            abs(lean_deg) >= 16.0 and
            knee_deg < 135.0
        )

        if is_carve_candidate:
            self.carve_frames_counter += 1
        else:
            self.carve_frames_counter = max(0, self.carve_frames_counter - 1)

        if self.carve_frames_counter >= 3:
            score = round(min(6.5, 3.5 + (abs(lean_deg) / 14.0)), 1)
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
        # Default Neutral Cruising (Glisse droite)
        # Authentically rated as WSL Poor (0.5 to 1.5)
        # ---------------------------------------------------------------------
        base_score = 1.0
        if knee_deg > 155.0:
            # Stiff posture penalty
            base_score = 0.8
        elif knee_deg < 140.0:
            base_score = 1.2

        return ManeuverDetection(
            maneuver_type=ManeuverType.NONE,
            name="Glisse Neutre",
            confidence=0.75,
            score_out_of_10=base_score,
            g_force=1.0,
            knee_flexion_deg=knee_deg,
            torso_lean_deg=lean_deg,
            description="Ligne droite d'accélération sans virage engagé",
            is_active=False,
        )


def compute_wsl_score(
    maneuvers_detected: List[ManeuverDetection],
    duration_sec: float,
    peak_speed_kmh: float,
    wave_height_m: float,
    mean_knee_angle_deg: float,
) -> Tuple[float, str, str]:
    """
    Computes an authentic World Surf League (WSL Article 133) Heat score (0.0 to 10.0).

    Official Scale:
    - 0.0 - 1.9 : Poor (Neutral glide, uncommitted turns, stiff legs, wipeout)
    - 2.0 - 3.9 : Fair (Controlled ride, basic functional turns, gentle carve)
    - 4.0 - 5.9 : Good (Committed turns with speed and flow in the critical pocket)
    - 6.0 - 7.9 : Very Good (High-speed linked combination of major maneuvers)
    - 8.0 - 10.0 : Excellent (Radical critical maneuvers in the heaviest section, deep barrel)
    """
    valid_maneuvers = [
        m for m in maneuvers_detected
        if m.is_active and m.maneuver_type != ManeuverType.NONE
    ]

    # CASE A: No genuine maneuvers performed (Simple straight-line glide)
    if not valid_maneuvers:
        # Base Poor score strictly between 0.5 and 1.8
        base = 0.6 + min(0.6, (duration_sec / 8.0) * 0.6)
        # Stiff legs penalty (> 150° extension)
        if mean_knee_angle_deg > 150.0:
            base *= 0.75
        # Slight bonus if surfing with good speed (> 15 km/h)
        if peak_speed_kmh > 15.0:
            base += 0.3
        final_score = round(float(np.clip(base, 0.4, 1.8)), 1)
        category = "Poor (0.0 - 1.9)"
        desc = "Glisse droite neutre sans manœuvre engagée (posture droite)"
        return final_score, category, desc

    # CASE B: Maneuvers detected
    # Count unique types of maneuvers performed
    types_set = set(m.maneuver_type for m in valid_maneuvers)
    max_m_score = max(m.score_out_of_10 for m in valid_maneuvers)

    # 1 maneuver: typically Fair (2.0 - 3.8)
    if len(types_set) == 1:
        base = max_m_score * 0.75
        if duration_sec > 4.0:
            base += 0.4
        final_score = round(float(np.clip(base, 2.0, 4.5)), 1)
        category = "Fair (2.0 - 3.9)" if final_score < 4.0 else "Good (4.0 - 5.9)"
        desc = f"Manœuvre isolée complétée ({valid_maneuvers[0].name})"
        return final_score, category, desc

    # 2+ combined maneuvers: Good (4.0 - 6.5) or Very Good (6.5 - 7.9)
    combo_bonus = (len(types_set) - 1) * 1.2
    base = max_m_score * 0.75 + combo_bonus
    if peak_speed_kmh > 20.0:
        base += 0.5
    final_score = round(float(np.clip(base, 4.0, 9.8)), 1)

    if final_score < 6.0:
        category = "Good (4.0 - 5.9)"
        desc = "Enchaînement de virages avec contrôle et vitesse"
    elif final_score < 8.0:
        category = "Very Good (6.0 - 7.9)"
        desc = "Combinaison engagée de manœuvres majeures avec flow"
    else:
        category = "Excellent (8.0 - 10.0)"
        desc = "Séquence radicale dans la section critique de la vague"

    return final_score, category, desc

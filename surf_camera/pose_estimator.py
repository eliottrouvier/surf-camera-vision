"""
Targeted Surfer Pose Estimation Engine (Level-2 Micro-Biomechanical Inactivity).
Runs YOLOv8-Pose on high-resolution dynamic crops of the active riding surfer,
extracting 17 COCO keypoints and calculating key biomechanical angles
(knee flexion, torso inclination, compression ratio, shoulder rotation).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from ultralytics import YOLO


# Keypoint Constants
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Skeleton connection pairs for rendering
SKELETON_EDGES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


@dataclass
class SurferPose:
    """Represents the 17 articulated keypoints and biomechanical posture of a surfer."""
    keypoints: np.ndarray         # (17, 3) [x, y, conf] in full frame coordinates
    crop_keypoints: np.ndarray    # (17, 3) [x, y, conf] in local crop coordinates
    crop_bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2) of the crop window
    confidence: float
    knee_angle_deg: float         # Average knee flexion angle (180 = straight, 90 = deep carve)
    torso_lean_deg: float         # Torso lean relative to vertical (signed, degrees)
    compression_ratio: float      # Pelvis-to-ankle height relative to standing height
    shoulder_tilt_deg: float      # Shoulder rotation relative to horizontal


def compute_angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Computes angle ABC at point B in degrees.
    Returns angle in [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-4 or norm_bc < 1e-4:
        return 180.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine) * 180.0 / np.pi)


class PoseEstimator:
    """
    Infers 2D human pose on the surfer with hardware acceleration.
    """
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        crop_margin_ratio: float = 0.20,
    ):
        self.conf_threshold = conf_threshold
        self.crop_margin_ratio = crop_margin_ratio

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = YOLO(model_name)

    def estimate(
        self,
        frame: np.ndarray,
        surfer_bbox: Tuple[float, float, float, float],
    ) -> Optional[SurferPose]:
        """
        Extracts high-resolution crop around the surfer, detects 17 keypoints,
        re-projects coordinates to the full frame, and computes posture metrics.
        """
        H, W = frame.shape[:2]
        sx1, sy1, sx2, sy2 = surfer_bbox
        sw = sx2 - sx1
        sh = sy2 - sy1

        # Expand crop with margin to capture arms and surfboard rail contact
        pad_x = int(sw * self.crop_margin_ratio)
        pad_y = int(sh * self.crop_margin_ratio)

        cx1 = max(0, int(sx1 - pad_x))
        cy1 = max(0, int(sy1 - pad_y))
        cx2 = min(W, int(sx2 + pad_x))
        cy2 = min(H, int(sy2 + pad_y))

        if (cx2 - cx1) < 20 or (cy2 - cy1) < 20:
            return None

        crop = frame[cy1:cy2, cx1:cx2]

        # Predict pose on crop
        results = self.model.predict(
            source=crop,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None

        res = results[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return None

        kpts_data = res.keypoints.data[0].cpu().numpy()  # (17, 3) [x, y, conf]
        if kpts_data.shape[0] < 17:
            return None

        # Re-project to full frame coordinates
        full_kpts = kpts_data.copy()
        full_kpts[:, 0] += cx1
        full_kpts[:, 1] += cy1

        mean_conf = float(np.mean(kpts_data[:, 2]))

        # Calculate Biomechanical Angles:
        # 1. Left and Right Knee Angles
        l_hip, l_knee, l_ankle = kpts_data[LEFT_HIP, :2], kpts_data[LEFT_KNEE, :2], kpts_data[LEFT_ANKLE, :2]
        r_hip, r_knee, r_ankle = kpts_data[RIGHT_HIP, :2], kpts_data[RIGHT_KNEE, :2], kpts_data[RIGHT_ANKLE, :2]

        angle_l = compute_angle_3pts(l_hip, l_knee, l_ankle)
        angle_r = compute_angle_3pts(r_hip, r_knee, r_ankle)
        avg_knee_angle = (angle_l + angle_r) / 2.0

        # 2. Torso Inclination (Mid-hips to Mid-shoulders vs Vertical)
        mid_hip = (kpts_data[LEFT_HIP, :2] + kpts_data[RIGHT_HIP, :2]) / 2.0
        mid_shoulder = (kpts_data[LEFT_SHOULDER, :2] + kpts_data[RIGHT_SHOULDER, :2]) / 2.0
        torso_vec = mid_shoulder - mid_hip  # Points upward towards shoulders

        # Vector pointing straight up is (0, -1) in image coordinates
        torso_angle = float(np.arctan2(torso_vec[0], -torso_vec[1]) * 180.0 / np.pi)

        # 3. Shoulder Tilt relative to horizontal
        shoulder_vec = kpts_data[RIGHT_SHOULDER, :2] - kpts_data[LEFT_SHOULDER, :2]
        shoulder_tilt = float(np.arctan2(shoulder_vec[1], shoulder_vec[0]) * 180.0 / np.pi)

        # 4. Compression Ratio (Distance hip to ankle normalized by bbox height)
        mid_ankle = (kpts_data[LEFT_ANKLE, :2] + kpts_data[RIGHT_ANKLE, :2]) / 2.0
        leg_span = np.linalg.norm(mid_hip - mid_ankle)
        compression = float(np.clip(leg_span / max(1.0, sh), 0.20, 1.0))

        return SurferPose(
            keypoints=full_kpts,
            crop_keypoints=kpts_data,
            crop_bbox=(cx1, cy1, cx2, cy2),
            confidence=mean_conf,
            knee_angle_deg=round(avg_knee_angle, 1),
            torso_lean_deg=round(torso_angle, 1),
            compression_ratio=round(compression, 2),
            shoulder_tilt_deg=round(shoulder_tilt, 1),
        )

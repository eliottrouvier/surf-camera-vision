"""
Joint Surfer-Board Metric Scale Estimator.
Estimates real-world scale (meters per pixel) directly from the surfer's
nominal physical height and stabilized bounding box height in riding posture.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import numpy as np


@dataclass
class ScaleEstimate:
    """Represents an estimated metric scale at a given frame."""
    meters_per_pixel: float
    confidence: float
    surfer_height_px: float
    board_length_px: float
    is_upright: bool
    latent_z: Optional[Any] = None


class MetricScaleEstimator:
    """
    Direct and deterministic metric scale estimator.
    Calculates meters per pixel based on nominal surfer height and stabilized
    bounding box height in riding posture (~88% of standing height).
    """
    def __init__(
        self,
        nominal_surfer_height_m: float = 1.75,
        nominal_board_length_m: float = 1.73,
        ema_alpha: float = 0.15,
        device: str = "cpu",
    ):
        self.nominal_surfer_height_m = nominal_surfer_height_m
        self.nominal_board_length_m = nominal_board_length_m
        self.ema_alpha = ema_alpha
        self.device = device

        self.current_m_per_px: Optional[float] = None
        self.history_scales: List[float] = []

    def update(
        self,
        surfer_bbox: Tuple[float, float, float, float],
        board_bbox: Optional[Tuple[float, float, float, float]] = None,
        board_mask: Optional[np.ndarray] = None,
        is_standing_hint: bool = False,
        pose: Optional[Any] = None,
    ) -> ScaleEstimate:
        """
        Updates and returns the calibrated metric scale for the current frame.
        Directly calculates scale based on upright bounding box height.
        """
        sx1, sy1, sx2, sy2 = surfer_bbox
        s_w = max(1.0, sx2 - sx1)
        s_h = max(1.0, sy2 - sy1)
        s_aspect = s_w / s_h

        # Compute board dimension if provided
        if board_mask is not None and len(board_mask) >= 5:
            rect = cv2_min_area_rect(board_mask)
            b_dim1, b_dim2 = rect[1]
            b_len = max(b_dim1, b_dim2)
        elif board_bbox is not None:
            bx1, by1, bx2, by2 = board_bbox
            b_w = max(1.0, bx2 - bx1)
            b_h = max(1.0, by2 - by1)
            b_len = float(np.sqrt(b_w**2 + b_h**2))
        else:
            b_len = s_h * 1.05

        is_upright = (s_aspect < 0.85) or is_standing_hint

        # Guard: If surfer is still paddling / prone in water (tiny box or not standing),
        # do not pollute the scale history with an uncalibrated prone box
        if not is_standing_hint and s_h < 35.0:
            fallback = self.current_m_per_px or 0.022
            return ScaleEstimate(
                meters_per_pixel=float(fallback),
                confidence=0.30,
                surfer_height_px=float(s_h),
                board_length_px=float(b_len),
                is_upright=False,
            )

        # In riding posture, effective stance height is ~88% of standing height
        if is_upright and s_h > 25.0:
            effective_height_m = self.nominal_surfer_height_m * 0.88
            measured_scale = effective_height_m / s_h
            confidence = 0.85
        elif b_len > 25.0:
            measured_scale = self.nominal_board_length_m / b_len
            confidence = 0.70
        else:
            measured_scale = self.current_m_per_px or 0.022
            confidence = 0.30

        # Physical clamping between 0.010 and 0.045 m/pixel (1.0 cm to 4.5 cm per pixel)
        measured_scale = float(np.clip(measured_scale, 0.010, 0.045))

        # Temporal EMA smoothing
        if self.current_m_per_px is None:
            self.current_m_per_px = measured_scale
        else:
            alpha = self.ema_alpha if is_upright else (self.ema_alpha * 0.4)
            self.current_m_per_px = alpha * measured_scale + (1.0 - alpha) * self.current_m_per_px

        self.history_scales.append(self.current_m_per_px)

        return ScaleEstimate(
            meters_per_pixel=float(self.current_m_per_px),
            confidence=float(confidence),
            surfer_height_px=float(s_h),
            board_length_px=float(b_len),
            is_upright=is_upright,
        )


def cv2_min_area_rect(poly: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Helper to compute rotated bounding box on 2D polygon."""
    import cv2
    pts = poly.astype(np.float32)
    return cv2.minAreaRect(pts)

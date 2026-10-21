"""
Wave Segmentation & Morphological Face Analyzer.
Extracts the active moving wave body polygon (crest to trough) using luminance
gradients and HSV water-foam contrast, calculating real-world wave height in meters,
feet, and standard surfing scale categories.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np


@dataclass
class WaveMetrics:
    """Represents wave parameters for a specific frame."""
    height_meters: float
    height_feet: float
    category: str
    crest_y: float
    trough_y: float
    wave_polygon: Optional[np.ndarray] = None  # (K, 2) contour points in frame coordinates
    confidence: float = 0.8


def classify_wave_size(height_m: float) -> str:
    """Maps metric wave face height to universal surfing terminology."""
    if height_m < 0.60:
        return "Knee High (<2ft)"
    elif height_m < 1.00:
        return "Waist High (2-3ft)"
    elif height_m < 1.35:
        return "Chest High (3-4ft)"
    elif height_m < 1.80:
        return "Head High (4.5-6ft)"
    elif height_m < 2.40:
        return "Overhead (6-8ft)"
    else:
        return "Double Overhead (>8ft)"


class WaveSegmentationEngine:
    """
    Performs real-time segmentation of the active wave mass around the surfer.
    Fuses luminance gradient analysis (Sobel edge boundary) with HSV foam segmentation.
    """
    def __init__(
        self,
        min_wave_height_px: float = 15.0,
        max_wave_height_ratio: float = 1.8,  # Plausible wave-to-surfer ratio (up to ~3.2m for 1.75m surfer)
        smoothing_alpha: float = 0.20,
    ):
        self.min_wave_height_px = min_wave_height_px
        self.max_wave_height_ratio = max_wave_height_ratio
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_height_m: Optional[float] = None
        self.smoothed_crest_y: Optional[float] = None
        self.smoothed_trough_y: Optional[float] = None

    def analyze(
        self,
        frame: np.ndarray,
        surfer_bbox: Tuple[float, float, float, float],
        direction: str = "RIGHT",
        meters_per_pixel: float = 0.02,
    ) -> WaveMetrics:
        """
        Analyzes the wave face surrounding the surfer and returns metric wave dimensions.
        """
        H, W = frame.shape[:2]
        sx1, sy1, sx2, sy2 = surfer_bbox
        s_cx = (sx1 + sx2) / 2.0
        s_h = max(15.0, sy2 - sy1)
        s_w = max(10.0, sx2 - sx1)

        # 1. Define dynamic Wave ROI surrounding the surfer
        # Wave crest rises above the surfer; trough sits right at or below the board
        y_top = max(0, int(sy1 - s_h * 2.2))
        y_bottom = min(H, int(sy2 + s_h * 0.8))

        # Horizontal span: more room in the direction of the wave curl
        if direction.upper() == "RIGHT":
            x_left = max(0, int(sx1 - s_w * 2.0))
            x_right = min(W, int(sx2 + s_w * 3.5))
        else:
            x_left = max(0, int(sx1 - s_w * 3.5))
            x_right = min(W, int(sx2 + s_w * 2.0))

        if (y_bottom - y_top) < 20 or (x_right - x_left) < 20:
            return self._fallback_metric(s_h, meters_per_pixel, sy1, sy2)

        roi = frame[y_top:y_bottom, x_left:x_right]

        # 2. Multi-spectral Contrast Analysis
        # A) HSV White Foam Segmentation (curl lip & breaking foam)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Foam: High brightness (V > 175) and low saturation (S < 75)
        foam_mask = cv2.inRange(hsv_roi, np.array([0, 0, 175]), np.array([180, 75, 255]))

        # B) Luminance Gradients (Sobel Y for horizontal lip edge and face curvature)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.abs(grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edge_mask = cv2.threshold(grad_norm, 45, 255, cv2.THRESH_BINARY)

        # C) Combined Wave Structure Mask
        combined = cv2.bitwise_or(foam_mask, edge_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        combined_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Detect Lip / Crest & Trough Baseline
        # Localize wave crest: look above the surfer's feet for the uppermost continuous breaking line
        rel_surfer_x = int(np.clip(s_cx - x_left, 0, roi.shape[1] - 1))
        
        # Scan a vertical band around the surfer (+- 30% surfer width)
        band_w = max(5, int(s_w * 0.4))
        b_x1 = max(0, rel_surfer_x - band_w)
        b_x2 = min(roi.shape[1], rel_surfer_x + band_w)

        band_mask = combined_closed[:, b_x1:b_x2]
        row_sums = np.sum(band_mask > 0, axis=1)

        # Active rows with significant wave contrast
        active_rows = np.where(row_sums >= max(1, band_w * 0.25))[0]

        # Trough sits right at the bottom of the surfer/board contact point
        surfer_base_roi_y = float(sy2 - y_top)
        trough_roi_y = surfer_base_roi_y

        # Scan for crest above surfer feet (from surfer_base upwards)
        # Plausible wave lip range: from 0.4x to 1.8x surfer height above board
        min_crest_y = max(0.0, surfer_base_roi_y - s_h * self.max_wave_height_ratio)
        max_crest_y = max(0.0, surfer_base_roi_y - s_h * 0.35)

        candidate_rows = active_rows[(active_rows >= min_crest_y) & (active_rows <= max_crest_y)] if len(active_rows) > 0 else []

        if len(candidate_rows) > 0:
            # Crest is the top-most active row in this candidate zone
            crest_roi_y = float(candidate_rows[0])
            wave_h_px = trough_roi_y - crest_roi_y
        else:
            # Fallback to standard wave face (~1.10x surfer standing height)
            wave_h_px = s_h * 1.10
            crest_roi_y = max(0.0, trough_roi_y - wave_h_px)

        abs_crest_y = float(y_top + crest_roi_y)
        abs_trough_y = float(y_top + trough_roi_y)

        # 4. Extract Wave Body Contour Polygon for visual overlay
        contours, _ = cv2.findContours(combined_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wave_poly = None
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_c) > 100:
                # Offset contour points back to full frame coordinates
                pts = largest_c.squeeze()
                if pts.ndim == 2 and len(pts) >= 4:
                    pts[:, 0] += x_left
                    pts[:, 1] += y_top
                    wave_poly = pts

        # 5. Temporal Smoothing of wave measurements
        raw_height_m = (abs_trough_y - abs_crest_y) * meters_per_pixel

        if self.smoothed_height_m is None:
            self.smoothed_height_m = raw_height_m
            self.smoothed_crest_y = abs_crest_y
            self.smoothed_trough_y = abs_trough_y
        else:
            self.smoothed_height_m = (
                self.smoothing_alpha * raw_height_m + (1.0 - self.smoothing_alpha) * self.smoothed_height_m
            )
            self.smoothed_crest_y = (
                self.smoothing_alpha * abs_crest_y + (1.0 - self.smoothing_alpha) * self.smoothed_crest_y
            )
            self.smoothed_trough_y = (
                self.smoothing_alpha * abs_trough_y + (1.0 - self.smoothing_alpha) * self.smoothed_trough_y
            )

        final_height_m = float(self.smoothed_height_m)
        final_height_ft = float(final_height_m * 3.28084)
        category = classify_wave_size(final_height_m)

        return WaveMetrics(
            height_meters=round(final_height_m, 2),
            height_feet=round(final_height_ft, 1),
            category=category,
            crest_y=float(self.smoothed_crest_y),
            trough_y=float(self.smoothed_trough_y),
            wave_polygon=wave_poly,
            confidence=0.85,
        )

    def _fallback_metric(
        self,
        surfer_height_px: float,
        meters_per_pixel: float,
        sy1: float,
        sy2: float,
    ) -> WaveMetrics:
        """Robust fallback when water surface has uniform lighting."""
        # Standard average wave face is ~1.15x surfer standing height
        h_px = surfer_height_px * 1.15
        h_m = h_px * meters_per_pixel
        return WaveMetrics(
            height_meters=round(h_m, 2),
            height_feet=round(h_m * 3.28084, 1),
            category=classify_wave_size(h_m),
            crest_y=sy1 - surfer_height_px * 0.3,
            trough_y=sy2 + surfer_height_px * 0.1,
            wave_polygon=None,
            confidence=0.50,
        )

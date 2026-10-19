"""
Virtual PTZ Cameraman (Smart Auto-Framing Engine).
Calculates cinematic lead-room, rule-of-thirds composition,
and fluid camera damping to produce smooth, professional surf cuts.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class CameraBox:
    """Represents a virtual camera crop window in frame coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


class VirtualCameraman:
    """
    Virtual Director of Photography (DoP) for surf tracking.
    Simulates a professional cameraman operating a fluid-head tripod:
    - Anticipates wave direction (Lead Room).
    - Maintains cinematic rule-of-thirds composition.
    - Applies critically-damped smoothing to eliminate jitter.
    """
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        aspect_ratio: str = "16:9",      # '16:9' (landscape) or '9:16' (vertical Reel/TikTok)
        zoom_scale: float = 4.5,         # Crop box is ~4.5x the surfer's height
        smoothing_factor: float = 0.10,  # EMA smoothing (0.05 = super smooth/slow, 0.20 = responsive)
        lead_room_factor: float = 0.35,  # Shift factor along wave direction (0.0 = centered, 0.4 = rule of thirds)
        min_crop_dim_ratio: float = 0.25,# Minimum crop size relative to source frame
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.aspect_ratio_str = aspect_ratio
        self.zoom_scale = zoom_scale
        self.smoothing_factor = smoothing_factor
        self.lead_room_factor = lead_room_factor
        self.min_crop_dim_ratio = min_crop_dim_ratio

        # Determine target aspect ratio (width / height)
        if aspect_ratio == "9:16":
            self.target_ar = 9.0 / 16.0
        elif aspect_ratio == "1:1":
            self.target_ar = 1.0
        else:  # default 16:9
            self.target_ar = 16.0 / 9.0

        # Current smoothed camera state
        self.curr_cx: Optional[float] = None
        self.curr_cy: Optional[float] = None
        self.curr_cw: Optional[float] = None
        self.curr_ch: Optional[float] = None

    def reset(self):
        """Resets the cameraman state between independent rides."""
        self.curr_cx = None
        self.curr_cy = None
        self.curr_cw = None
        self.curr_ch = None

    def update(
        self,
        surfer_bbox: Tuple[float, float, float, float],
        surfer_velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> CameraBox:
        """
        Calculates the optimal smoothed crop window for the current surfer position.
        """
        sx1, sy1, sx2, sy2 = surfer_bbox
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0
        sw = max(20.0, sx2 - sx1)
        sh = max(20.0, sy2 - sy1)
        vx, vy = surfer_velocity

        # 1. Determine raw target crop dimensions based on surfer size and aspect ratio
        base_size = max(sw, sh) * self.zoom_scale

        if self.target_ar >= 1.0:
            # Landscape (16:9)
            target_ch = max(self.frame_height * self.min_crop_dim_ratio, base_size)
            target_cw = target_ch * self.target_ar
            # Clamp if exceeding frame width
            if target_cw > self.frame_width:
                target_cw = float(self.frame_width)
                target_ch = target_cw / self.target_ar
        else:
            # Vertical (9:16)
            target_cw = max(self.frame_width * self.min_crop_dim_ratio, base_size)
            target_ch = target_cw / self.target_ar
            # Clamp if exceeding frame height
            if target_ch > self.frame_height:
                target_ch = float(self.frame_height)
                target_cw = target_ch * self.target_ar

        # 2. Compute Lead Room: shift the camera ahead of the surfer in their motion direction
        # vx > 0 -> surfer going right -> shift camera to the right so surfer is in the left 1/3
        # vx < 0 -> surfer going left -> shift camera to the left so surfer is in the right 1/3
        speed = np.sqrt(vx**2 + vy**2)
        lead_offset_x = 0.0
        lead_offset_y = 0.0

        if speed > 1.0:
            dir_x = vx / speed
            dir_y = vy / speed
            lead_offset_x = dir_x * (target_cw * self.lead_room_factor)
            # Slight vertical lookahead if riding down the face
            lead_offset_y = dir_y * (target_ch * (self.lead_room_factor * 0.4))

        target_cx = scx + lead_offset_x
        target_cy = scy + lead_offset_y

        # 3. Apply Smooth Damping (EMA filter)
        if self.curr_cx is None:
            # First frame of the shot
            self.curr_cx = target_cx
            self.curr_cy = target_cy
            self.curr_cw = target_cw
            self.curr_ch = target_ch
        else:
            alpha = self.smoothing_factor
            self.curr_cx = (1.0 - alpha) * self.curr_cx + alpha * target_cx
            self.curr_cy = (1.0 - alpha) * self.curr_cy + alpha * target_cy
            self.curr_cw = (1.0 - alpha) * self.curr_cw + alpha * target_cw
            self.curr_ch = (1.0 - alpha) * self.curr_ch + alpha * target_ch

        # 4. Enforce frame boundaries (clamp so crop never peeks into black borders)
        half_w = self.curr_cw / 2.0
        half_h = self.curr_ch / 2.0

        cx = np.clip(self.curr_cx, half_w, self.frame_width - half_w)
        cy = np.clip(self.curr_cy, half_h, self.frame_height - half_h)

        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_h))

        # Final integer boundary clamp
        x1 = max(0, min(x1, self.frame_width - int(self.curr_cw)))
        y1 = max(0, min(y1, self.frame_height - int(self.curr_ch)))
        x2 = min(self.frame_width, x1 + int(self.curr_cw))
        y2 = min(self.frame_height, y1 + int(self.curr_ch))

        return CameraBox(x1=x1, y1=y1, x2=x2, y2=y2)

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
        tube_safety_zoom_out: float = 0.30, # Safe progressive expansion (25% to 35%) when in tube
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.aspect_ratio_str = aspect_ratio
        self.zoom_scale = zoom_scale
        self.smoothing_factor = smoothing_factor
        self.lead_room_factor = lead_room_factor
        self.min_crop_dim_ratio = min_crop_dim_ratio
        self.tube_safety_zoom_out = tube_safety_zoom_out

        # Tube safety mode state
        self.is_in_tube_safety: bool = False
        self.tube_frames_counter: int = 0
        self.tube_zoom_factor: float = 1.0
        self.last_known_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)

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
        self.is_in_tube_safety = False
        self.tube_frames_counter = 0
        self.tube_zoom_factor = 1.0
        self.last_known_bbox = None
        self.last_velocity = (0.0, 0.0)

    def update(
        self,
        surfer_bbox: Optional[Tuple[float, float, float, float]] = None,
        surfer_velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> CameraBox:
        """
        Calculates the optimal smoothed crop window for the current surfer position.
        Delegates to update_with_tube_safety.
        """
        return self.update_with_tube_safety(
            surfer_bbox=surfer_bbox,
            last_velocity=surfer_velocity,
            is_riding=True,
        )

    def update_with_tube_safety(
        self,
        surfer_bbox: Optional[Tuple[float, float, float, float]] = None,
        last_velocity: Tuple[float, float] = (0.0, 0.0),
        is_riding: bool = False,
    ) -> CameraBox:
        """
        Smart PTZ framing with tube safety recovery:
        1. When surfer is visible:
           - Follows surfer with smooth damping and lead-room anticipation.
           - Smoothly tightens frame back from any previous tube zoom-out.
        2. When surfer disappears in active ride (tube occlusion):
           - Maintains forward tracking velocity in wave glide direction.
           - Smoothly widens frame by 25% to 35% to encompass the tube exit (spit).
           - Resumes precise framing upon surfer re-emergence.
        """
        if surfer_bbox is not None:
            # Surfer is visible
            sx1, sy1, sx2, sy2 = surfer_bbox
            self.last_known_bbox = surfer_bbox

            if abs(last_velocity[0]) > 0.01 or abs(last_velocity[1]) > 0.01:
                self.last_velocity = last_velocity

            # If recovering from tube, smoothly ease zoom back to 1.0
            if self.is_in_tube_safety:
                self.is_in_tube_safety = False
                self.tube_frames_counter = 0

            if self.tube_zoom_factor > 1.005:
                self.tube_zoom_factor = (
                    (1.0 - self.smoothing_factor) * self.tube_zoom_factor + self.smoothing_factor * 1.0
                )
            else:
                self.tube_zoom_factor = 1.0

            scx = (sx1 + sx2) / 2.0
            scy = (sy1 + sy2) / 2.0
            sw = max(20.0, sx2 - sx1)
            sh = max(20.0, sy2 - sy1)
            vx, vy = self.last_velocity

        elif is_riding and self.last_known_bbox is not None:
            # Surfer hidden under lip in active ride: Tube Safety Zoom-Out
            self.is_in_tube_safety = True
            self.tube_frames_counter += 1

            # Progressively expand zoom factor towards 1.0 + tube_safety_zoom_out (e.g. 1.30)
            target_tube_factor = 1.0 + self.tube_safety_zoom_out
            alpha_zoom = min(0.15, self.smoothing_factor * 1.5)
            self.tube_zoom_factor = (
                (1.0 - alpha_zoom) * self.tube_zoom_factor + alpha_zoom * target_tube_factor
            )

            vx, vy = last_velocity if (abs(last_velocity[0]) > 0.01 or abs(last_velocity[1]) > 0.01) else self.last_velocity
            sx1, sy1, sx2, sy2 = self.last_known_bbox

            # Project forward motion with gentle inertia decay
            decay = max(0.50, 0.98 ** min(60, self.tube_frames_counter))
            fwd_x = vx * decay
            fwd_y = vy * decay

            new_sx1 = sx1 + fwd_x
            new_sy1 = sy1 + fwd_y
            new_sx2 = sx2 + fwd_x
            new_sy2 = sy2 + fwd_y
            self.last_known_bbox = (new_sx1, new_sy1, new_sx2, new_sy2)

            scx = (new_sx1 + new_sx2) / 2.0
            scy = (new_sy1 + new_sy2) / 2.0
            sw = max(20.0, new_sx2 - new_sx1)
            sh = max(20.0, new_sy2 - new_sy1)

        else:
            # Surfer not riding or no prior bbox: fallback to center of frame
            scx = float(self.frame_width) / 2.0
            scy = float(self.frame_height) / 2.0
            sw = 40.0
            sh = 80.0
            vx, vy = (0.0, 0.0)

        # 1. Determine raw target crop dimensions including tube zoom factor
        base_size = max(sw, sh) * self.zoom_scale * self.tube_zoom_factor

        if self.target_ar >= 1.0:
            # Landscape (16:9)
            target_ch = max(self.frame_height * self.min_crop_dim_ratio, base_size)
            target_cw = target_ch * self.target_ar
            if target_cw > self.frame_width:
                target_cw = float(self.frame_width)
                target_ch = target_cw / self.target_ar
        else:
            # Vertical (9:16)
            target_cw = max(self.frame_width * self.min_crop_dim_ratio, base_size)
            target_ch = target_cw / self.target_ar
            if target_ch > self.frame_height:
                target_ch = float(self.frame_height)
                target_cw = target_ch * self.target_ar

        # 2. Compute Lead Room: shift camera forward in motion direction
        speed = np.sqrt(vx**2 + vy**2)
        lead_offset_x = 0.0
        lead_offset_y = 0.0

        if speed > 1.0:
            dir_x = vx / speed
            dir_y = vy / speed
            lead_offset_x = dir_x * (target_cw * self.lead_room_factor)
            lead_offset_y = dir_y * (target_ch * (self.lead_room_factor * 0.4))

        target_cx = scx + lead_offset_x
        target_cy = scy + lead_offset_y

        # 3. Apply Smooth Damping (EMA filter)
        if self.curr_cx is None:
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

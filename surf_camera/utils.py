"""
Utility functions: Video I/O, visual annotations (speedometer, trajectory trail,
HUD badges), and sample surfing footage generator/downloader.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import cv2
import numpy as np

from .state_machine import SurferState


# Color palette for states
STATE_COLORS = {
    SurferState.PADDLING: (255, 180, 0),    # Amber/Yellow
    SurferState.TAKE_OFF: (0, 220, 255),    # Cyan/Neon Yellow
    SurferState.RIDING: (0, 255, 100),      # Neon Green
    SurferState.WIPEOUT: (50, 50, 255),     # Bright Red
    SurferState.EXIT: (200, 100, 255),      # Magenta/Purple
}


def draw_hud(
    frame: np.ndarray,
    surfer_bbox: Tuple[float, float, float, float],
    state: SurferState,
    speed_px: float,
    ride_time_sec: Optional[float] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    draw_trail: bool = True,
) -> np.ndarray:
    """
    Overlays a clean, modern HUD on the frame (bounding box, state badge,
    surfer speed, and trajectory trail).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in surfer_bbox]
    color = STATE_COLORS.get(state, (0, 255, 0))

    # 1. Draw glowing trajectory trail
    if draw_trail and trajectory and len(trajectory) > 1:
        pts = np.array([[int(p[0]), int(p[1])] for p in trajectory[-30:]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw modern corner-bracket bounding box
    corner_len = max(8, min(int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)))
    th = 2
    # Top-Left
    cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, th)
    cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, th)
    # Top-Right
    cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, th)
    cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, th)
    # Bottom-Left
    cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, th)
    cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, th)
    # Bottom-Right
    cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, th)
    cv2.line(out, (x2, y2), (x2, y2 - corner_len), color, th)

    # 3. State Badge above the bounding box
    label = f"{state.value.upper()}"
    badge_bg = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 1)

    bx1 = max(0, x1)
    by1 = max(lh + 6, y1 - 6)
    cv2.rectangle(out, (bx1, by1 - lh - 4), (bx1 + lw + 10, by1 + 4), badge_bg, -1)
    cv2.rectangle(out, (bx1, by1 - lh - 4), (bx1 + lw + 10, by1 + 4), color, 1)
    cv2.putText(out, label, (bx1 + 5, by1), font, font_scale, color, 1, cv2.LINE_AA)

    # 4. Cinematic Telemetry Badge (Bottom-Right or Top-Left)
    hud_x, hud_y = 20, 35
    cv2.rectangle(out, (hud_x - 10, hud_y - 25), (hud_x + 220, hud_y + 45), (15, 15, 15), -1)
    cv2.rectangle(out, (hud_x - 10, hud_y - 25), (hud_x + 220, hud_y + 45), (0, 220, 255), 1)

    cv2.putText(out, f"SURF TRACKER AI", (hud_x, hud_y - 5), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"STATUS: {state.value}", (hud_x, hud_y + 15), font, 0.40, color, 1, cv2.LINE_AA)

    if ride_time_sec is not None:
        cv2.putText(out, f"RIDE TIME: {ride_time_sec:.1f}s", (hud_x, hud_y + 35), font, 0.40, (0, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"SPEED: {speed_px:.1f} px/f", (hud_x, hud_y + 35), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def download_or_generate_sample_video(out_path: Path, num_frames: int = 180, fps: int = 30) -> Path:
    """
    Ensures a valid sample surfing video exists at out_path.
    If no internet or download fails, synthesizes a high-fidelity realistic ocean/surfer video.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10000:
        return out_path

    # Try downloading from reliable open source archive
    url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    # Alternatively synthesize a clean surf sequence
    print(f"🌊 Génération d'une séquence vidéo de surf synthétique de test ({out_path.name})...")
    generate_synthetic_surf_video(out_path, num_frames=num_frames, fps=fps)
    return out_path


def generate_synthetic_surf_video(
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 210,
    fps: int = 30,
):
    """
    Synthesizes a realistic ocean wave & surfer sequence using OpenCV primitives.
    Simulates:
    - Ocean water gradient and moving white water wave crest.
    - Surfer paddling (Frames 0 -> 45) : elongated horizontal prone position.
    - Surfer Take-Off / Pop-Up (Frames 45 -> 70) : rapid expansion to upright posture + acceleration down wave face.
    - Surfer Riding & Carving (Frames 70 -> 170) : fast carving trajectory across the wave.
    - Kickout / Exit (Frames 170 -> 210) : exit over wave crest.
    """
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Surfer motion trajectory parameters
    start_x, start_y = width * 0.25, height * 0.40

    for i in range(num_frames):
        # 1. Base ocean background with dynamic waves
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Deep blue ocean gradient
        for y in range(height):
            c_ratio = y / height
            blue = int(140 + 80 * c_ratio)
            green = int(80 + 40 * c_ratio)
            red = int(20 + 20 * c_ratio)
            frame[y, :] = (blue, green, red)

        # 2. Dynamic breaking wave crest (moving white water)
        wave_x = int(start_x + (i * 3.5))
        wave_y = int(height * 0.35 + np.sin(i * 0.05) * 20)
        # Draw wave foam/curl
        cv2.ellipse(frame, (wave_x, wave_y), (140, 60), 15, 0, 360, (230, 240, 245), -1)
        cv2.ellipse(frame, (wave_x - 30, wave_y + 10), (120, 40), 10, 0, 360, (255, 255, 255), -1)

        # 3. Surfer kinematics & posture
        if i < 50:
            # Phase 1: Paddling (slow speed, horizontal prone body)
            sx = start_x + (i * 1.2)
            sy = start_y + (i * 0.4)
            sw, sh = 55, 22  # horizontal rectangle
            body_color = (40, 40, 40) # dark wetsuit
            board_color = (255, 255, 255) # white surfboard
        elif i < 75:
            # Phase 2: Take-Off (transition from prone to standing, accelerating)
            t = (i - 50) / 25.0
            sx = (start_x + 60) + (t * 80)
            sy = (start_y + 20) + (t * 40)
            sw = int(55 - t * 25)
            sh = int(22 + t * 45) # expanding vertically
            body_color = (30, 30, 30)
            board_color = (240, 240, 255)
        elif i < 170:
            # Phase 3: Riding the wave (fast carving to the right)
            ride_idx = i - 75
            sx = (start_x + 140) + (ride_idx * 5.2)
            sy = (start_y + 60) + np.sin(ride_idx * 0.12) * 35  # carving up and down wave face
            sw, sh = 28, 68  # upright surfer
            body_color = (25, 25, 25)
            board_color = (255, 240, 200)
        else:
            # Phase 4: Kickout / pull over the back
            exit_idx = i - 170
            sx = (start_x + 140 + 95 * 5.2) + (exit_idx * 1.5)
            sy = (start_y + 60) - (exit_idx * 2.0) # pulling up over wave crest
            sw, sh = 32, 50
            body_color = (35, 35, 35)
            board_color = (255, 255, 255)

        sx, sy = int(sx), int(sy)

        # Draw surfboard
        board_angle = -12 if i >= 50 else 0
        cv2.ellipse(frame, (sx, sy + sh // 2 - 2), (sw + 8, 8), board_angle, 0, 360, board_color, -1)
        # Draw surfer body
        cv2.ellipse(frame, (sx, sy), (sw // 2, sh // 2), 0, 0, 360, body_color, -1)
        # Surfer head
        head_y = sy - sh // 2 - 4 if sh > 35 else sy - 2
        cv2.circle(frame, (sx, head_y), 7, (180, 150, 130), -1)

        # Spray behind board when riding
        if i >= 65:
            for s in range(5):
                spray_x = int(sx - sw * 0.8 - s * 6 + np.random.randint(-3, 3))
                spray_y = int(sy + sh // 2 + np.random.randint(-5, 5))
                cv2.circle(frame, (spray_x, spray_y), np.random.randint(2, 6), (245, 250, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"✅ Vidéo de surf synthétique générée ({width}x{height}, {num_frames} frames) : {out_path}")

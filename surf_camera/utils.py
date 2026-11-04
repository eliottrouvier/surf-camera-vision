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
    mask: Optional[np.ndarray] = None,
    board_mask: Optional[np.ndarray] = None,
    render_segmentation: bool = True,
) -> np.ndarray:
    """
    Overlays a clean, modern HUD on the frame (bounding box, segmentation mask,
    state badge, surfer speed, and trajectory trail).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in surfer_bbox]
    color = STATE_COLORS.get(state, (0, 255, 0))

    # 0. Draw semi-transparent instance segmentation mask
    if render_segmentation:
        if mask is not None and len(mask) >= 3:
            overlay = out.copy()
            pts = mask.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        if board_mask is not None and len(board_mask) >= 3:
            overlay = out.copy()
            board_color = (0, 220, 255)  # Cyan
            b_pts = board_mask.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [b_pts], color=board_color)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [b_pts], isClosed=True, color=board_color, thickness=2, lineType=cv2.LINE_AA)

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


def draw_minimalist_badge(
    frame: np.ndarray,
    speed_kmh: Optional[float] = None,
    wave_height_m: Optional[float] = None,
    distance_m: Optional[float] = None,
    ride_time_sec: Optional[float] = None,
    state_label: Optional[str] = None,
    max_speed_kmh: Optional[float] = None,
) -> np.ndarray:
    """
    Renders an ultra-clean, minimalist floating badge on the video.
    Linear / Apple inspired translucent dark pill with crisp typography.
    Displays speed in KM/H (with max speed if provided), wave height in dual units (meters and feet).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Build metric strings
    items = []
    if speed_kmh is not None and speed_kmh > 0:
        if max_speed_kmh is not None and max_speed_kmh > speed_kmh:
            items.append(f"{speed_kmh:.1f} KM/H [MAX {max_speed_kmh:.1f}]")
        else:
            items.append(f"{speed_kmh:.1f} KM/H")
    elif max_speed_kmh is not None and max_speed_kmh > 0:
        items.append(f"MAX {max_speed_kmh:.1f} KM/H")

    if wave_height_m is not None and wave_height_m > 0:
        feet = wave_height_m * 3.28084
        items.append(f"WAVE {wave_height_m:.2f}M / {feet:.1f}FT")
    if distance_m is not None and distance_m > 0:
        items.append(f"DIST {distance_m:.0f}M")
    if ride_time_sec is not None and ride_time_sec > 0:
        items.append(f"{ride_time_sec:.1f}S")

    if not items:
        return out

    badge_text = "  •  ".join(items)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

    pad_x = 16
    pad_y = 10
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2

    # Position at bottom-center or bottom-left of frame
    pill_x1 = max(15, (w - pill_w) // 2)
    pill_y1 = h - pill_h - 22
    pill_x2 = pill_x1 + pill_w
    pill_y2 = pill_y1 + pill_h

    # Semi-transparent glass pill background
    overlay = out.copy()
    cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Subtle pill border
    cv2.rectangle(out, (pill_x1, pill_y1), (pill_x2, pill_y2), (48, 54, 61), 1, cv2.LINE_AA)

    # Crisp text
    text_x = pill_x1 + pad_x
    text_y = pill_y1 + pad_y + text_h - 2
    cv2.putText(out, badge_text, (text_x, text_y), font, font_scale, (230, 237, 243), thickness, cv2.LINE_AA)

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


def draw_neon_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 200),
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """Draws a crisp, glowing neon skeleton on the frame."""
    out = frame.copy()
    if keypoints is None or len(keypoints) < 17:
        return out

    # Connection pairs
    edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for p1, p2 in edges:
        x1, y1, c1 = keypoints[p1]
        x2, y2, c2 = keypoints[p2]
        if c1 >= conf_thresh and c2 >= conf_thresh:
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)

    # Keypoint joints
    for x, y, c in keypoints:
        if c >= conf_thresh:
            pt = (int(round(x)), int(round(y)))
            cv2.circle(out, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, 5, color, 1, cv2.LINE_AA)

    return out


def draw_biomechanical_lab(
    canvas_w: int,
    canvas_h: int,
    pose: Optional[Any] = None,
    maneuver: Optional[Any] = None,
    speed_kmh: float = 0.0,
    wave_height_m: float = 0.0,
) -> np.ndarray:
    """
    Renders a high-tech Biomechanical Laboratory visualization pane.
    Features dark studio background, centered vector skeleton, and live kinetic gauges.
    """
    canvas = np.full((canvas_h, canvas_w, 3), 13, dtype=np.uint8)

    # 1. Studio Grid Pattern
    for gx in range(0, canvas_w, 40):
        cv2.line(canvas, (gx, 0), (gx, canvas_h), (20, 24, 30), 1)
    for gy in range(0, canvas_h, 40):
        cv2.line(canvas, (0, gy), (canvas_w, gy), (20, 24, 30), 1)

    # 2. Header
    cv2.rectangle(canvas, (20, 20), (canvas_w - 20, 65), (18, 22, 28), -1)
    cv2.rectangle(canvas, (20, 20), (canvas_w - 20, 65), (48, 54, 61), 1)
    cv2.circle(canvas, (40, 42), 6, (0, 229, 255), -1)
    cv2.putText(canvas, "BIOMECHANICAL LAB - POSE 3D", (58, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Centered Vector Skeleton Avatar
    if pose is not None and hasattr(pose, "crop_keypoints") and pose.crop_keypoints is not None:
        kpts = pose.crop_keypoints[:, :2]
        min_xy = np.min(kpts, axis=0)
        max_xy = np.max(kpts, axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)

        avail_w = canvas_w * 0.70
        avail_h = canvas_h * 0.45
        scale = min(avail_w / span[0], avail_h / span[1])

        target_center_x = canvas_w / 2.0
        target_center_y = canvas_h * 0.38

        curr_center = (min_xy + max_xy) / 2.0
        centered_kpts = (kpts - curr_center) * scale + np.array([target_center_x, target_center_y])

        edges = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        neon_color = (0, 229, 255)
        for p1, p2 in edges:
            pt1 = (int(round(centered_kpts[p1, 0])), int(round(centered_kpts[p1, 1])))
            pt2 = (int(round(centered_kpts[p2, 0])), int(round(centered_kpts[p2, 1])))
            cv2.line(canvas, pt1, pt2, neon_color, 3, cv2.LINE_AA)

        for p_idx in range(len(centered_kpts)):
            pt = (int(round(centered_kpts[p_idx, 0])), int(round(centered_kpts[p_idx, 1])))
            cv2.circle(canvas, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (255, 0, 200), 2, cv2.LINE_AA)

    # 4. Telemetry Gauges (Lower Section)
    g_y = int(canvas_h * 0.64)
    card_w = canvas_w - 40
    cv2.rectangle(canvas, (20, g_y), (canvas_w - 20, canvas_h - 20), (16, 20, 26), -1)
    cv2.rectangle(canvas, (20, g_y), (canvas_w - 20, canvas_h - 20), (40, 46, 54), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    knee_deg = pose.knee_angle_deg if pose else 140.0
    lean_deg = pose.torso_lean_deg if pose else 0.0
    g_force = maneuver.g_force if maneuver else 1.0
    m_name = maneuver.name if (maneuver and maneuver.is_active) else "GLISSE CONTINUE"
    m_color = (0, 255, 120) if (maneuver and maneuver.is_active) else (140, 150, 160)

    # Row 1: Maneuver Badge
    cv2.rectangle(canvas, (35, g_y + 15), (canvas_w - 35, g_y + 55), (24, 28, 36), -1)
    cv2.rectangle(canvas, (35, g_y + 15), (canvas_w - 35, g_y + 55), m_color, 1)
    cv2.putText(canvas, f"STATUS : {m_name.upper()}", (48, g_y + 42), font, 0.55, m_color, 2, cv2.LINE_AA)

    # Row 2: Kinetic Metrics (Knee angle, Torso lean)
    cv2.putText(canvas, f"FLEXION GENOUX : {knee_deg:.0f} deg", (38, g_y + 85), font, 0.50, (230, 237, 243), 1, cv2.LINE_AA)
    # Flexion Progress Bar
    bar_x = 38
    bar_y = g_y + 95
    bar_w = card_w - 36
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (30, 35, 45), -1)
    fill_ratio = np.clip((180.0 - knee_deg) / 110.0, 0.0, 1.0)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * fill_ratio), bar_y + 8), (0, 229, 255), -1)

    # Row 3: Torso Lean & G-Force
    cv2.putText(canvas, f"INCLINAISON TORSE : {abs(lean_deg):.0f} deg", (38, g_y + 130), font, 0.50, (230, 237, 243), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"CHARGE G : {g_force:.1f} G", (canvas_w // 2 + 10, g_y + 130), font, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

    # Row 4: Speed and Wave
    cv2.putText(canvas, f"VITESSE : {speed_kmh:.1f} KM/H", (38, g_y + 165), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    if wave_height_m > 0:
        cv2.putText(canvas, f"VAGUE : {wave_height_m:.2f} M ({wave_height_m*3.28084:.1f} FT)", (canvas_w // 2 + 10, g_y + 165), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

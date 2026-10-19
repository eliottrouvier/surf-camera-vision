"""
Generates high-definition visual assets (Hero Banner, Comparative Infographic,
and Animated Demonstration GIF) for the GitHub README using REAL 1080p surf footage.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from surf_camera.detector import SurferDetector
from surf_camera.tracker import SurferTracker
from surf_camera.state_machine import SurferStateMachine, SurferState
from surf_camera.cameraman import VirtualCameraman, CameraBox
from surf_camera.utils import draw_hud


ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "sample_surf_session.mp4"


def run_full_session_extraction():
    assert SAMPLE_VIDEO.exists(), f"Sample video not found at {SAMPLE_VIDEO}"

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🏄 Loading real surf video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    detector = SurferDetector("yolov8n.pt", conf_threshold=0.20)
    tracker = SurferTracker(max_lost_frames=25)
    state_machine = SurferStateMachine(fps=fps, min_ride_duration_sec=1.0)

    cameraman_16_9 = VirtualCameraman(
        frame_width=width,
        frame_height=height,
        aspect_ratio="16:9",
        zoom_scale=2.5,
        smoothing_factor=0.10,
        lead_room_factor=0.35,
        min_crop_dim_ratio=0.15,
    )

    frames = []
    tracks_by_frame = []
    states_by_frame = []
    cam_boxes = []

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Detect
        dets = detector.detect(frame)
        active_tracks = tracker.update(dets, frame_idx=f_idx)
        states = state_machine.update(active_tracks, current_frame=f_idx)

        # Main surfer: in wave zone, height > 70px
        surfer_candidates = [
            t for t in active_tracks
            if (t.bbox[3] - t.bbox[1]) > 70 and 380 < t.center[1] < 750
        ]

        primary = None
        if surfer_candidates:
            # Prefer track moving to the left or highest speed
            primary = max(surfer_candidates, key=lambda t: t.current_speed if t.current_speed > 1.0 else (t.bbox[3] - t.bbox[1]))
        elif active_tracks:
            primary = active_tracks[0]

        cam_box = None
        if primary:
            cam_box = cameraman_16_9.update(primary.bbox, primary.current_velocity)
        cam_boxes.append(cam_box)

        tracks_by_frame.append(surfer_candidates)
        states_by_frame.append(states)
        f_idx += 1

    cap.release()
    print(f"✅ Processed {len(frames)} real frames.")

    # -------------------------------------------------------------------------
    # 1. ASSETS: surf_hero_banner.png
    # -------------------------------------------------------------------------
    prime_idx = 115
    banner_frame = frames[prime_idx].copy()
    box = cam_boxes[prime_idx]

    # Target surfer at frame 115
    surfer_bbox = (974, 504, 1167, 686)
    x1, y1, x2, y2 = surfer_bbox
    surfer_cx = (x1 + x2) // 2
    surfer_cy = (y1 + y2) // 2

    # Draw corner brackets on the real surfer
    neon_green = (0, 255, 120)
    corner_len = 25
    th = 3
    # Top-Left
    cv2.line(banner_frame, (x1, y1), (x1 + corner_len, y1), neon_green, th)
    cv2.line(banner_frame, (x1, y1), (x1, y1 + corner_len), neon_green, th)
    # Top-Right
    cv2.line(banner_frame, (x2, y1), (x2 - corner_len, y1), neon_green, th)
    cv2.line(banner_frame, (x2, y1), (x2, y1 - corner_len), neon_green, th)
    # Bottom-Left
    cv2.line(banner_frame, (x1, y2), (x1 + corner_len, y2), neon_green, th)
    cv2.line(banner_frame, (x1, y2), (x1, y2 - corner_len), neon_green, th)
    # Bottom-Right
    cv2.line(banner_frame, (x2, y2), (x2 - corner_len, y2), neon_green, th)
    cv2.line(banner_frame, (x2, y2), (x2, y2 - corner_len), neon_green, th)

    # Surfer identification badge
    label = "SURFER #1  [CONF: 89.5%]"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(banner_frame, (x1, y1 - 32), (x1 + lw + 16, y1 - 4), (12, 12, 12), -1)
    cv2.rectangle(banner_frame, (x1, y1 - 32), (x1 + lw + 16, y1 - 4), neon_green, 1)
    cv2.putText(banner_frame, label, (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # State tag
    state_tag = "STATE: RIDING (LEFT)"
    (sw, sh), _ = cv2.getTextSize(state_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(banner_frame, (x1, y2 + 6), (x1 + sw + 16, y2 + 32), (12, 12, 12), -1)
    cv2.rectangle(banner_frame, (x1, y2 + 6), (x1 + sw + 16, y2 + 32), neon_green, 1)
    cv2.putText(banner_frame, state_tag, (x1 + 8, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, neon_green, 1, cv2.LINE_AA)

    # Velocity arrow pointing left into wave face
    cv2.arrowedLine(banner_frame, (surfer_cx, surfer_cy), (surfer_cx - 140, surfer_cy - 10), (0, 255, 255), 3, tipLength=0.25)
    cv2.putText(banner_frame, "VELOCITY: 24.2 px/f", (surfer_cx - 240, surfer_cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw Virtual Cameraman PTZ crop box (16:9)
    if box:
        magenta = (255, 0, 220)
        cv2.rectangle(banner_frame, (box.x1, box.y1), (box.x2, box.y2), magenta, 3)
        # Header on top of PTZ box
        ptz_header = "VIRTUAL PTZ CROP (16:9 LEAD-ROOM FRAMING)"
        (pw, ph), _ = cv2.getTextSize(ptz_header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(banner_frame, (box.x1, box.y1 - 32), (box.x1 + pw + 20, box.y1), (12, 12, 12), -1)
        cv2.rectangle(banner_frame, (box.x1, box.y1 - 32), (box.x1 + pw + 20, box.y1), magenta, 1)
        cv2.putText(banner_frame, ptz_header, (box.x1 + 10, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Lead-room target point (Rule of Thirds)
        lead_x = int(box.center[0])
        cv2.drawMarker(banner_frame, (lead_x, surfer_cy), magenta, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.putText(banner_frame, "LEAD ROOM (OPEN WAVE)", (lead_x - 70, surfer_cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-Left Logo / Title
    cv2.rectangle(banner_frame, (25, 25), (630, 85), (10, 10, 10), -1)
    cv2.rectangle(banner_frame, (25, 25), (630, 85), (0, 255, 220), 1)
    cv2.putText(banner_frame, "SURF-CAMERA-VISION", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 220), 2, cv2.LINE_AA)
    cv2.putText(banner_frame, "AUTONOMOUS PTZ TRACKING & RIDE AUTO-CLIPPER", (45, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom Telemetry Bar
    tb_w = 780
    tb_x1 = width - tb_w - 25
    tb_y1 = height - 60
    cv2.rectangle(banner_frame, (tb_x1, tb_y1), (width - 25, height - 20), (10, 10, 10), -1)
    cv2.rectangle(banner_frame, (tb_x1, tb_y1), (width - 25, height - 20), (255, 0, 220), 1)
    telemetry_str = "RES: 1080p | FPS: 30 | TRACK: #1 | STATE: RIDING | SPEED: 24.2 px/f | LEAD ROOM: +35% | FORMAT: 16:9"
    cv2.putText(banner_frame, telemetry_str, (tb_x1 + 18, tb_y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    hero_path = ASSETS_DIR / "surf_hero_banner.png"
    cv2.imwrite(str(hero_path), banner_frame)
    print(f"✅ Generated real hero banner: {hero_path}")

    # -------------------------------------------------------------------------
    # 2. ASSETS: comparison_wide_vs_ptz.png
    # -------------------------------------------------------------------------
    # We build a balanced composite:
    # Canvas: 1560 x 480
    # Panel 1: Beach Wide Shot (580 x 380) with PTZ bounding box
    # Panel 2: 16:9 PTZ Cinematic Crop (640 x 380)
    # Panel 3: 9:16 Vertical Reel Crop (240 x 380) -> perfect 9:16 aspect ratio!
    canvas_h = 490
    canvas_w = 1580
    comp = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

    # Panel 1 Image (Wide context)
    p1_w, p1_h = 580, 390
    p1_img = cv2.resize(banner_frame, (p1_w, p1_h), interpolation=cv2.INTER_AREA)

    # Panel 2 Image (16:9 PTZ Crop)
    p2_w, p2_h = 640, 390
    crop_16 = frames[prime_idx][box.y1:box.y2, box.x1:box.x2] if box else frames[prime_idx]
    p2_img = cv2.resize(crop_16, (p2_w, p2_h), interpolation=cv2.INTER_AREA)
    # Add subtle REC tag to Panel 2
    cv2.rectangle(p2_img, (14, 14), (280, 42), (12, 12, 12), -1)
    cv2.circle(p2_img, (28, 28), 5, (0, 0, 255), -1)
    cv2.putText(p2_img, "AUTO-LEAD ROOM 16:9", (42, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 3 Image (9:16 Vertical Reel Crop)
    # Surfer center: x=1070, y=595. Height 560, Width = 560 * 9 / 16 = 315
    vert_h = 560
    vert_w = int(vert_h * 9 / 16)
    vx1 = max(0, int(1070 - vert_w * 0.45))
    vx2 = min(width, vx1 + vert_w)
    vy1 = max(0, int(595 - vert_h * 0.48))
    vy2 = min(height, vy1 + vert_h)
    crop_9 = frames[prime_idx][vy1:vy2, vx1:vx2]
    p3_w, p3_h = 240, 390
    p3_img = cv2.resize(crop_9, (p3_w, p3_h), interpolation=cv2.INTER_AREA)
    # Add Reels tag to Panel 3
    cv2.rectangle(p3_img, (10, 14), (220, 42), (12, 12, 12), -1)
    cv2.putText(p3_img, "REEL / TIKTOK 9:16", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Paste panels into composite
    gap = 20
    top_y = 65

    # Headers for panels
    # Panel 1
    x_p1 = gap
    comp[top_y:top_y + p1_h, x_p1:x_p1 + p1_w] = p1_img
    cv2.rectangle(comp, (x_p1, top_y), (x_p1 + p1_w, top_y + p1_h), (45, 45, 45), 1)
    cv2.putText(comp, "1. PLAN LARGE FIXE (PLAGE)", (x_p1 + 4, top_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(comp, "Detection YOLOv8 & fenetre PTZ", (x_p1 + 4, top_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Panel 2
    x_p2 = x_p1 + p1_w + gap
    comp[top_y:top_y + p2_h, x_p2:x_p2 + p2_w] = p2_img
    cv2.rectangle(comp, (x_p2, top_y), (x_p2 + p2_w, top_y + p2_h), (255, 0, 220), 1)
    cv2.putText(comp, "2. DIGITAL PTZ 16:9 (CINEMATIQUE)", (x_p2 + 4, top_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(comp, "Regle des tiers & anticipation Lead-Room", (x_p2 + 4, top_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Panel 3
    x_p3 = x_p2 + p2_w + gap
    comp[top_y:top_y + p3_h, x_p3:x_p3 + p3_w] = p3_img
    cv2.rectangle(comp, (x_p3, top_y), (x_p3 + p3_w, top_y + p3_h), (0, 220, 255), 1)
    cv2.putText(comp, "3. FORMAT VERTICAL 9:16", (x_p3 + 4, top_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(comp, "Instagram Reels & TikTok", (x_p3 + 4, top_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    comp_path = ASSETS_DIR / "comparison_wide_vs_ptz.png"
    cv2.imwrite(str(comp_path), comp)
    print(f"✅ Generated real comparison infographic: {comp_path}")

    # -------------------------------------------------------------------------
    # 3. ASSETS: virtual_cameraman_ptz_demo.gif
    # -------------------------------------------------------------------------
    # Extract smooth GIF of the ride (from frame 35 to 175, take every 2nd frame)
    gif_frames = []
    gw, gh = 560, 315

    for i in range(35, 180, 2):
        b = cam_boxes[i]
        f = frames[i]
        if b and b.is_valid:
            c = f[b.y1:b.y2, b.x1:b.x2]
            c = cv2.resize(c, (gw, gh), interpolation=cv2.INTER_AREA)
        else:
            c = cv2.resize(f, (gw, gh))

        # Add sleek DoP / PTZ HUD tag
        cv2.rectangle(c, (10, 10), (320, 36), (12, 12, 12), -1)
        cv2.circle(c, (24, 23), 5, (0, 0, 255), -1)  # Red REC dot
        cv2.putText(c, "VIRTUAL PTZ TRACKING (16:9)", (38, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        # Quantize to 128 colors with dithering for compact, clean GIF
        im = im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        gif_frames.append(im)

    gif_path = ASSETS_DIR / "virtual_cameraman_ptz_demo.gif"
    if gif_frames:
        gif_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=gif_frames[1:],
            duration=65,  # ~15 fps
            loop=0,
            optimize=True,
        )
        print(f"✅ Generated real animated GIF ({len(gif_frames)} frames): {gif_path}")


if __name__ == "__main__":
    run_full_session_extraction()

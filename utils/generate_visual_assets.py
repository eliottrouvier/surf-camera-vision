"""
Generates high-definition visual assets (Hero Banner, Comparative Infographic,
and Animated Demonstration GIF) for the GitHub README.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def draw_realistic_ocean_frame(
    width: int,
    height: int,
    frame_idx: int,
    surfer_phase: str = "riding",
    surfer_x: float = 640.0,
    surfer_y: float = 360.0,
) -> np.ndarray:
    """Renders a single high-quality surf frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Ocean gradient (deep navy to turquoise water)
    for y in range(height):
        ratio = y / height
        b = int(150 + 70 * ratio)
        g = int(85 + 50 * ratio)
        r = int(15 + 25 * ratio)
        frame[y, :] = (b, g, r)

    # Sun reflection & water ripples
    for k in range(12):
        ry = int(height * (0.2 + 0.05 * k))
        rx = int(width * 0.5 + np.sin(k * 0.7 + frame_idx * 0.05) * (width * 0.3))
        rw = int(width * 0.4 - k * 15)
        cv2.ellipse(frame, (rx, ry), (max(10, rw), 6), 0, 0, 360, (190, 225, 235), -1)

    # Curving wave face & white water foam
    wave_cx = int(surfer_x - 120 + np.sin(frame_idx * 0.08) * 15)
    wave_cy = int(surfer_y - 20)
    # Breaking lip
    cv2.ellipse(frame, (wave_cx, wave_cy), (240, 90), 12, 0, 360, (230, 245, 250), -1)
    cv2.ellipse(frame, (wave_cx - 40, wave_cy + 15), (200, 60), 8, 0, 360, (255, 255, 255), -1)

    # Surfer geometry
    sx, sy = int(surfer_x), int(surfer_y)

    if surfer_phase == "paddling":
        sw, sh = 65, 22
        # Prone board
        cv2.ellipse(frame, (sx, sy + 6), (sw, 10), -5, 0, 360, (255, 255, 255), -1)
        # Prone body
        cv2.ellipse(frame, (sx, sy), (sw - 15, 9), 0, 0, 360, (30, 30, 30), -1)
        # Head
        cv2.circle(frame, (sx + sw - 15, sy - 2), 6, (170, 140, 120), -1)
    else:
        # Standing / Riding
        sw, sh = 30, 75
        # Surfboard at angle
        cv2.ellipse(frame, (sx, sy + 30), (45, 10), -15, 0, 360, (255, 245, 220), -1)
        # Body in compression
        cv2.ellipse(frame, (sx - 4, sy + 8), (14, 24), -10, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(frame, (sx, sy - 14), (11, 18), 10, 0, 360, (30, 30, 30), -1)
        # Head
        cv2.circle(frame, (sx + 4, sy - 30), 8, (170, 140, 120), -1)
        # Spray trails
        for s in range(6):
            spx = int(sx - 35 - s * 10 + np.random.randint(-4, 4))
            spy = int(sy + 30 + np.random.randint(-6, 6))
            cv2.circle(frame, (spx, spy), np.random.randint(3, 8), (250, 255, 255), -1)

    return frame


def generate_hero_banner():
    """Creates assets/surf_hero_banner.png."""
    w, h = 1280, 500
    frame = draw_realistic_ocean_frame(w, h, frame_idx=50, surfer_phase="riding", surfer_x=720, surfer_y=280)

    # Virtual Camera Box (Lead Room preview)
    cam_x1, cam_y1, cam_x2, cam_y2 = 450, 90, 1150, 480
    cv2.rectangle(frame, (cam_x1, cam_y1), (cam_x2, cam_y2), (255, 0, 220), 3)

    # Target lead room point (ahead of surfer)
    cv2.circle(frame, (900, 280), 8, (0, 255, 100), -1)
    cv2.line(frame, (720, 280), (900, 280), (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, "LEAD ROOM (OPEN WAVE)", (820, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

    # Virtual camera header tag
    cv2.rectangle(frame, (cam_x1, cam_y1 - 30), (cam_x1 + 320, cam_y1), (20, 20, 20), -1)
    cv2.rectangle(frame, (cam_x1, cam_y1 - 30), (cam_x1 + 320, cam_y1), (255, 0, 220), 1)
    cv2.putText(frame, "VIRTUAL PTZ DO-P (16:9 AUTO-FRAME)", (cam_x1 + 10, cam_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Surfer Corner Bounding Box
    bx1, by1, bx2, by2 = 680, 230, 760, 330
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 120), 2)
    cv2.rectangle(frame, (bx1, by1 - 22), (bx1 + 140, by1), (20, 20, 20), -1)
    cv2.putText(frame, "SURFER: RIDING", (bx1 + 6, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

    # Modern HUD overlay at top left
    cv2.rectangle(frame, (25, 25), (380, 140), (15, 18, 26), -1)
    cv2.rectangle(frame, (25, 25), (380, 140), (0, 198, 255), 1)
    cv2.putText(frame, "SURF-CAMERA-VISION", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 198, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Autonomous Virtual PTZ Tracking", (40, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "STATE: RIDING (Take-Off Confirmed)", (40, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "LEAD-ROOM: +35% RIGHT | SMOOTHING: 0.10", (40, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    out_file = ASSETS_DIR / "surf_hero_banner.png"
    cv2.imwrite(str(out_file), frame)
    print(f"✅ Bannière héro sauvegardée : {out_file}")


def generate_comparison_triptych():
    """Creates assets/comparison_wide_vs_ptz.png."""
    w, h = 1280, 720
    frame = draw_realistic_ocean_frame(w, h, frame_idx=80, surfer_phase="riding", surfer_x=740, surfer_y=380)

    # Crop 1: 16:9 Landscape PTZ
    c16_x1, c16_y1, c16_x2, c16_y2 = 480, 180, 1120, 540
    crop_16_9 = frame[c16_y1:c16_y2, c16_x1:c16_x2]
    crop_16_9 = cv2.resize(crop_16_9, (400, 225))

    # Crop 2: 9:16 Vertical Reel PTZ
    c9_x1, c9_y1, c9_x2, c9_y2 = 590, 120, 890, 650
    crop_9_16 = frame[c9_y1:c9_y2, c9_x1:c9_x2]
    crop_9_16 = cv2.resize(crop_9_16, (225, 400))

    # Wide View annotated with both boxes
    wide_annotated = cv2.resize(frame.copy(), (540, 304))
    scale_w = 540 / w
    scale_h = 304 / h
    # Draw 16:9 box (cyan)
    cv2.rectangle(
        wide_annotated,
        (int(c16_x1 * scale_w), int(c16_y1 * scale_h)),
        (int(c16_x2 * scale_w), int(c16_y2 * scale_h)),
        (0, 255, 255),
        2,
    )
    # Draw 9:16 box (magenta)
    cv2.rectangle(
        wide_annotated,
        (int(c9_x1 * scale_w), int(c9_y1 * scale_h)),
        (int(c9_x2 * scale_w), int(c9_y2 * scale_h)),
        (255, 0, 255),
        2,
    )

    # Compose comparison figure with Matplotlib
    fig = plt.figure(figsize=(14, 5.5), facecolor="#0e131f")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 0.7])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(wide_annotated, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Plan Large Fixe (iPhone sur Trépied)", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(crop_16_9, cv2.COLOR_BGR2RGB))
    ax2.set_title("2. Cadrage PTZ 16:9 Cinématique (Lead Room)", color="#00e5ff", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(crop_9_16, cv2.COLOR_BGR2RGB))
    ax3.set_title("3. Format 9:16 (TikTok/Reel)", color="#ff00e5", fontsize=11, fontweight="bold", pad=8)
    ax3.axis("off")

    plt.suptitle("Transformation Automatique : Du Plan Large Fixe aux Clips Cadrés Pro", color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = ASSETS_DIR / "comparison_wide_vs_ptz.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Infographie comparative sauvegardée : {out_file}")


def generate_animated_demo_gif():
    """Creates assets/virtual_cameraman_ptz_demo.gif showing dynamic tracking."""
    frames_gif = []
    w, h = 640, 360

    # Simulate 24 frames of a surfer carving rightwards with virtual camera following
    for i in range(25):
        # Surfer trajectory (carving right)
        sx = 200 + i * 14
        sy = 190 + np.sin(i * 0.25) * 20

        raw_frame = draw_realistic_ocean_frame(w, h, frame_idx=i, surfer_phase="riding", surfer_x=sx, surfer_y=sy)

        # Virtual PTZ crop box centered around surfer with lead-room ahead
        cam_w = 320
        cam_h = 180
        cam_cx = sx + 45  # lead room to the right
        cam_cy = sy

        cx1 = int(np.clip(cam_cx - cam_w // 2, 0, w - cam_w))
        cy1 = int(np.clip(cam_cy - cam_h // 2, 0, h - cam_h))
        cx2 = cx1 + cam_w
        cy2 = cy1 + cam_h

        # Extract cropped frame
        crop = raw_frame[cy1:cy2, cx1:cx2]

        # Draw mini HUD on the cropped view
        cv2.putText(crop, f"REC [LIVE PTZ] 16:9", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(crop, f"RIDE TIME: {i/10.0:.1f}s", (12, cam_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(crop, f"SPEED: {18.0 + np.sin(i*0.2)*4:.1f} km/h", (cam_w - 130, cam_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert to RGB PIL Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames_gif.append(Image.fromarray(rgb))

    out_file = ASSETS_DIR / "virtual_cameraman_ptz_demo.gif"
    frames_gif[0].save(
        out_file,
        save_all=True,
        append_images=frames_gif[1:],
        duration=120,
        loop=0,
    )
    print(f"✅ GIF animé démonstratif généré : {out_file}")


def main():
    print("🎨 Génération des assets visuels haute fidélité pour le README...")
    generate_hero_banner()
    generate_comparison_triptych()
    generate_animated_demo_gif()
    print("🎉 Tous les assets visuels sont prêts dans assets/ !")


if __name__ == "__main__":
    main()

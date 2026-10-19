"""
CLI Runner for Surf-Camera-Vision.
Usage:
    python track_surf.py --input session.mp4 --output-dir outputs/ --aspect 16:9
    python track_surf.py --input session.mp4 --aspect 9:16 --lead-room 0.40
"""

import argparse
import sys
import time
from pathlib import Path

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Surf-Camera-Vision: Autonomous Virtual PTZ Cameraman for Surfers"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Chemin vers la vidéo de surf (.mp4, .mov)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Dossier d'exportation des clips")
    parser.add_argument("--aspect", "-a", type=str, default="16:9", choices=["16:9", "9:16", "1:1"],
                        help="Format de cadrage virtuel (16:9 paysage ou 9:16 vertical TikTok/Reel)")
    parser.add_argument("--zoom-scale", type=float, default=4.5, help="Facteur de zoom virtuel par rapport au surfeur")
    parser.add_argument("--smoothing", type=float, default=0.10, help="Facteur d'amortissement cinématique (0.05=très lent, 0.20=rapide)")
    parser.add_argument("--lead-room", type=float, default=0.35, help="Espace laissé devant le surfeur sur la vague ouverte")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Durée minimale (en secondes) pour valider un ride")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Poids du modèle YOLO (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--no-hud", action="store_true", help="Désactiver l'affichage des métriques incrustées (HUD)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Erreur : Fichier vidéo introuvable : {input_path}")
        sys.exit(1)

    print("=" * 70)
    print("  🏄 Surf-Camera-Vision — Cameraman Virtuel & Tracker de Rides")
    print("=" * 70)
    print(f"• Vidéo d'entrée        : {input_path}")
    print(f"• Format de cadrage     : {args.aspect}")
    print(f"• Facteur Lead Room     : {args.lead_room:.2f}")
    print(f"• Facteur Zoom          : {args.zoom_scale:.1f}x")
    print(f"• Amorti cinématique    : {args.smoothing:.2f}")
    print(f"• Modèle détecteur      : {args.model}")
    print(f"• Dossier de sortie     : {args.output_dir}")
    print("=" * 70)

    config = PipelineConfig(
        model_name=args.model,
        aspect_ratio=args.aspect,
        zoom_scale=args.zoom_scale,
        smoothing_factor=args.smoothing,
        lead_room_factor=args.lead_room,
        min_ride_duration_sec=args.min_duration,
        render_hud=not args.no_hud,
    )

    pipeline = SurfTrackingPipeline(config)

    start_t = time.time()
    def print_progress(ratio: float, msg: str):
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {ratio*100:5.1f}% | {msg}")
        sys.stdout.flush()

    results = pipeline.process_video(
        video_path=str(input_path),
        output_dir=args.output_dir,
        progress_callback=print_progress,
        save_full_diagnostic=True,
        save_ride_clips=True,
    )

    elapsed = time.time() - start_t
    print(f"\n\n🎉 Traitement terminé en {elapsed:.1f}s !")
    print(f"• Frames traitées       : {results['total_frames']}")
    print(f"• Vagues surfées (rides): {results['rides_detected']}")

    if results["diagnostic_video"]:
        print(f"• Vidéo diagnostic      : {results['diagnostic_video']}")

    if len(results["ride_clips"]) > 0:
        print("\n🎬 Clips cinématiques découpés :")
        for i, clip in enumerate(results["ride_clips"], 1):
            ride = results["rides"][i - 1]
            print(f"  [{i}] {clip} ({ride.duration_sec:.1f}s, direction: {ride.direction}, max speed: {ride.max_speed_px:.1f} px/f)")
    else:
        print("\nℹ️ Aucun ride suffisant n'a été détecté (seuil min : {args.min_duration}s).")


if __name__ == "__main__":
    main()

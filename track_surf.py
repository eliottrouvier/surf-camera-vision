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
    parser.add_argument("--min-duration", type=float, default=1.2, help="Durée minimale (en secondes) pour valider un ride")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Poids du modèle YOLO (yolov8n-seg.pt, yolov8n.pt)")
    parser.add_argument("--surfer-height", type=float, default=1.85, help="Hauteur de référence du surfeur en mètres (défaut : 1.85m / 6'1\")")
    parser.add_argument("--board-length", type=float, default=1.83, help="Longueur de référence de la planche en mètres")
    parser.add_argument("--no-hud", action="store_true", help="Désactiver l'affichage des métriques incrustées (HUD)")
    parser.add_argument("--no-seg", action="store_true", help="Désactiver l'affichage des masques de segmentation")
    parser.add_argument("--no-split", action="store_true", help="Désactiver la génération de la vidéo double écran (Split-Screen)")
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
    print(f"• Segmentation instance : {'Désactivée' if args.no_seg else 'Activée'}")
    print(f"• Vidéo Double Écran    : {'Désactivée' if args.no_split else 'Activée (1920x1080 Side-by-Side)'}")
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
        render_segmentation=not args.no_seg,
        consolidate_rides=True,
        save_split_screen=not args.no_split,
        surfer_height_m=args.surfer_height,
        board_length_m=args.board_length,
        enable_wave_metrics=True,
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
        print("\n🎬 Clips cinématiques recadrés :")
        for i, clip in enumerate(results["ride_clips"], 1):
            ride = results["rides"][i - 1]
            print(f"  [{i}] {clip} ({ride.duration_sec:.1f}s, direction: {ride.direction}, max speed: {ride.max_speed_kmh:.1f} km/h, wave: {ride.wave_height_meters:.2f}m [{ride.wave_category}], dist: {ride.distance_meters:.1f}m)")

    if len(results.get("split_clips", [])) > 0:
        print("\n📺 Vidéos Double Écran (Side-by-Side 1080p) :")
        for i, split_clip in enumerate(results["split_clips"], 1):
            ride = results["rides"][i - 1] if i - 1 < len(results["rides"]) else None
            metric_tag = f" — Vague {ride.wave_height_meters:.1f}m | {ride.max_speed_kmh:.1f} km/h" if ride else ""
            print(f"  [{i}] {split_clip}{metric_tag}")
    elif len(results["ride_clips"]) == 0:
        print(f"\nℹ️ Aucun ride suffisant n'a été détecté (seuil min : {args.min_duration}s).")


if __name__ == "__main__":
    main()

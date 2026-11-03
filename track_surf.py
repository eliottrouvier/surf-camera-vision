"""
CLI Runner for Surf-Camera-Vision.
Usage:
    python track_surf.py --input session.mp4
    python track_surf.py --input session.mp4 --aspect 9:16 --surfer-height-cm 185 --board-length-ft-in "5'8\""
"""

import argparse
import sys
import time
from pathlib import Path

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.config import (
    load_config,
    cm_to_meters,
    parse_ft_in,
    format_surfer_height,
    format_wave_dual_units,
)


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Surf-Camera-Vision: Autonomous Virtual PTZ Cameraman for Surfers"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Chemin vers la vidéo de surf (.mp4, .mov)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Dossier d'exportation des clips")
    parser.add_argument("--config", "-c", type=str, default=None, help="Chemin vers un fichier config.yaml alternatif")
    parser.add_argument("--aspect", "-a", type=str, default=cfg["cameraman"]["aspect_ratio"], choices=["16:9", "9:16", "1:1"],
                        help="Format de cadrage virtuel (9:16 vertical TikTok/Reel ou 16:9 paysage)")
    parser.add_argument("--zoom-scale", type=float, default=cfg["cameraman"]["zoom_scale"], help="Facteur de zoom virtuel")
    parser.add_argument("--smoothing", type=float, default=cfg["cameraman"]["smoothing_factor"], help="Amortissement cinématique EMA")
    parser.add_argument("--lead-room", type=float, default=cfg["cameraman"]["lead_room_factor"], help="Anticipation règle des tiers")
    parser.add_argument("--min-duration", type=float, default=cfg["state_machine"]["min_ride_duration_sec"], help="Durée min pour valider un ride (s)")
    parser.add_argument("--model", type=str, default=cfg["detection"]["model_name"], help="Poids YOLO (yolov8n-seg.pt, yolov8n.pt)")

    # Units: favored cm for surfer, ft/in for board
    parser.add_argument("--surfer-height-cm", type=float, default=cfg["surfer"]["height_cm"],
                        help="Taille du surfeur en centimètres (ex: 185)")
    parser.add_argument("--surfer-height", type=float, default=None,
                        help="Taille du surfeur en mètres (ex: 1.85)")
    parser.add_argument("--board-length-ft-in", type=str, default=cfg["surfboard"]["length_ft_in"],
                        help="Longueur de planche en pieds et pouces (ex: 5'8\")")
    parser.add_argument("--board-length", type=float, default=None,
                        help="Longueur de planche en mètres (ex: 1.73)")

    parser.add_argument("--no-hud", action="store_true", help="Désactiver l'affichage des métriques incrustées (HUD)")
    parser.add_argument("--no-seg", action="store_true", help="Désactiver l'affichage des masques de segmentation")
    parser.add_argument("--no-split", action="store_true", help="Désactiver la génération de la vidéo double écran (Split-Screen)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Erreur : Fichier vidéo introuvable : {input_path}")
        sys.exit(1)

    # Resolve surfer height (favoring cm)
    if args.surfer_height is not None:
        surfer_h_m = args.surfer_height
        surfer_h_cm = surfer_h_m * 100.0
    else:
        surfer_h_cm = args.surfer_height_cm
        surfer_h_m = cm_to_meters(surfer_h_cm)

    # Resolve board length (favoring ft/in)
    if args.board_length is not None:
        board_l_m = args.board_length
        board_str = f"{board_l_m:.2f} m"
    else:
        board_str = args.board_length_ft_in
        board_l_m = parse_ft_in(board_str)

    print("=" * 75)
    print("  🏄 Surf-Camera-Vision — Cameraman Virtuel & Télémétrie de Rides")
    print("=" * 75)
    print(f"• Vidéo d'entrée        : {input_path}")
    print(f"• Format de cadrage     : {args.aspect}")
    print(f"• Facteur Lead Room     : {args.lead_room:.2f}")
    print(f"• Facteur Zoom          : {args.zoom_scale:.1f}x")
    print(f"• Amorti cinématique    : {args.smoothing:.2f}")
    print(f"• Taille Surfeur (ref)  : {format_surfer_height(surfer_h_cm)}")
    print(f"• Taille Planche (ref)  : {board_str} ({board_l_m:.2f} m)")
    print(f"• Modèle détecteur      : {args.model}")
    print(f"• Télémétrie & Masques  : {'Désactivés' if args.no_hud else 'Activés'}")
    print(f"• Vidéo Double Écran    : {'Désactivée' if args.no_split else 'Activée (1920x1080 Side-by-Side)'}")
    print(f"• Dossier d'export      : {args.output_dir}")
    print("=" * 75)

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
        surfer_height_m=surfer_h_m,
        board_length_m=board_l_m,
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
            wave_text = format_wave_dual_units(ride.wave_height_meters)
            print(f"  [{i}] {clip}")
            print(f"      ⚡ Vitesse Pic : {ride.max_speed_kmh:.1f} km/h (Moyenne : {ride.avg_speed_kmh:.1f} km/h)")
            print(f"      🌊 Taille Vague: {wave_text} [{ride.wave_category}]")
            print(f"      ⏱️ Durée       : {ride.duration_sec:.1f}s ({ride.direction}) | Glisse : {ride.distance_meters:.1f}m")

    if len(results.get("split_clips", [])) > 0:
        print("\n📺 Vidéos Double Écran (Side-by-Side 1080p) :")
        for i, split_clip in enumerate(results["split_clips"], 1):
            ride = results["rides"][i - 1] if i - 1 < len(results["rides"]) else None
            metric_tag = f" — Vitesse max: {ride.max_speed_kmh:.1f} km/h | Vague: {ride.wave_height_meters:.2f}m ({ride.wave_height_feet:.1f}ft)" if ride else ""
            print(f"  [{i}] {split_clip}{metric_tag}")
    elif len(results["ride_clips"]) == 0:
        print(f"\nℹ️ Aucun ride suffisant n'a été détecté (seuil min : {args.min_duration}s).")


if __name__ == "__main__":
    main()

"""
One-click Demo Script for Surf-Camera-Vision.
Generates/downloads a sample surf session, executes the full pipeline,
and outputs both 16:9 cinematic landscape and 9:16 vertical TikTok/Reel clips.

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.utils import download_or_generate_sample_video


def run_demo():
    print("=" * 75)
    print("  🏄 SURF-CAMERA-VISION — DÉMONSTRATION COMPLÈTE EN UN CLIC")
    print("=" * 75)

    data_dir = Path("data")
    outputs_dir = Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sample_video = data_dir / "sample_surf_session.mp4"

    # Step 1: Ensure sample video exists
    print("\n[1/3] Vérification / Génération de la séquence vidéo de surf de test...")
    download_or_generate_sample_video(sample_video, num_frames=210, fps=30)

    # Step 2: Run 16:9 Landscape Cinematic Auto-Cameraman
    print("\n[2/3] Exécution du pipeline en format Paysage Cinématique (16:9)...")
    config_16_9 = PipelineConfig(
        model_name="yolov8n.pt",
        aspect_ratio="16:9",
        zoom_scale=4.5,
        smoothing_factor=0.10,
        lead_room_factor=0.35,
        min_ride_duration_sec=1.5,
        render_hud=True,
    )
    pipeline_16_9 = SurfTrackingPipeline(config_16_9)

    start_t = time.time()
    results_16_9 = pipeline_16_9.process_video(
        video_path=str(sample_video),
        output_dir=str(outputs_dir),
        save_full_diagnostic=True,
        save_ride_clips=True,
    )
    elapsed_16_9 = time.time() - start_t
    print(f"  ✅ Traitement 16:9 terminé en {elapsed_16_9:.1f}s !")
    print(f"  👉 Vagues détectées : {results_16_9['rides_detected']}")
    for clip in results_16_9["ride_clips"]:
        print(f"     🎥 Clip 16:9 sauvegardé : {clip}")

    # Step 3: Run 9:16 Vertical Reel/TikTok Auto-Cameraman
    print("\n[3/3] Exécution du pipeline en format Vertical Réseaux Sociaux (9:16)...")
    config_9_16 = PipelineConfig(
        model_name="yolov8n.pt",
        aspect_ratio="9:16",
        zoom_scale=4.0,
        smoothing_factor=0.10,
        lead_room_factor=0.30,
        min_ride_duration_sec=1.5,
        render_hud=True,
    )
    pipeline_9_16 = SurfTrackingPipeline(config_9_16)

    start_t = time.time()
    results_9_16 = pipeline_9_16.process_video(
        video_path=str(sample_video),
        output_dir=str(outputs_dir),
        save_full_diagnostic=False,
        save_ride_clips=True,
    )
    elapsed_9_16 = time.time() - start_t
    print(f"  ✅ Traitement 9:16 terminé en {elapsed_9_16:.1f}s !")
    for clip in results_9_16["ride_clips"]:
        print(f"     📱 Clip 9:16 sauvegardé : {clip}")
    for split_clip in results_9_16.get("split_clips", []):
        print(f"     📺 Split-Screen 1080p sauvegardé : {split_clip}")

    print("\n" + "=" * 75)
    print("🎉 DÉMO RÉUSSIE AVEC SUCCÈS !")
    print("=" * 75)
    print(f"📁 Tous les rendus sont disponibles dans le dossier : {outputs_dir.resolve()}/")
    print("🌐 Pour lancer l'application web interactive : streamlit run app.py")


if __name__ == "__main__":
    run_demo()

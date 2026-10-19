"""
Surf-Camera-Vision: Interactive Streamlit Web Application.
Upload or load surf videos, configure virtual cameraman framing,
and review/export automated cinematic ride cuts.
"""

import os
import tempfile
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.utils import download_or_generate_sample_video, STATE_COLORS
from surf_camera.state_machine import SurferState


st.set_page_config(
    page_title="Surf-Camera-Vision — Auto Cameraman",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern surf aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8892b0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #111927;
        border: 1px solid #1f293d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏄 Surf-Camera-Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cameraman Virtuel Intelligent (Digital PTZ) & Tracker de Vagues Automatisé</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Paramètres du Cameraman")

aspect_option = st.sidebar.selectbox(
    "Format de cadrage virtuel",
    options=["16:9", "9:16", "1:1"],
    index=0,
    help="16:9 = Format paysage standard (YouTube) | 9:16 = Format vertical optimisé pour Instagram Reels et TikTok"
)

lead_room = st.sidebar.slider(
    "Espace devant le surfeur (Lead Room)",
    min_value=0.10,
    max_value=0.50,
    value=0.35,
    step=0.05,
    help="Décale le cadrage dans le sens de la vague pour placer le surfeur sur le tiers arrière et filmer la face ouverte."
)

zoom_scale = st.sidebar.slider(
    "Facteur de zoom virtuel",
    min_value=2.5,
    max_value=7.0,
    value=4.5,
    step=0.5,
    help="Échelle du zoom par rapport à la taille du surfeur."
)

smoothing = st.sidebar.slider(
    "Amorti cinématique (Fluid Head)",
    min_value=0.05,
    max_value=0.25,
    value=0.10,
    step=0.01,
    help="Plus la valeur est basse, plus le mouvement de la caméra virtuelle est doux et fluide (élimine les secousses)."
)

st.sidebar.header("🎯 Sensibilité de Détection")
conf_thresh = st.sidebar.slider("Seuil de confiance YOLO", 0.15, 0.60, 0.25, 0.05)
min_duration = st.sidebar.slider("Durée minimale d'un ride (s)", 1.0, 5.0, 1.5, 0.5)

# --- VIDEO SOURCE SELECTION ---
st.subheader("📹 1. Sélection de la Vidéo de Session")
col_src1, col_src2 = st.columns([1, 1])

with col_src1:
    uploaded_file = st.file_uploader(
        "Glisser-déposer une vidéo de surf (MP4, MOV)",
        type=["mp4", "mov", "avi"],
    )

with col_src2:
    use_demo = st.button("🌊 Charger la Session Démo Haute Définition", use_container_width=True)

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
demo_video_path = data_dir / "sample_surf_session.mp4"

video_target_path = None

if uploaded_file is not None:
    temp_path = data_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_target_path = temp_path
    st.success(f"Vidéo chargée : {uploaded_file.name}")
elif use_demo or demo_video_path.exists():
    download_or_generate_sample_video(demo_video_path)
    video_target_path = demo_video_path
    st.info(f"Session démo prête : {demo_video_path.name}")

if video_target_path:
    st.video(str(video_target_path))

    # --- EXECUTION BUTTON ---
    st.subheader("🚀 2. Lancement du Tracking & Cadrage")
    if st.button("🎬 Lancer le Cameraman Virtuel & Découper les Vagues", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        config = PipelineConfig(
            model_name="yolov8n.pt",
            conf_threshold=conf_thresh,
            aspect_ratio=aspect_option,
            zoom_scale=zoom_scale,
            smoothing_factor=smoothing,
            lead_room_factor=lead_room,
            min_ride_duration_sec=min_duration,
            render_hud=True,
        )

        pipeline = SurfTrackingPipeline(config)

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(ratio)
            status_text.text(msg)

        start_time = time.time()
        results = pipeline.process_video(
            video_path=str(video_target_path),
            output_dir="outputs",
            progress_callback=update_progress,
            save_full_diagnostic=True,
            save_ride_clips=True,
        )
        elapsed = time.time() - start_time
        st.session_state["results"] = results
        st.session_state["elapsed"] = elapsed

# --- RESULTS DISPLAY ---
if "results" in st.session_state:
    results = st.session_state["results"]
    elapsed = st.session_state["elapsed"]
    rides = results["rides"]

    st.markdown("---")
    st.subheader("📊 3. Résultats & Télémétrie de la Session")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Vagues Détectées (Rides)", len(rides))
    with m2:
        total_surf_time = sum(r.duration_sec for r in rides)
        st.metric("Temps Total Surfé", f"{total_surf_time:.1f} s")
    with m3:
        max_speed = max([r.max_speed_px for r in rides], default=0.0)
        st.metric("Vitesse Max", f"{max_speed:.1f} px/f")
    with m4:
        st.metric("Temps de Calcul", f"{elapsed:.1f} s")

    # Tabs for Outputs
    tab1, tab2, tab3 = st.tabs([
        "🎬 Clips Cinématiques Découpés",
        "🌐 Vue d'Ensemble & Diagnostic PTZ",
        "📈 Analyse des Trajectoires & Vitesse"
    ])

    with tab1:
        if len(results["ride_clips"]) == 0:
            st.warning("Aucun ride de durée suffisante n'a été détecté dans cet extrait.")
        else:
            st.markdown(f"**{len(results['ride_clips'])} clip(s) vidéo généré(s)** avec cadrage automatique et Lead Room :")
            for idx, clip_path in enumerate(results["ride_clips"]):
                ride = rides[idx]
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    st.video(clip_path)
                with col_c2:
                    st.markdown(f"### Vague #{ride.ride_id}")
                    st.markdown(f"- **Direction :** `{ride.direction}` (droite/gauche)")
                    st.markdown(f"- **Durée du ride :** `{ride.duration_sec:.1f}s`")
                    st.markdown(f"- **Vitesse max :** `{ride.max_speed_px:.1f} px/f`")
                    st.markdown(f"- **Take-off :** Frame `{ride.takeoff_frame}`")
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Télécharger le Clip #{ride.ride_id} ({aspect_option})",
                            data=f,
                            file_name=Path(clip_path).name,
                            mime="video/mp4",
                            key=f"dl_{ride.ride_id}",
                        )

    with tab2:
        if results["diagnostic_video"] and Path(results["diagnostic_video"]).exists():
            st.markdown("Vue panoramique avec boîte de cadrage virtuel (magenta) et états du surfeur :")
            st.video(results["diagnostic_video"])

    with tab3:
        if len(rides) > 0:
            st.markdown("#### Profil de Vitesse & Déferlement")
            fig, ax = plt.subplots(figsize=(10, 4))
            for r in rides:
                frames = [p[2] for p in r.trajectory]
                # Recompute instantaneous displacement
                diffs = [np.sqrt((r.trajectory[i][0] - r.trajectory[i-1][0])**2 + (r.trajectory[i][1] - r.trajectory[i-1][1])**2) for i in range(1, len(r.trajectory))]
                diffs = [diffs[0]] + diffs if diffs else [0]
                ax.plot(frames, diffs, label=f"Ride #{r.ride_id} ({r.direction})", lw=2)

            ax.set_title("Vitesse instantanée du surfeur au fil de la vague", fontsize=11, fontweight="bold")
            ax.set_xlabel("Numéro de Frame")
            ax.set_ylabel("Déplacement (pixels / frame)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close()

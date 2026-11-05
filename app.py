"""
Surf-Camera-Vision: Studio Web Épuré.
Cameraman virtuel autonome (Digital PTZ), détection automatique de take-off,
qualification du déferlement et télémétrie physique de glisse.
"""

from pathlib import Path
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.config import (
    resolve_model_path,
    format_ft_in,
    format_wave_dual_units,
    meters_to_feet,
)


st.set_page_config(
    page_title="Surf-Camera-Vision",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS épuré et moderne (Studio Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #090c10;
        color: #e6edf3;
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
        padding-top: 4px;
    }
    .brand-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #58a6ff;
    }
    .brand-subtitle {
        color: #8b949e;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
    }

    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚡ Chargement accéléré du modèle IA...")
def get_cached_pipeline(
    model_type: str,
    aspect_ratio: str,
    surfer_height_m: float,
    min_ride_duration_sec: float,
) -> SurfTrackingPipeline:
    """Maintient les modèles pré-chargés en mémoire vive unifiée."""
    cfg = PipelineConfig(
        model_name=resolve_model_path(model_type),
        model_type=model_type,
        conf_threshold=0.20,
        aspect_ratio=aspect_ratio,
        surfer_height_m=surfer_height_m,
        min_ride_duration_sec=min_ride_duration_sec,
        enable_pose_estimation=False,
        save_split_screen=False,
        fast_scan=True,
    )
    return SurfTrackingPipeline(cfg)


# Barre latérale de réglages épurée
st.sidebar.markdown("### ⚙️ Réglages Cameraman")

# 1. Sélecteur de modèle
model_options = [
    "Modèle Dédié Surf Marin (yolov8n_surf_v2_best.pt)",
    "Modèle Standard COCO (yolov8n.pt)",
]
model_choice = st.sidebar.selectbox(
    "Modèle de Détection IA",
    model_options,
    index=0,
    help="Modèle dédié fine-tuné sur conditions réelles marines vs modèle standard COCO.",
)
model_type = "surf_v2" if "Surf Marin" in model_choice else "coco"

# 2. Sélecteur de format
format_options = [
    "📱 9:16 Vertical Mobile (Reels / TikTok)",
    "🖥️ 16:9 Cinématique",
]
format_choice = st.sidebar.radio(
    "Format Cadrage Virtuel",
    format_options,
    index=0,
    help="Format de recadrage automatique généré par l'Auto-Cameraman.",
)
aspect_ratio = "9:16" if "9:16" in format_choice else "16:9"

# 3. Slider morphologique : Taille du surfeur (1.50 m à 2.05 m, défaut 1.75 m)
surfer_height_m = st.sidebar.slider(
    "Taille du surfeur (m)",
    min_value=1.50,
    max_value=2.05,
    value=1.75,
    step=0.01,
    help="Taille réelle du surfeur servant à étalonner l'échelle métrique absolue.",
)
st.sidebar.caption(f"📏 Calibrage : **{surfer_height_m:.2f} m** ({format_ft_in(surfer_height_m)})")

# 4. Durée minimale de vague (slider 1.5s à 5.0s, défaut 1.5s)
min_ride_duration_sec = st.sidebar.slider(
    "Durée minimale de vague (s)",
    min_value=1.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Durée active minimale requise pour valider un ride et exporter le clip recadré.",
)

# En-tête de la page principale
st.markdown("""
<div class="brand-container">
    <span class="brand-title">🏄 Surf-Camera-Vision</span>
    <span class="brand-badge">STUDIO • AUTO-CAMERAMAN PTZ</span>
</div>
<div class="brand-subtitle">
    Cameraman virtuel intelligent, détection automatique de take-off & métriques physiques de vague
</div>
""", unsafe_allow_html=True)

# Zone principale : Dépôt vidéo & Démonstration
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
demo_video_path = data_dir / "sample_surf_session.mp4"

st.markdown("#### 🌊 Source Vidéo")
col_upload, col_demo = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Déposer une vidéo de surf (MP4, MOV)",
        type=["mp4", "mov"],
        help="Glissez et déposez votre vidéo de surf pour lancer le traitement autonome.",
    )

with col_demo:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    use_demo = st.button("🌊 Charger la Démo (1080p)", use_container_width=True)

video_target_path = None

if uploaded_file is not None:
    temp_path = data_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_target_path = temp_path
    st.caption(f"Fichier actif : `{uploaded_file.name}`")
elif use_demo:
    if not demo_video_path.exists():
        from surf_camera.utils import download_or_generate_sample_video
        download_or_generate_sample_video(demo_video_path)
    video_target_path = demo_video_path
    st.caption(f"Fichier actif : `{demo_video_path.name}` (Séquence de référence)")

if video_target_path is not None:
    col_vid, col_action = st.columns([2, 1])
    with col_vid:
        st.video(str(video_target_path))

    with col_action:
        st.markdown("#### ⚡ Prêt pour le Cadrage")
        st.caption(
            f"Modèle : **{model_type}** • Format : **{aspect_ratio}** • "
            f"Surfeur : **{surfer_height_m:.2f} m** • Min : **{min_ride_duration_sec:.1f} s**"
        )
        launch_btn = st.button("🏄 Lancer l'Auto-Cameraman", type="primary", use_container_width=True)

    if launch_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(ratio)
            status_text.caption(f"⚡ {msg}")

        pipeline = get_cached_pipeline(
            model_type=model_type,
            aspect_ratio=aspect_ratio,
            surfer_height_m=surfer_height_m,
            min_ride_duration_sec=min_ride_duration_sec,
        )
        pipeline.config.aspect_ratio = aspect_ratio
        pipeline.config.surfer_height_m = surfer_height_m
        pipeline.config.min_ride_duration_sec = min_ride_duration_sec

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
        st.session_state["active_aspect"] = aspect_ratio

        progress_bar.empty()
        status_text.empty()
        st.success(f"Analyse terminée en {elapsed:.1f}s ! {len(results['rides'])} vague(s) validée(s).")


# Affichage des résultats
if "results" in st.session_state:
    results = st.session_state["results"]
    rides = results.get("rides", [])
    active_aspect = st.session_state.get("active_aspect", aspect_ratio)

    st.divider()
    st.markdown("### 📊 Synthèse de la Session")

    # Métriques globales
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Vagues Validées", len(rides))
    with m2:
        max_speed = max([r.max_speed_kmh for r in rides], default=0.0)
        st.metric("Vitesse Pic Max", f"{max_speed:.1f} km/h")
    with m3:
        avg_speeds = [r.avg_speed_kmh for r in rides if r.avg_speed_kmh > 0]
        avg_speed = float(np.mean(avg_speeds)) if avg_speeds else 0.0
        st.metric("Vitesse Moyenne", f"{avg_speed:.1f} km/h")
    with m4:
        max_wave_h = max([r.wave_height_meters for r in rides], default=0.0)
        st.metric("Vague Maximale", f"{max_wave_h:.2f} m", help=f"{meters_to_feet(max_wave_h):.1f} ft")
    with m5:
        total_dist = sum(r.distance_meters for r in rides)
        st.metric("Distance Totale", f"{total_dist:.0f} m")
    with m6:
        total_time = sum(r.duration_sec for r in rides)
        st.metric("Temps Cumulé", f"{total_time:.1f} s")

    st.markdown("### 🎬 Clips Recadrés & Métriques Physiques")
    if len(results.get("ride_clips", [])) == 0:
        st.info(f"Aucune vague active d'au moins {min_ride_duration_sec:.1f}s n'a été détectée.")
    else:
        for idx, clip_path in enumerate(results["ride_clips"]):
            ride = rides[idx]
            wave_text = format_wave_dual_units(ride.wave_height_meters)

            st.markdown(f"#### 🏄 Vague #{ride.ride_id} — {ride.wave_behavior} ({ride.direction})")
            col_clip, col_meta = st.columns([3, 2])
            with col_clip:
                st.video(clip_path)
            with col_meta:
                st.markdown(f"""
                - **⚡ Vitesse Pic :** `{ride.max_speed_kmh:.1f} km/h`
                - **🏄 Vitesse Moyenne :** `{ride.avg_speed_kmh:.1f} km/h`
                - **🌊 Hauteur de Vague :** `{wave_text}`
                - **🏷️ Catégorie :** `{ride.wave_category}`
                - **🧭 Comportement :** `{ride.wave_behavior}`
                - **⏱️ Durée réelle :** `{ride.duration_sec:.1f} s`
                - **📏 Distance de glisse :** `{ride.distance_meters:.1f} m`
                """)
                with open(clip_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Télécharger le Clip #{ride.ride_id} ({active_aspect})",
                        data=f,
                        file_name=Path(clip_path).name,
                        mime="video/mp4",
                        key=f"dl_clip_{ride.ride_id}",
                        use_container_width=True,
                    )

    # Graphique de télémétrie de glisse
    if len(rides) > 0:
        st.markdown("### 📈 Télémétrie Cinématique de Glisse")
        fps = results.get("fps", 30.0)
        fig, (ax_spd, ax_wave) = plt.subplots(2, 1, figsize=(10, 6.2), facecolor="#090c10")
        fig.subplots_adjust(hspace=0.38)

        # Graphique 1 : Vitesse en km/h
        ax_spd.set_facecolor("#0d1117")
        for r in rides:
            if r.speed_series_kmh:
                t_axis = np.arange(len(r.speed_series_kmh)) / fps
                ax_spd.plot(
                    t_axis,
                    r.speed_series_kmh,
                    label=f"Vague #{r.ride_id} ({r.direction} • Pic: {r.max_speed_kmh:.1f} km/h)",
                    lw=2.2,
                )
        ax_spd.set_title("Profil de Vitesse du Surfeur (km/h)", fontsize=11, fontweight="600", color="#ffffff")
        ax_spd.set_xlabel("Temps de glisse (secondes)", color="#8b949e", fontsize=9)
        ax_spd.set_ylabel("Vitesse (km/h)", color="#8b949e", fontsize=9)
        ax_spd.tick_params(colors="#8b949e")
        for spine in ax_spd.spines.values():
            spine.set_color("#21262d")
        ax_spd.grid(True, alpha=0.15, color="#8b949e")
        ax_spd.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#ffffff", loc="upper left")

        # Graphique 2 : Hauteur de vague
        ax_wave.set_facecolor("#0d1117")
        for r in rides:
            series = getattr(r, "wave_height_series_m", [])
            if series:
                t_axis = np.arange(len(series)) / fps
                w_lbl = f"Vague #{r.ride_id} ({r.wave_height_meters:.2f} m / {r.wave_height_feet:.1f} ft • {r.wave_category})"
                ax_wave.plot(t_axis, series, label=w_lbl, color="#00e5ff", lw=2.0)
        ax_wave.set_title("Hauteur de Face de Vague (mètres)", fontsize=11, fontweight="600", color="#ffffff")
        ax_wave.set_xlabel("Temps de glisse (secondes)", color="#8b949e", fontsize=9)
        ax_wave.set_ylabel("Hauteur (m)", color="#8b949e", fontsize=9)
        ax_wave.tick_params(colors="#8b949e")
        for spine in ax_wave.spines.values():
            spine.set_color("#21262d")
        ax_wave.grid(True, alpha=0.15, color="#8b949e")
        ax_wave.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#ffffff", loc="upper left")

        st.pyplot(fig)
        plt.close(fig)

    # Vue panoramique de contrôle (overview large avec boîte de cadrage PTZ) dans un expander pliable
    with st.expander("🌐 Vue panoramique de contrôle (Overview PTZ)", expanded=False):
        diag_path = results.get("diagnostic_video")
        if diag_path and Path(diag_path).exists():
            st.caption("Plan large original annoté montrant la boîte de cadrage PTZ dynamique, le suivi et la segmentation de la vague.")
            st.video(diag_path)
        else:
            st.info("Aucune vidéo panoramique de contrôle disponible pour cette session.")

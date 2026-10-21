"""
Surf-Camera-Vision: Minimalist Studio Web Application.
Upload or load surf videos, configure virtual cameraman framing,
and review/export automated cinematic ride cuts with instance segmentation.
"""

from pathlib import Path
import time
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.utils import download_or_generate_sample_video, STATE_COLORS
from surf_camera.state_machine import SurferState


st.set_page_config(
    page_title="Surf-Camera-Vision",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MINIMALIST STUDIO CSS (Linear / Apple Dark Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #090c10;
        color: #e6edf3;
    }
    
    /* Header minimalist */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
        padding-top: 4px;
    }
    .brand-title {
        font-size: 1.6rem;
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
        margin-bottom: 1.8rem;
    }
    
    /* Clean Cards */
    .studio-card {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    /* Metric Card */
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
        font-size: 1.4rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="brand-container">
    <span class="brand-title">🏄 Surf-Camera-Vision</span>
    <span class="brand-badge">STUDIO v1.2</span>
</div>
<div class="brand-subtitle">Cameraman virtuel intelligent (Digital PTZ) & segmentation instance de vagues</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ⚙️ Paramètres Studio")

with st.sidebar.expander("🎥 Cadrage & Composition", expanded=True):
    aspect_option = st.selectbox(
        "Format de cadrage",
        options=["9:16", "16:9", "1:1"],
        index=0,
        help="9:16 (Instagram Reels / TikTok) | 16:9 (YouTube) | 1:1 (Carré)"
    )
    zoom_scale = st.slider(
        "Facteur de zoom",
        min_value=2.0,
        max_value=6.5,
        value=4.0,
        step=0.5,
        help="Échelle du zoom virtuel par rapport à la taille du surfeur."
    )
    lead_room = st.slider(
        "Anticipation Lead-Room",
        min_value=0.10,
        max_value=0.50,
        value=0.35,
        step=0.05,
        help="Décalage cinématique dans le sens du déferlement (règle des tiers)."
    )
    smoothing = st.slider(
        "Amorti cinématique (EMA)",
        min_value=0.04,
        max_value=0.25,
        value=0.10,
        step=0.01,
        help="Inertie de la rotule fluide. Valeur basse = suivi ultra-doux sans tremblement."
    )
    enable_split_screen = st.toggle("Générer la vidéo Double Écran (Side-by-Side 1080p)", value=True)

with st.sidebar.expander("🎯 Détection & Segmentation", expanded=True):
    model_choice = st.selectbox(
        "Modèle YOLO",
        options=["yolov8n-seg.pt (Instance Segmentation)", "yolov8n.pt (Boîtes seules)"],
        index=0,
    )
    model_name = "yolov8n-seg.pt" if "seg" in model_choice else "yolov8n.pt"
    
    enable_segmentation = st.toggle("Masques de segmentation polygonaux", value=True)
    enable_hud = st.toggle("Afficher la télémétrie incrustée (HUD)", value=True)
    conf_thresh = st.slider("Seuil de confiance", 0.15, 0.60, 0.20, 0.05)

with st.sidebar.expander("⚡ Consolidation & Anti-Fragmentation", expanded=True):
    enable_consolidation = st.toggle("Fusionner les segments de même vague", value=True)
    min_active_duration = st.slider("Durée min de glisse réelle (s)", 0.8, 3.5, 1.2, 0.2)
    max_merge_gap = st.slider("Tolérance coupure spray / écume (s)", 1.0, 3.5, 2.0, 0.5)
    speed_thresh = st.slider("Seuil vitesse take-off (px/f)", 2.0, 6.0, 3.5, 0.5)

with st.sidebar.expander("🌊 Paramètres Vagues & Échelle Métrique", expanded=True):
    surfer_ref_h = st.slider("Taille du surfeur de référence (m)", 1.50, 2.05, 1.85, 0.05, help="Hauteur de référence (6'1\" = 1.85m) utilisée par le VAE pour calibrer les mètres par pixel.")
    board_ref_l = st.slider("Longueur planche de référence (m)", 1.50, 2.70, 1.83, 0.05, help="Longueur standard de la planche (ex: 6'0\" = 1.83m).")
    enable_wave_metrics = st.toggle("Activer segmentation & calculs de vague", value=True)


# --- WORKSPACE DIRECTORIES ---
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
demo_video_path = data_dir / "sample_surf_session.mp4"

# --- STEP 1: VIDEO SOURCE ---
st.markdown("#### 1. Vidéo de Session")
col_upload, col_demo = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Déposer une vidéo de surf (MP4, MOV, AVI, MKV, WebM)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed"
    )

with col_demo:
    use_demo = st.button("🌊 Charger la Session Démo (1080p)", use_container_width=True)

video_target_path = None

if uploaded_file is not None:
    temp_path = data_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_target_path = temp_path
    st.caption(f"Fichier actif : `{uploaded_file.name}`")
elif use_demo or demo_video_path.exists():
    download_or_generate_sample_video(demo_video_path)
    video_target_path = demo_video_path
    st.caption(f"Fichier actif : `{demo_video_path.name}` (Séquence réelle 1080p)")

if video_target_path:
    col_vid, col_action = st.columns([2, 1])
    with col_vid:
        st.video(str(video_target_path))
    
    with col_action:
        st.markdown("#### 2. Traitement")
        st.caption("Exécute la détection, la segmentation, la consolidation anti-fragmentation et le cadrage PTZ.")
        launch_btn = st.button("🎬 Lancer le Cadrage & Découpage", type="primary", use_container_width=True)

    if launch_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        config = PipelineConfig(
            model_name=model_name,
            conf_threshold=conf_thresh,
            aspect_ratio=aspect_option,
            zoom_scale=zoom_scale,
            smoothing_factor=smoothing,
            lead_room_factor=lead_room,
            min_ride_duration_sec=min_active_duration,
            speed_takeoff_thresh=speed_thresh,
            render_hud=enable_hud,
            render_segmentation=enable_segmentation,
            consolidate_rides=enable_consolidation,
            max_merge_gap_sec=max_merge_gap,
            save_split_screen=enable_split_screen,
            surfer_height_m=surfer_ref_h,
            board_length_m=board_ref_l,
            enable_wave_metrics=enable_wave_metrics,
        )

        pipeline = SurfTrackingPipeline(config)

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(ratio)
            status_text.caption(f"⚡ {msg}")

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
        st.session_state["target_aspect"] = aspect_option


# --- STEP 3: RESULTS & CLIPS ---
if "results" in st.session_state:
    results = st.session_state["results"]
    elapsed = st.session_state["elapsed"]
    aspect_selected = st.session_state.get("target_aspect", "16:9")
    rides = results["rides"]

    st.markdown("---")
    st.markdown("#### 3. Résultats de Session")

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Vagues Validées", len(rides))
    with m2:
        max_speed = max([r.max_speed_kmh for r in rides], default=0.0)
        st.metric("Vitesse Pic", f"{max_speed:.1f} km/h")
    with m3:
        avg_speeds = [r.avg_speed_kmh for r in rides if r.avg_speed_kmh > 0]
        avg_speed = float(np.mean(avg_speeds)) if avg_speeds else 0.0
        st.metric("Vitesse Moyenne", f"{avg_speed:.1f} km/h")
    with m4:
        max_wave_h = max([r.wave_height_meters for r in rides], default=0.0)
        st.metric("Vague Max", f"{max_wave_h:.2f} m", help=f"{max_wave_h*3.28084:.1f} ft")
    with m5:
        total_dist = sum(r.distance_meters for r in rides)
        st.metric("Distance Totale", f"{total_dist:.0f} m")
    with m6:
        total_time = sum(r.duration_sec for r in rides)
        st.metric("Temps Cumulé", f"{total_time:.1f} s")

    # Tabs for Outputs
    tab_split, tab_clips, tab_diag, tab_telemetry = st.tabs([
        "📺 Studio Double Écran (Original vs Recadré)",
        "✂️ Clips Recadrés",
        "🌐 Vue Panoramique & Diagnostic",
        "📈 Analyse Cinématique & Vagues"
    ])

    with tab_split:
        split_clips = results.get("split_clips", [])
        if len(split_clips) == 0:
            st.info("Aucune vidéo double écran disponible (activez l'option dans le volet latéral ou vérifiez qu'une vague a été détectée).")
        else:
            for idx, split_path in enumerate(split_clips):
                ride = rides[idx] if idx < len(rides) else None
                wave_info = f" • Vague {ride.wave_height_meters:.1f}m ({ride.wave_category})" if (ride and ride.wave_height_meters > 0) else ""
                ride_name = f"Vague #{ride.ride_id} — {ride.duration_sec:.1f}s ({ride.direction}){wave_info}" if ride else f"Ride #{idx+1}"
                st.markdown(f"##### 🌊 {ride_name}")
                st.caption(f"Comparatif synchronisé 1080p : Plan large fixe plage avec carré de cadrage PTZ vs Découpage dynamique ({aspect_selected}).")
                st.video(split_path)

                col_btn_split, col_btn_crop = st.columns(2)
                with col_btn_split:
                    with open(split_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Télécharger Double Écran (1920x1080)",
                            data=f,
                            file_name=Path(split_path).name,
                            mime="video/mp4",
                            key=f"dl_split_{idx}",
                            use_container_width=True,
                        )
                with col_btn_crop:
                    if idx < len(results.get("ride_clips", [])):
                        crop_path = results["ride_clips"][idx]
                        with open(crop_path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Télécharger Clip Seul ({aspect_selected})",
                                data=f,
                                file_name=Path(crop_path).name,
                                mime="video/mp4",
                                key=f"dl_crop_split_{idx}",
                                use_container_width=True,
                            )
                st.markdown("---")

    with tab_clips:
        if len(results["ride_clips"]) == 0:
            st.info("Aucune vague continue d'au moins " + str(min_active_duration) + "s n'a été détectée.")
        else:
            for idx, clip_path in enumerate(results["ride_clips"]):
                ride = rides[idx]
                st.markdown(f"##### Vague #{ride.ride_id} — {ride.duration_sec:.1f}s ({ride.direction})")
                c_vid, c_info = st.columns([2, 1])
                with c_vid:
                    st.video(clip_path)
                with c_info:
                    st.markdown(f"""
                    - **Vitesse pic :** `{ride.max_speed_kmh:.1f} km/h`
                    - **Vitesse moyenne :** `{ride.avg_speed_kmh:.1f} km/h`
                    - **Hauteur de vague :** `{ride.wave_height_meters:.2f} m ({ride.wave_height_feet:.1f} ft)`
                    - **Catégorie :** `{ride.wave_category}`
                    - **Distance de glisse :** `{ride.distance_meters:.1f} m`
                    - **Orientation :** `{ride.direction}`
                    - **Départ pop-up :** `Frame #{ride.takeoff_frame}`
                    """)
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Télécharger le Clip #{ride.ride_id} ({aspect_selected})",
                            data=f,
                            file_name=Path(clip_path).name,
                            mime="video/mp4",
                            key=f"dl_{ride.ride_id}",
                            use_container_width=True,
                        )

    with tab_diag:
        if results["diagnostic_video"] and Path(results["diagnostic_video"]).exists():
            st.caption("Flux complet avec masques de segmentation, masse d'eau polygonale et fenêtre de cadrage PTZ.")
            st.video(results["diagnostic_video"])

    with tab_telemetry:
        if len(rides) > 0:
            fig, (ax_spd, ax_wave) = plt.subplots(2, 1, figsize=(10, 6.0), facecolor="#090c10")
            fig.subplots_adjust(hspace=0.35)

            fps = results.get("fps", 30.0)

            # Subplot 1: Speed in km/h
            ax_spd.set_facecolor("#0d1117")
            for r in rides:
                if r.speed_series_kmh:
                    time_axis = np.arange(len(r.speed_series_kmh)) / fps
                    ax_spd.plot(time_axis, r.speed_series_kmh, label=f"Vague #{r.ride_id} ({r.direction} • Pic {r.max_speed_kmh:.1f} km/h)", lw=2.2)
            ax_spd.set_title("Profil de Vitesse Réelle du Surfeur (km/h)", fontsize=11, fontweight="600", color="#ffffff")
            ax_spd.set_xlabel("Temps de glisse (secondes)", color="#8b949e", fontsize=9)
            ax_spd.set_ylabel("Vitesse (km/h)", color="#8b949e", fontsize=9)
            ax_spd.tick_params(colors="#8b949e")
            for spine in ax_spd.spines.values():
                spine.set_color("#21262d")
            ax_spd.grid(True, alpha=0.15, color="#8b949e")
            ax_spd.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#ffffff", loc="upper left")

            # Subplot 2: Wave Height in meters
            ax_wave.set_facecolor("#0d1117")
            for r in rides:
                if r.wave_height_series_m:
                    time_axis = np.arange(len(r.wave_height_series_m)) / fps
                    ax_wave.plot(time_axis, r.wave_height_series_m, label=f"Vague #{r.ride_id} ({r.wave_category})", color="#00e5ff", lw=2.0)
            ax_wave.set_title("Évolution de la Hauteur de Face de Vague (mètres creux ➔ crête)", fontsize=11, fontweight="600", color="#ffffff")
            ax_wave.set_xlabel("Temps de glisse (secondes)", color="#8b949e", fontsize=9)
            ax_wave.set_ylabel("Hauteur (mètres)", color="#8b949e", fontsize=9)
            ax_wave.tick_params(colors="#8b949e")
            for spine in ax_wave.spines.values():
                spine.set_color("#21262d")
            ax_wave.grid(True, alpha=0.15, color="#8b949e")
            ax_wave.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#ffffff", loc="upper left")

            st.pyplot(fig)
            plt.close()

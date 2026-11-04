"""
End-to-end Surf Tracking Pipeline.
Orchestrates detection, tracking, state machine, and virtual PTZ auto-cameraman.
Renders full tracking diagnostic video and discrete cropped ride clips.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import cv2
import numpy as np

from .detector import SurferDetector, Detection
from .tracker import SurferTracker, Track
from .state_machine import SurferStateMachine, SurferState, RideEvent
from .cameraman import VirtualCameraman, CameraBox
from .scale_estimator import MetricScaleEstimator, ScaleEstimate
from .wave_analyzer import WaveSegmentationEngine, WaveMetrics, classify_wave_size
from .pose_estimator import PoseEstimator, SurferPose
from .maneuver_classifier import ManeuverClassifier, ManeuverDetection, ManeuverType
from .utils import draw_hud, draw_minimalist_badge, draw_neon_skeleton, draw_biomechanical_lab


@dataclass
class PipelineConfig:
    """Configuration settings for the Surf Tracking Pipeline."""
    model_name: str = "yolov8n-seg.pt"
    conf_threshold: float = 0.25
    aspect_ratio: str = "16:9"         # '16:9' or '9:16'
    zoom_scale: float = 4.5
    smoothing_factor: float = 0.10     # Cameraman smoothing (EMA)
    lead_room_factor: float = 0.35     # Lead-room anticipation
    min_ride_duration_sec: float = 1.2
    speed_takeoff_thresh: float = 3.5
    pre_roll_sec: float = 2.0
    post_roll_sec: float = 1.5
    render_hud: bool = True
    render_segmentation: bool = True
    consolidate_rides: bool = True
    max_merge_gap_sec: float = 2.0
    save_split_screen: bool = True
    output_resolution: Tuple[int, int] = (1280, 720) # Resolution for exported ride clips
    surfer_height_m: float = 1.85  # 185 cm / 6'1" (1.85 m)
    board_length_m: float = 1.73   # 5'8" (1.73 m)
    enable_wave_metrics: bool = True
    enable_pose_estimation: bool = True
    pose_model_name: str = "yolov8n-pose.pt"
    split_mode: str = "biomechanical"  # 'biomechanical' or 'wide_ptz'

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "PipelineConfig":
        """Builds PipelineConfig from config.yaml with fallback to defaults."""
        from .config import load_config
        cfg = load_config(path)
        pose_cfg = cfg.get("pose_estimation", {})
        split_mode = cfg.get("cameraman", {}).get("split_mode", "biomechanical")
        return cls(
            model_name=cfg["detection"]["model_name"],
            conf_threshold=cfg["detection"]["conf_threshold"],
            aspect_ratio=cfg["cameraman"]["aspect_ratio"],
            zoom_scale=cfg["cameraman"]["zoom_scale"],
            smoothing_factor=cfg["cameraman"]["smoothing_factor"],
            lead_room_factor=cfg["cameraman"]["lead_room_factor"],
            min_ride_duration_sec=cfg["state_machine"]["min_ride_duration_sec"],
            speed_takeoff_thresh=cfg["state_machine"]["speed_takeoff_thresh"],
            max_merge_gap_sec=cfg["state_machine"]["max_merge_gap_sec"],
            consolidate_rides=cfg["state_machine"]["consolidate_rides"],
            render_hud=cfg["detection"]["enable_hud"],
            render_segmentation=cfg["detection"]["enable_segmentation"],
            save_split_screen=cfg["cameraman"]["enable_split_screen"],
            surfer_height_m=cfg["surfer"]["height_m"],
            board_length_m=cfg["surfboard"]["length_m"],
            enable_wave_metrics=cfg["wave_metrics"]["enable"],
            enable_pose_estimation=pose_cfg.get("enable", True),
            pose_model_name=pose_cfg.get("model_name", "yolov8n-pose.pt"),
            split_mode=split_mode,
        )


class SurfTrackingPipeline:
    """
    Complete computer vision pipeline for automated surf filming.
    Processes full session footage, detects wave rides, and creates
    stabilized, cinematically framed video clips.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.detector = SurferDetector(
            model_name=self.config.model_name,
            conf_threshold=self.config.conf_threshold,
        )
        self.tracker = SurferTracker()
        self.scale_estimator = MetricScaleEstimator(
            nominal_surfer_height_m=self.config.surfer_height_m,
            nominal_board_length_m=self.config.board_length_m,
        )
        self.wave_engine = WaveSegmentationEngine()
        if self.config.enable_pose_estimation:
            try:
                self.pose_estimator = PoseEstimator(model_name=self.config.pose_model_name)
            except Exception as e:
                print(f"⚠️ Avertissement : Impossible de charger le modèle de pose ({e}).")
                self.pose_estimator = None
            self.maneuver_classifier = ManeuverClassifier()
        else:
            self.pose_estimator = None
            self.maneuver_classifier = None
        self.state_machine: Optional[SurferStateMachine] = None
        self.cameraman: Optional[VirtualCameraman] = None

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        save_full_diagnostic: bool = True,
        save_ride_clips: bool = True,
    ) -> Dict:
        """
        Processes a full session video and extracts surfing rides.
        Returns a dictionary with summary statistics and paths to generated clips.
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize State Machine and Cameraman with video dimensions & FPS
        self.state_machine = SurferStateMachine(
            fps=fps,
            min_ride_duration_sec=self.config.min_ride_duration_sec,
            speed_takeoff_thresh=self.config.speed_takeoff_thresh,
            pre_roll_sec=self.config.pre_roll_sec,
            post_roll_sec=self.config.post_roll_sec,
        )
        self.cameraman = VirtualCameraman(
            frame_width=width,
            frame_height=height,
            aspect_ratio=self.config.aspect_ratio,
            zoom_scale=self.config.zoom_scale,
            smoothing_factor=self.config.smoothing_factor,
            lead_room_factor=self.config.lead_room_factor,
        )

        # Writers setup
        diag_path = out_dir / f"{video_path.stem}_tracked_overview.mp4"
        diag_writer = None
        if save_full_diagnostic:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            diag_writer = cv2.VideoWriter(str(diag_path), fourcc, fps, (width, height))
            if not diag_writer.isOpened():
                diag_writer = cv2.VideoWriter(str(diag_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # In-memory buffer of frame metadata and active camera boxes
        frame_history: List[np.ndarray] = []
        camera_boxes_history: List[Optional[CameraBox]] = []
        active_surfer_history: List[Optional[Tuple[float, float, float, float]]] = []
        frame_states_history: List[Dict[int, SurferState]] = []
        frame_metrics_history: List[Dict] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect Surfers
            detections = self.detector.detect(frame)

            # 2. Update Tracks
            active_tracks = self.tracker.update(detections, frame_idx=frame_idx)

            # 3. Update State Machine
            current_states = self.state_machine.update(active_tracks, current_frame=frame_idx)
            frame_states_history.append(current_states)

            # 4. Identify Primary Surfer for Virtual Cameraman
            # Priority: Surfer currently in TAKE_OFF or RIDING state
            primary_track: Optional[Track] = None
            riding_tracks = [t for t in active_tracks if current_states.get(t.track_id) in (SurferState.RIDING, SurferState.TAKE_OFF)]
            if len(riding_tracks) > 0:
                primary_track = riding_tracks[0]
            elif len(active_tracks) > 0:
                # Default to fastest moving surfer
                primary_track = max(active_tracks, key=lambda t: t.current_speed)

            # 5. Compute Virtual Cameraman Framing & Metrics
            cam_box: Optional[CameraBox] = None
            current_scale = 0.02
            current_spd_kmh = 0.0
            wave_met: Optional[WaveMetrics] = None
            current_pose: Optional[SurferPose] = None
            current_maneuver: Optional[ManeuverDetection] = None

            if primary_track is not None:
                cam_box = self.cameraman.update(
                    surfer_bbox=primary_track.bbox,
                    surfer_velocity=primary_track.current_velocity,
                )
                if self.config.enable_wave_metrics:
                    is_standing = (current_states.get(primary_track.track_id) == SurferState.RIDING)
                    scale_est = self.scale_estimator.update(
                        surfer_bbox=primary_track.bbox,
                        board_mask=primary_track.current_board_mask,
                        is_standing_hint=is_standing,
                    )
                    current_scale = scale_est.meters_per_pixel
                    current_spd_kmh = primary_track.current_speed * current_scale * fps * 3.6

                    vx, _ = primary_track.current_velocity
                    direction = "RIGHT" if vx >= 0 else "LEFT"
                    wave_met = self.wave_engine.analyze(
                        frame=frame,
                        surfer_bbox=primary_track.bbox,
                        direction=direction,
                        meters_per_pixel=current_scale,
                    )

                # Pose Estimation and Maneuver Classification on riding/takeoff surfer
                if self.pose_estimator is not None and current_states.get(primary_track.track_id) in (SurferState.RIDING, SurferState.TAKE_OFF):
                    current_pose = self.pose_estimator.estimate(frame, primary_track.bbox)
                    if self.maneuver_classifier is not None:
                        current_maneuver = self.maneuver_classifier.update(
                            pose=current_pose,
                            wave=wave_met,
                            surfer_center=primary_track.centroid,
                            velocity=primary_track.current_velocity,
                            speed_kmh=current_spd_kmh,
                        )

            camera_boxes_history.append(cam_box)
            active_surfer_history.append(primary_track.bbox if primary_track else None)
            frame_metrics_history.append({
                "scale_m_per_px": current_scale,
                "speed_kmh": current_spd_kmh,
                "wave": wave_met,
                "pose": current_pose,
                "maneuver": current_maneuver,
            })

            # 6. Render Diagnostic Overview
            if diag_writer is not None:
                diag_frame = frame.copy()

                # Draw wave mass polygon if segmented
                if wave_met is not None and wave_met.wave_polygon is not None and self.config.render_segmentation:
                    w_poly = wave_met.wave_polygon.astype(np.int32).reshape((-1, 1, 2))
                    w_overlay = diag_frame.copy()
                    cv2.fillPoly(w_overlay, [w_poly], color=(200, 160, 40))
                    cv2.addWeighted(w_overlay, 0.20, diag_frame, 0.80, 0, diag_frame)
                    cv2.polylines(diag_frame, [w_poly], isClosed=True, color=(255, 210, 80), thickness=1, lineType=cv2.LINE_AA)

                # Draw neon skeleton on surfer if estimated
                if current_pose is not None and self.config.enable_pose_estimation:
                    diag_frame = draw_neon_skeleton(diag_frame, current_pose.keypoints)

                for track in active_tracks:
                    st = current_states.get(track.track_id, SurferState.PADDLING)
                    diag_frame = draw_hud(
                        diag_frame,
                        surfer_bbox=track.bbox,
                        state=st,
                        speed_px=track.current_speed,
                        trajectory=track.history,
                        mask=track.current_mask,
                        board_mask=track.current_board_mask,
                        render_segmentation=self.config.render_segmentation,
                    )

                # Draw active maneuver badge in diagnostic HUD
                if current_maneuver is not None and current_maneuver.is_active:
                    m_text = f"{current_maneuver.name.upper()} ({current_maneuver.score_out_of_10:.1f}/10)"
                    cv2.rectangle(diag_frame, (20, height - 60), (450, height - 20), (16, 20, 26), -1)
                    cv2.rectangle(diag_frame, (20, height - 60), (450, height - 20), (0, 229, 255), 1)
                    cv2.putText(diag_frame, m_text, (35, height - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 229, 255), 2, cv2.LINE_AA)

                # Draw Virtual Camera Crop Window (Magenta boundary rectangle)
                if cam_box is not None:
                    cv2.rectangle(
                        diag_frame,
                        (cam_box.x1, cam_box.y1),
                        (cam_box.x2, cam_box.y2),
                        (255, 0, 200),
                        2,
                    )
                    cv2.putText(
                        diag_frame,
                        f"VIRTUAL PTZ [{self.config.aspect_ratio}]",
                        (cam_box.x1 + 8, cam_box.y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 200),
                        1,
                        cv2.LINE_AA,
                    )
                diag_writer.write(diag_frame)

            if progress_callback and frame_idx % 15 == 0:
                progress = min(0.95, frame_idx / total_frames)
                progress_callback(progress, f"Analyse des frames ({frame_idx}/{total_frames})...")

            frame_idx += 1

        cap.release()
        if diag_writer is not None:
            diag_writer.release()

        # Finalize any ongoing ride at the end of the video
        riding_ids = list(self.state_machine.active_rides.keys())
        for tid in riding_ids:
            self.state_machine._finalize_ride(tid, frame_idx, end_state=SurferState.EXIT)

        # Consolidate fragmented rides into unified waves
        final_rides = self.state_machine.get_consolidated_rides(max_gap_sec=self.config.max_merge_gap_sec) if self.config.consolidate_rides else self.state_machine.completed_rides

        # Enrich each ride with physical wave & kinematic metrics
        for ride in final_rides:
            s_idx = max(0, ride.start_frame)
            e_idx = min(len(frame_metrics_history), ride.end_frame)
            ride_mets = frame_metrics_history[s_idx:e_idx]
            if ride_mets:
                speeds = [m["speed_kmh"] for m in ride_mets if m.get("speed_kmh", 0.0) > 0.0]
                waves = [m["wave"] for m in ride_mets if m.get("wave") is not None]
                wave_hs = [w.height_meters for w in waves if w.height_meters > 0.0]
                maneuvers = [
                    m["maneuver"] for m in ride_mets
                    if m.get("maneuver") is not None and m["maneuver"].is_active and m["maneuver"].maneuver_type != ManeuverType.NONE
                ]

                if speeds:
                    ride.max_speed_kmh = round(float(np.max(speeds)), 1)
                    ride.avg_speed_kmh = round(float(np.mean(speeds)), 1)
                    ride.speed_series_kmh = [round(float(s), 1) for s in speeds]

                dist_m = sum(m.get("speed_kmh", 0.0) / 3.6 / fps for m in ride_mets)
                ride.distance_meters = round(float(dist_m), 1)

                if wave_hs:
                    rep_wave_h = float(np.percentile(wave_hs, 80))
                    ride.wave_height_meters = round(rep_wave_h, 2)
                    ride.wave_height_feet = round(rep_wave_h * 3.28084, 1)
                    ride.wave_category = classify_wave_size(rep_wave_h)
                    ride.wave_height_series_m = [round(float(h), 2) for h in wave_hs]
                else:
                    ride.wave_height_meters = 1.35
                    ride.wave_height_feet = 4.4
                    ride.wave_category = classify_wave_size(1.35)

                if maneuvers:
                    unique_names = list(dict.fromkeys(m.name for m in maneuvers))
                    ride.maneuvers_detected = unique_names
                    best_m = max(maneuvers, key=lambda m: m.score_out_of_10)
                    ride.best_maneuver = best_m.name
                    ride.maneuver_score = best_m.score_out_of_10
                else:
                    ride.best_maneuver = "Glisse Neutre"
                    ride.maneuver_score = 5.0
                    ride.maneuvers_detected = ["Glisse Neutre"]

        # Step 7: Render Cropped Ride Clips
        rendered_clips = []
        split_clips = []
        if save_ride_clips and len(final_rides) > 0:
            rendered_clips = self._render_ride_clips(
                video_path=video_path,
                out_dir=out_dir,
                rides=final_rides,
                camera_boxes=camera_boxes_history,
                active_bboxes=active_surfer_history,
                states_history=frame_states_history,
                frame_metrics_history=frame_metrics_history,
                fps=fps,
            )
            if self.config.save_split_screen:
                split_clips = self._render_split_screen_clips(
                    video_path=video_path,
                    out_dir=out_dir,
                    rides=final_rides,
                    camera_boxes=camera_boxes_history,
                    frame_metrics_history=frame_metrics_history,
                    fps=fps,
                )

        if progress_callback:
            progress_callback(1.0, f"Session terminée ! {len(final_rides)} ride(s) détecté(s).")

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "rides_detected": len(final_rides),
            "rides": final_rides,
            "diagnostic_video": str(diag_path) if save_full_diagnostic else None,
            "ride_clips": rendered_clips,
            "split_clips": split_clips,
        }

    def _render_ride_clips(
        self,
        video_path: Path,
        out_dir: Path,
        rides: List[RideEvent],
        camera_boxes: List[Optional[CameraBox]],
        active_bboxes: List[Optional[Tuple[float, float, float, float]]],
        states_history: List[Dict[int, SurferState]],
        frame_metrics_history: List[Dict],
        fps: float,
    ) -> List[str]:
        """
        Rereads specific frame ranges and exports cropped, stabilized video clips for each ride.
        """
        clip_paths = []
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        target_w, target_h = self.config.output_resolution

        if self.config.aspect_ratio == "9:16":
            # For 9:16 vertical reels, swap target resolution
            target_w, target_h = target_h, target_w

        for ride in rides:
            clip_name = f"{video_path.stem}_ride_{ride.ride_id:02d}_{ride.direction.lower()}_{ride.duration_sec:.1f}s.mp4"
            clip_path = out_dir / clip_name

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, ride.start_frame)

            writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (target_w, target_h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

            for f_idx in range(ride.start_frame, min(ride.end_frame, len(camera_boxes))):
                ret, frame = cap.read()
                if not ret:
                    break

                box = camera_boxes[f_idx]
                if box is None or not box.is_valid:
                    crop = frame
                else:
                    crop = frame[box.y1:box.y2, box.x1:box.x2]
                    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                        crop = frame

                # Resize to target output resolution
                crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Minimalist Floating Telemetry Badge Overlay
                if self.config.render_hud:
                    met = frame_metrics_history[f_idx] if f_idx < len(frame_metrics_history) else {}
                    spd_kmh = met.get("speed_kmh", 0.0)
                    w_obj = met.get("wave")
                    wave_h = w_obj.height_meters if w_obj else ride.wave_height_meters
                    time_in_ride = max(0.0, (f_idx - ride.takeoff_frame) / fps)
                    cum_dist = sum(
                        frame_metrics_history[i].get("speed_kmh", 0.0) / 3.6 / fps
                        for i in range(ride.start_frame, min(f_idx + 1, len(frame_metrics_history)))
                    )
                    crop_resized = draw_minimalist_badge(
                        crop_resized,
                        speed_kmh=spd_kmh,
                        wave_height_m=wave_h,
                        distance_m=cum_dist,
                        ride_time_sec=time_in_ride,
                        max_speed_kmh=ride.max_speed_kmh,
                    )

                writer.write(crop_resized)

            cap.release()
            writer.release()
            clip_paths.append(str(clip_path))

        return clip_paths

    def _render_split_screen_clips(
        self,
        video_path: Path,
        out_dir: Path,
        rides: List[RideEvent],
        camera_boxes: List[Optional[CameraBox]],
        frame_metrics_history: List[Dict],
        fps: float,
    ) -> List[str]:
        """
        Renders a dual-screen comparison video (1920x1080) for each ride:
        - Mode 'biomechanical': Left: Action crop with real surfer + neon joints; Right: High-tech 3D skeleton studio & kinetic gauges.
        - Mode 'wide_ptz': Left: Wide beach shot with PTZ box; Right: Dynamic stabilized PTZ crop.
        """
        split_paths = []
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        canvas_w, canvas_h = 1920, 1080
        is_vertical = (self.config.aspect_ratio == "9:16")
        is_biomech = (self.config.split_mode == "biomechanical")

        for ride in rides:
            mode_tag = "biomech" if is_biomech else "split_screen"
            split_name = f"{video_path.stem}_ride_{ride.ride_id:02d}_{mode_tag}_{ride.duration_sec:.1f}s.mp4"
            split_path = out_dir / split_name

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, ride.start_frame)

            writer = cv2.VideoWriter(str(split_path), fourcc, fps, (canvas_w, canvas_h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(split_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

            for f_idx in range(ride.start_frame, min(ride.end_frame, len(camera_boxes))):
                ret, frame = cap.read()
                if not ret:
                    break

                box = camera_boxes[f_idx] if f_idx < len(camera_boxes) else None
                canvas = np.full((canvas_h, canvas_w, 3), 13, dtype=np.uint8)

                met = frame_metrics_history[f_idx] if f_idx < len(frame_metrics_history) else {}
                spd_kmh = met.get("speed_kmh", 0.0)
                w_obj = met.get("wave")
                wave_h = w_obj.height_meters if w_obj else ride.wave_height_meters
                pose_obj = met.get("pose")
                m_obj = met.get("maneuver")

                if is_biomech:
                    # =========================================================
                    # STUDIO BIOMECHANICAL LAB (Side-by-Side 1080p)
                    # =========================================================
                    # 1. Left Action Pane
                    if box is not None and box.is_valid:
                        crop = frame[box.y1:box.y2, box.x1:box.x2].copy()
                        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                            crop = frame.copy()
                    else:
                        crop = frame.copy()

                    # Overlay neon skeleton on the cropped surfer if pose detected
                    if pose_obj is not None and box is not None and box.is_valid:
                        kpts_crop = pose_obj.keypoints.copy()
                        kpts_crop[:, 0] -= box.x1
                        kpts_crop[:, 1] -= box.y1
                        crop = draw_neon_skeleton(crop, kpts_crop, color=(0, 229, 255))

                    pane_w = 920
                    pane_h = 1040
                    if is_vertical:
                        scaled_w = int(pane_h * 9 / 16)
                        crop_scaled = cv2.resize(crop, (scaled_w, pane_h), interpolation=cv2.INTER_AREA)
                        x_off = 20 + (pane_w - scaled_w) // 2
                        canvas[20:1060, x_off:x_off + scaled_w] = crop_scaled
                    else:
                        scaled_h = int(pane_w * 9 / 16)
                        crop_scaled = cv2.resize(crop, (pane_w, scaled_h), interpolation=cv2.INTER_AREA)
                        y_off = 20 + (pane_h - scaled_h) // 2
                        canvas[y_off:y_off + scaled_h, 20:20 + pane_w] = crop_scaled

                    # Left Action Badges
                    cv2.rectangle(canvas, (35, 35), (420, 75), (18, 22, 28), -1)
                    cv2.rectangle(canvas, (35, 35), (420, 75), (48, 54, 61), 1)
                    cv2.circle(canvas, (55, 55), 6, (0, 255, 120), -1)
                    cv2.putText(canvas, "PLAN RECADRE (ACTION REELLE)", (75, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                    if m_obj is not None and m_obj.is_active:
                        m_txt = f"{m_obj.name.upper()}  •  {m_obj.score_out_of_10}/10"
                        cv2.rectangle(canvas, (35, 995), (450, 1045), (18, 22, 28), -1)
                        cv2.rectangle(canvas, (35, 995), (450, 1045), (0, 229, 255), 1)
                        cv2.putText(canvas, m_txt, (50, 1030), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 2, cv2.LINE_AA)

                    # 2. Right Biomechanical Lab Pane
                    lab_pane = draw_biomechanical_lab(
                        canvas_w=pane_w,
                        canvas_h=pane_h,
                        pose=pose_obj,
                        maneuver=m_obj,
                        speed_kmh=spd_kmh,
                        wave_height_m=wave_h,
                    )
                    canvas[20:1060, 960:960 + pane_w] = lab_pane

                    # Borders and separator
                    cv2.rectangle(canvas, (20, 20), (940, 1060), (35, 40, 50), 1)
                    cv2.rectangle(canvas, (960, 20), (1880, 1060), (35, 40, 50), 1)
                    cv2.line(canvas, (950, 0), (950, 1080), (25, 30, 40), 2)

                else:
                    # =========================================================
                    # ORIGINAL WIDE VS PTZ SPLIT SCREEN
                    # =========================================================
                    wide_frame = frame.copy()
                    if box is not None and box.is_valid:
                        cv2.rectangle(wide_frame, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 220), 4)
                        box_lbl = f"VIRTUAL PTZ [{self.config.aspect_ratio}]"
                        cv2.rectangle(wide_frame, (box.x1, max(0, box.y1 - 32)), (box.x1 + 270, box.y1), (12, 12, 12), -1)
                        cv2.rectangle(wide_frame, (box.x1, max(0, box.y1 - 32)), (box.x1 + 270, box.y1), (255, 0, 220), 1)
                        cv2.putText(wide_frame, box_lbl, (box.x1 + 8, max(0, box.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    if box is not None and box.is_valid:
                        crop = frame[box.y1:box.y2, box.x1:box.x2]
                        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                            crop = frame
                    else:
                        crop = frame

                    if is_vertical:
                        left_resized = cv2.resize(wide_frame, (1280, 720), interpolation=cv2.INTER_AREA)
                        canvas[180:900, 0:1280] = left_resized
                        cv2.rectangle(canvas, (0, 180), (1279, 899), (45, 45, 45), 1)

                        cv2.rectangle(canvas, (24, 196), (340, 236), (10, 10, 10), -1)
                        cv2.rectangle(canvas, (24, 196), (340, 236), (0, 255, 200), 1)
                        cv2.putText(canvas, "PLAN LARGE FIXE (PLAGE)", (36, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 200), 2, cv2.LINE_AA)

                        cv2.line(canvas, (1280 + 16, 0), (1280 + 16, 1080), (35, 40, 50), 2)

                        right_resized = cv2.resize(crop, (608, 1080), interpolation=cv2.INTER_AREA)
                        if self.config.render_hud:
                            time_in_ride = max(0.0, (f_idx - ride.takeoff_frame) / fps)
                            cum_dist = sum(
                                frame_metrics_history[i].get("speed_kmh", 0.0) / 3.6 / fps
                                for i in range(ride.start_frame, min(f_idx + 1, len(frame_metrics_history)))
                            )
                            right_resized = draw_minimalist_badge(
                                right_resized,
                                speed_kmh=spd_kmh,
                                wave_height_m=wave_h,
                                distance_m=cum_dist,
                                ride_time_sec=time_in_ride,
                                max_speed_kmh=ride.max_speed_kmh,
                            )

                        canvas[0:1080, 1280+32:1920] = right_resized

                        cv2.rectangle(canvas, (1328, 20), (1750, 60), (10, 10, 10), -1)
                        cv2.circle(canvas, (1348, 40), 6, (0, 0, 255), -1)
                        cv2.putText(canvas, "DIGITAL PTZ (REELS 9:16)", (1366, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                    else:
                        pane_w = 940
                        pane_h = int(pane_w * 9 / 16)
                        y_off = (canvas_h - pane_h) // 2

                        left_resized = cv2.resize(wide_frame, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
                        canvas[y_off:y_off+pane_h, 15:15+pane_w] = left_resized

                        cv2.rectangle(canvas, (35, y_off + 15), (340, y_off + 50), (10, 10, 10), -1)
                        cv2.putText(canvas, "PLAN LARGE FIXE (PLAGE)", (45, y_off + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

                        cv2.line(canvas, (960, 0), (960, 1080), (35, 40, 50), 2)

                        right_resized = cv2.resize(crop, (pane_w, pane_h), interpolation=cv2.INTER_AREA)
                        if self.config.render_hud:
                            time_in_ride = max(0.0, (f_idx - ride.takeoff_frame) / fps)
                            cum_dist = sum(
                                frame_metrics_history[i].get("speed_kmh", 0.0) / 3.6 / fps
                                for i in range(ride.start_frame, min(f_idx + 1, len(frame_metrics_history)))
                            )
                            right_resized = draw_minimalist_badge(
                                right_resized,
                                speed_kmh=spd_kmh,
                                wave_height_m=wave_h,
                                distance_m=cum_dist,
                                ride_time_sec=time_in_ride,
                                max_speed_kmh=ride.max_speed_kmh,
                            )

                        canvas[y_off:y_off+pane_h, 965:965+pane_w] = right_resized

                        cv2.rectangle(canvas, (985, y_off + 15), (1380, y_off + 50), (10, 10, 10), -1)
                        cv2.circle(canvas, (1005, y_off + 32), 6, (0, 0, 255), -1)
                        cv2.putText(canvas, f"DIGITAL PTZ ({self.config.aspect_ratio})", (1022, y_off + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                writer.write(canvas)

            cap.release()
            writer.release()
            split_paths.append(str(split_path))

        return split_paths

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
from .utils import draw_hud, draw_minimalist_badge, draw_neon_skeleton


@dataclass
class PipelineConfig:
    """Configuration settings for the Surf Tracking Pipeline."""
    model_name: str = "weights/yolov8n_surf_v2_best.pt" if Path("weights/yolov8n_surf_v2_best.pt").exists() else "yolov8n-seg.pt"
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
    save_split_screen: bool = False
    output_resolution: Tuple[int, int] = (1280, 720) # Resolution for exported ride clips
    surfer_height_m: float = 1.75  # 175 cm / 5'9" (nominal surfer height)
    board_length_m: float = 1.73   # 5'8" (1.73 m)
    enable_wave_metrics: bool = True
    enable_pose_estimation: bool = False
    pose_model_name: str = "yolov8n-pose.pt"
    split_mode: str = "wide_ptz"
    fast_scan: bool = True

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "PipelineConfig":
        """Builds PipelineConfig from config.yaml with fallback to defaults."""
        from .config import load_config
        cfg = load_config(path)
        pose_cfg = cfg.get("pose_estimation", {})
        split_mode = cfg.get("cameraman", {}).get("split_mode", "wide_ptz")
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
            save_split_screen=cfg.get("cameraman", {}).get("enable_split_screen", False),
            surfer_height_m=cfg["surfer"]["height_m"],
            board_length_m=cfg["surfboard"]["length_m"],
            enable_wave_metrics=cfg["wave_metrics"]["enable"],
            enable_pose_estimation=pose_cfg.get("enable", False),
            pose_model_name=pose_cfg.get("model_name", "yolov8n-pose.pt"),
            split_mode=split_mode,
            fast_scan=cfg.get("fast_scan", True),
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
        self.pose_estimator = None
        if self.config.enable_pose_estimation:
            try:
                self.pose_estimator = PoseEstimator(model_name=self.config.pose_model_name)
            except Exception as e:
                print(f"⚠️ Avertissement : Impossible de charger le modèle de pose ({e}).")
                self.pose_estimator = None
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
        Processes a surf session video file through the linear mono-pass pipeline.
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        # Video writer for full session diagnostic
        diag_writer = None
        diag_path = None
        if save_full_diagnostic:
            diag_path = out_dir / f"{video_path.stem}_diagnostic.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            diag_writer = cv2.VideoWriter(str(diag_path), fourcc, fps, (width, height))
            if not diag_writer.isOpened():
                diag_writer = cv2.VideoWriter(str(diag_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        camera_boxes_history: List[Optional[CameraBox]] = []
        frame_states_history: List[Dict[int, SurferState]] = []
        active_surfer_history: List[Optional[Tuple[float, float, float, float]]] = []
        frame_metrics_history: List[Dict] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detection
            detections = self.detector.detect(frame)

            # 2. Tracking
            active_tracks = self.tracker.update(detections, frame_idx=frame_idx)

            # 3. State Machine Update
            current_states = self.state_machine.update(active_tracks, current_frame=frame_idx)
            frame_states_history.append(current_states)

            # 4. Virtual Cameraman & Primary Target Framing
            primary_track = None
            riding_tracks = [
                t for t in active_tracks
                if current_states.get(t.track_id) in (SurferState.RIDING, SurferState.TAKE_OFF)
            ]
            if riding_tracks:
                primary_track = max(riding_tracks, key=lambda t: t.current_speed)
            elif active_tracks:
                primary_track = max(active_tracks, key=lambda t: t.bbox[2]*t.bbox[3])

            cam_box = None
            current_scale = 0.022
            current_spd_kmh = 0.0
            wave_met: Optional[WaveMetrics] = None
            current_pose: Optional[SurferPose] = None

            if primary_track is not None:
                cam_box = self.cameraman.update(
                    surfer_bbox=primary_track.bbox,
                    surfer_velocity=primary_track.current_velocity,
                )
                is_standing = (current_states.get(primary_track.track_id) == SurferState.RIDING)

                if self.pose_estimator is not None and current_states.get(primary_track.track_id) in (SurferState.RIDING, SurferState.TAKE_OFF):
                    current_pose = self.pose_estimator.estimate(frame, primary_track.bbox)

                if self.config.enable_wave_metrics:
                    scale_est = self.scale_estimator.update(
                        surfer_bbox=primary_track.bbox,
                        board_mask=primary_track.current_board_mask,
                        is_standing_hint=is_standing,
                        pose=current_pose,
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

            camera_boxes_history.append(cam_box)
            active_surfer_history.append(primary_track.bbox if primary_track else None)
            frame_metrics_history.append({
                "scale_m_per_px": current_scale,
                "speed_kmh": current_spd_kmh,
                "wave": wave_met,
                "pose": current_pose,
            })

            # Render Diagnostic Overview
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
            ride_states = frame_states_history[s_idx:e_idx] if s_idx < len(frame_states_history) else []

            # Isolate stabilized riding frames (ignore pre-roll paddling and initial pop-up transition)
            takeoff_settled_frame = max(s_idx, ride.takeoff_frame + int(fps * 0.5))
            riding_mets = [
                m for i, m in enumerate(ride_mets)
                if (s_idx + i) >= takeoff_settled_frame and (
                    i >= len(ride_states) or ride_states[i].get(ride.surfer_id) in (SurferState.RIDING, SurferState.TAKE_OFF)
                )
            ]
            eval_mets = riding_mets if len(riding_mets) >= 5 else ride_mets

            if eval_mets:
                speeds = [m["speed_kmh"] for m in eval_mets if m.get("speed_kmh", 0.0) > 0.0]
                waves = [m["wave"] for m in eval_mets if m.get("wave") is not None]
                wave_hs = [w.height_meters for w in waves if w.height_meters > 0.0]

                if speeds:
                    # Rolling median smoothing to reject detector jitter
                    smooth_speeds = [
                        float(np.median(speeds[max(0, j-2):min(len(speeds), j+3)]))
                        for j in range(len(speeds))
                    ]
                    # Peak speed is 95th percentile of stabilized riding speeds
                    ride.max_speed_kmh = round(float(np.percentile(smooth_speeds, 95)), 1)
                    ride.avg_speed_kmh = round(float(np.mean(smooth_speeds)), 1)
                    ride.speed_series_kmh = [round(float(s), 1) for s in smooth_speeds]

                dist_m = sum(m.get("speed_kmh", 0.0) / 3.6 / fps for m in eval_mets)
                ride.distance_meters = round(float(dist_m), 1)

                if wave_hs:
                    # Representative wave face height (40th percentile of stabilized face)
                    rep_wave_h = float(np.percentile(wave_hs, 40))
                    ride.wave_height_meters = round(rep_wave_h, 2)
                    ride.wave_height_feet = round(rep_wave_h * 3.28084, 1)
                    ride.wave_category = classify_wave_size(rep_wave_h)
                    ride.wave_height_series_m = [round(float(h), 2) for h in wave_hs]
                else:
                    ride.wave_height_meters = 1.00
                    ride.wave_height_feet = 3.3
                    ride.wave_category = classify_wave_size(1.00)

        # Step 7: Render Cropped Ride Clips
        rendered_clips = []
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

        if progress_callback:
            progress_callback(1.0, f"Session terminée ! {len(final_rides)} ride(s) détecté(s).")

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "rides_detected": len(final_rides),
            "rides": final_rides,
            "diagnostic_video": str(diag_path) if save_full_diagnostic else None,
            "ride_clips": rendered_clips,
            "split_clips": [],
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

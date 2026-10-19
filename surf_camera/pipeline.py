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
from .utils import draw_hud


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
    output_resolution: Tuple[int, int] = (1280, 720) # Resolution for exported ride clips


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

            # 5. Compute Virtual Cameraman Framing
            cam_box: Optional[CameraBox] = None
            if primary_track is not None:
                cam_box = self.cameraman.update(
                    surfer_bbox=primary_track.bbox,
                    surfer_velocity=primary_track.current_velocity,
                )
            camera_boxes_history.append(cam_box)
            active_surfer_history.append(primary_track.bbox if primary_track else None)

            # 6. Render Diagnostic Overview
            if diag_writer is not None:
                diag_frame = frame.copy()
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
        }

    def _render_ride_clips(
        self,
        video_path: Path,
        out_dir: Path,
        rides: List[RideEvent],
        camera_boxes: List[Optional[CameraBox]],
        active_bboxes: List[Optional[Tuple[float, float, float, float]]],
        states_history: List[Dict[int, SurferState]],
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
                    # Fallback to center frame
                    crop = frame
                else:
                    crop = frame[box.y1:box.y2, box.x1:box.x2]
                    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                        crop = frame

                # Resize to target output resolution
                crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Optional Telemetry Overlay on the cropped cinematic clip
                if self.config.render_hud:
                    st_dict = states_history[f_idx] if f_idx < len(states_history) else {}
                    surfer_st = st_dict.get(ride.surfer_id, SurferState.RIDING)
                    time_in_ride = max(0.0, (f_idx - ride.takeoff_frame) / fps)

                    # Modern cinematic overlay
                    cv2.putText(
                        crop_resized,
                        f"RIDE #{ride.ride_id} | {ride.direction}",
                        (20, target_h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        crop_resized,
                        f"{surfer_st.value.upper()} | {time_in_ride:.1f}s",
                        (20, target_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 100) if surfer_st == SurferState.RIDING else (0, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(crop_resized)

            cap.release()
            writer.release()
            clip_paths.append(str(clip_path))

        return clip_paths

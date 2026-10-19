"""
Multi-surfer spatial tracker with velocity estimation and occlusion recovery.
Maintains identity persistence across wave spray and duck dives.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


@dataclass
class Track:
    """Represents a continuous track of a surfer across frames."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    start_frame: int
    last_frame: int
    history: List[Tuple[float, float]] = field(default_factory=list)  # [(cx, cy), ...]
    frame_indices: List[int] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)  # [(vx, vy), ...]
    hits: int = 1
    lost_frames: int = 0
    confirmed: bool = False
    current_mask: Optional[np.ndarray] = None
    current_board_mask: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def current_aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def current_velocity(self) -> Tuple[float, float]:
        if len(self.velocities) == 0:
            return (0.0, 0.0)
        # Smoothed over the last 3 velocity vectors
        recent = self.velocities[-3:]
        vx = float(np.mean([v[0] for v in recent]))
        vy = float(np.mean([v[1] for v in recent]))
        return (vx, vy)

    @property
    def current_speed(self) -> float:
        vx, vy = self.current_velocity
        return float(np.sqrt(vx**2 + vy**2))


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


class SurferTracker:
    """
    Spatio-temporal tracker designed for ocean dynamics.
    Handles scale variations and occlusions caused by wave crests and foam.
    """
    def __init__(
        self,
        max_lost_frames: int = 25,
        min_hits_to_confirm: int = 3,
        iou_match_thresh: float = 0.25,
        max_center_dist_ratio: float = 2.5,
    ):
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_match_thresh = iou_match_thresh
        self.max_center_dist_ratio = max_center_dist_ratio

        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """
        Updates the active tracks with new detections at the given frame index.
        Returns the list of currently active confirmed tracks.
        """
        # Step 1: Predict new track positions based on velocity
        active_track_ids = list(self.tracks.keys())
        predicted_boxes = {}
        for tid in active_track_ids:
            track = self.tracks[tid]
            vx, vy = track.current_velocity
            x1, y1, x2, y2 = track.bbox
            predicted_boxes[tid] = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

        # Step 2: Build Cost Matrix (combining IoU and center distance)
        matched_tracks = set()
        matched_dets = set()

        if len(active_track_ids) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(active_track_ids):
                track = self.tracks[tid]
                pred_box = predicted_boxes[tid]
                for j, det in enumerate(detections):
                    iou = compute_iou(pred_box, det.bbox)
                    # Normalized center distance
                    tcx, tcy = (pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0
                    dcx, dcy = det.center
                    dist = np.sqrt((tcx - dcx)**2 + (tcy - dcy)**2)
                    max_allowed = max(track.width, track.height) * self.max_center_dist_ratio
                    dist_score = min(1.0, dist / max(1.0, max_allowed))

                    # Cost is low when IoU is high and distance is low
                    cost = (1.0 - iou) * 0.7 + dist_score * 0.3
                    cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - self.iou_match_thresh):
                    tid = active_track_ids[r]
                    matched_tracks.add(tid)
                    matched_dets.add(c)
                    self._update_track(self.tracks[tid], detections[c], frame_idx)

        # Step 3: Handle unmatched active tracks (increment lost_frames)
        for tid in active_track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.lost_frames += 1
                # Forward project bounding box slightly
                vx, vy = track.current_velocity
                x1, y1, x2, y2 = track.bbox
                track.bbox = (x1 + vx * 0.5, y1 + vy * 0.5, x2 + vx * 0.5, y2 + vy * 0.5)

        # Step 4: Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, frame_idx)

        # Step 5: Remove dead tracks
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.lost_frames > self.max_lost_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return confirmed active tracks
        return [
            track for track in self.tracks.values()
            if track.confirmed and track.lost_frames == 0
        ]

    def _update_track(self, track: Track, det: Detection, frame_idx: int):
        """Updates track kinematics with a matched detection."""
        old_center = track.center
        new_center = det.center
        dt = max(1, frame_idx - track.last_frame)
        vx = (new_center[0] - old_center[0]) / dt
        vy = (new_center[1] - old_center[1]) / dt

        track.bbox = det.bbox
        track.confidence = det.confidence
        track.last_frame = frame_idx
        track.history.append(new_center)
        track.frame_indices.append(frame_idx)
        track.aspect_ratios.append(det.aspect_ratio)
        track.velocities.append((vx, vy))
        track.hits += 1
        track.lost_frames = 0
        track.current_mask = det.mask
        track.current_board_mask = det.board_mask

        if track.hits >= self.min_hits_to_confirm:
            track.confirmed = True

    def _create_track(self, det: Detection, frame_idx: int):
        """Initializes a new track for a newly detected surfer."""
        tid = self._next_id
        self._next_id += 1

        new_track = Track(
            track_id=tid,
            bbox=det.bbox,
            confidence=det.confidence,
            start_frame=frame_idx,
            last_frame=frame_idx,
            history=[det.center],
            frame_indices=[frame_idx],
            aspect_ratios=[det.aspect_ratio],
            velocities=[(0.0, 0.0)],
            hits=1,
            lost_frames=0,
            confirmed=False,
            current_mask=det.mask,
            current_board_mask=det.board_mask,
        )
        self.tracks[tid] = new_track

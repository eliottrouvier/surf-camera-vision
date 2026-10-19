"""
Detector module for surfers and surfboards in open-water environments.
Leverages Ultralytics YOLO models with multi-scale confidence thresholds
and hardware acceleration (Apple Silicon MPS, CUDA, CPU).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a detected object in a video frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in absolute pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    has_board: bool = False
    mask: Optional[np.ndarray] = None          # Polygon contour coordinates (N, 2)
    board_mask: Optional[np.ndarray] = None    # Surfboard polygon contour coordinates (M, 2)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center coordinates (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio: > 1.0 when prone (paddling), < 1.0 when standing/riding."""
        return self.width / self.height

    @property
    def area(self) -> float:
        return self.width * self.height


class SurferDetector:
    """
    Detects surfers and boards in panoramic or zoomed surf footage.
    Supports standard object detection (yolov8n.pt) and instance segmentation (yolov8n-seg.pt).
    """
    PERSON_CLASS = 0
    SURFBOARD_CLASS = 37

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        min_box_area: float = 100.0,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_box_area = min_box_area

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection & instance segmentation on a single RGB or BGR frame.
        Associates detected surfboards with adjacent person detections.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS, self.SURFBOARD_CLASS],
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks_xy = res.masks.xy if (hasattr(res, "masks") and res.masks is not None) else None

        persons: List[Detection] = []
        boards: List[Tuple[Tuple[float, float, float, float], Optional[np.ndarray]]] = []

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i, (bbox, conf, cls_id) in enumerate(zip(coords, confs, cls_ids)):
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            poly = masks_xy[i] if (masks_xy is not None and i < len(masks_xy)) else None

            if cls_id == self.PERSON_CLASS:
                persons.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name="surfer",
                    mask=poly,
                ))
            elif cls_id == self.SURFBOARD_CLASS:
                boards.append(((float(x1), float(y1), float(x2), float(y2)), poly))

        # Associate surfboards with persons if overlapping or nearby
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            for (bx1, by1, bx2, by2), b_poly in boards:
                # Check bounding box overlap
                ix1 = max(px1, bx1)
                iy1 = max(py1, by1)
                ix2 = min(px2, bx2)
                iy2 = min(py2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    person.has_board = True
                    person.board_mask = b_poly
                    break
                # Or check vertical proximity (board right below surfer)
                if abs(person.center[0] - (bx1 + bx2)/2.0) < person.width * 1.5:
                    if 0 < by1 - py2 < person.height * 0.8:
                        person.has_board = True
                        person.board_mask = b_poly
                        break

        return persons

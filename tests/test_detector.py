"""
Unit tests for SurferDetector and multi-model weight switching.
Validates loading of both specialized marine model (surf_v2) and standard COCO model (coco).
"""

import pytest
import numpy as np
from pathlib import Path

from surf_camera.detector import SurferDetector, Detection
from surf_camera.config import resolve_model_path, SurfConfig


def test_resolve_model_path():
    """Validates automatic weight path resolution for surf_v2 and coco."""
    surf_path = resolve_model_path("surf_v2")
    assert "yolov8n_surf_v2_best.pt" in surf_path or "yolov8n.pt" in surf_path
    assert Path(surf_path).exists()

    coco_path = resolve_model_path("coco")
    assert "yolov8n.pt" in coco_path
    assert Path(coco_path).exists()


def test_surfer_detector_surf_v2_loading():
    """Validates that surf_v2 model initializes with surfer and surfboard class mapping."""
    detector = SurferDetector(model_type="surf_v2", conf_threshold=0.20)
    assert detector.model is not None
    assert detector.person_class_id == 0
    # In surf_v2 fine-tuned model, surfboard is class 1
    assert detector.board_class_id == 1

    # Run detection on synthetic frame without crash
    synthetic_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    detections = detector.detect(synthetic_frame)
    assert isinstance(detections, list)


def test_surfer_detector_coco_loading():
    """Validates that coco model initializes with person (0) and surfboard (37) class mapping."""
    detector = SurferDetector(model_type="coco", conf_threshold=0.20)
    assert detector.model is not None
    assert detector.person_class_id == 0
    # In standard COCO 80 classes, surfboard is class 37
    assert detector.board_class_id == 37

    # Run detection on synthetic frame without crash
    synthetic_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    detections = detector.detect(synthetic_frame)
    assert isinstance(detections, list)


def test_surf_config_defaults():
    """Validates default SurfConfig dataclass contract."""
    cfg = SurfConfig()
    assert cfg.model_type == "surf_v2"
    assert cfg.conf_threshold == 0.20
    assert cfg.aspect_ratio == "9:16"
    assert cfg.surfer_height_m == 1.75
    assert cfg.tube_safety_zoom_out == 0.30
    assert cfg.tube_max_memory_sec == 2.5

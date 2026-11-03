"""
Surf-Camera-Vision: Autonomous Virtual PTZ Cameraman & Wave Ride Tracker for Surfers.
"""

from .detector import SurferDetector, Detection
from .tracker import SurferTracker, Track
from .state_machine import SurferStateMachine, SurferState, RideEvent
from .cameraman import VirtualCameraman, CameraBox
from .scale_estimator import JointScaleVAE, MetricScaleEstimator, ScaleEstimate
from .wave_analyzer import WaveSegmentationEngine, WaveMetrics, classify_wave_size
from .pipeline import SurfTrackingPipeline, PipelineConfig
from .config import (
    load_config,
    save_config,
    cm_to_meters,
    meters_to_cm,
    parse_ft_in,
    format_ft_in,
    format_surfer_height,
    format_wave_dual_units,
    PRESETS,
    DEFAULT_CONFIG,
)

__version__ = "0.2.1"
__all__ = [
    "SurferDetector",
    "Detection",
    "SurferTracker",
    "Track",
    "SurferStateMachine",
    "SurferState",
    "RideEvent",
    "VirtualCameraman",
    "CameraBox",
    "JointScaleVAE",
    "MetricScaleEstimator",
    "ScaleEstimate",
    "WaveSegmentationEngine",
    "WaveMetrics",
    "classify_wave_size",
    "SurfTrackingPipeline",
    "PipelineConfig",
    "load_config",
    "save_config",
    "cm_to_meters",
    "meters_to_cm",
    "parse_ft_in",
    "format_ft_in",
    "format_surfer_height",
    "format_wave_dual_units",
    "PRESETS",
    "DEFAULT_CONFIG",
]

"""
Surf-Camera-Vision: Autonomous Virtual PTZ Cameraman & Wave Ride Tracker for Surfers.
"""

from .detector import SurferDetector, Detection
from .tracker import SurferTracker, Track
from .state_machine import SurferStateMachine, SurferState, RideEvent
from .cameraman import VirtualCameraman, CameraBox
from .pipeline import SurfTrackingPipeline

__version__ = "0.1.0"
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
    "SurfTrackingPipeline",
]

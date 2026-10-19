"""
Integration tests for the full SurfTrackingPipeline.
"""

from pathlib import Path
import pytest
from surf_camera.pipeline import SurfTrackingPipeline, PipelineConfig
from surf_camera.utils import generate_synthetic_surf_video


def test_pipeline_synthetic_execution(tmp_path: Path):
    video_path = tmp_path / "test_surf.mp4"
    out_dir = tmp_path / "outputs"

    # Generate a brief 90-frame synthetic surf video
    generate_synthetic_surf_video(video_path, width=640, height=360, num_frames=90, fps=30)

    config = PipelineConfig(
        model_name="yolov8n.pt",
        aspect_ratio="16:9",
        min_ride_duration_sec=0.5,
        output_resolution=(640, 360),
    )
    pipeline = SurfTrackingPipeline(config)

    results = pipeline.process_video(
        video_path=str(video_path),
        output_dir=str(out_dir),
        save_full_diagnostic=True,
        save_ride_clips=True,
    )

    assert results["total_frames"] == 90
    assert results["fps"] == 30.0
    assert results["diagnostic_video"] is not None
    assert Path(results["diagnostic_video"]).exists()
    assert "split_clips" in results
    if len(results["rides"]) > 0:
        assert len(results["split_clips"]) > 0
        assert Path(results["split_clips"][0]).exists()

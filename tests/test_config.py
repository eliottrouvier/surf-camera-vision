"""
Unit tests for configuration persistence, unit conversion helpers,
and PipelineConfig YAML loading.
"""

import pytest
from surf_camera.config import (
    parse_ft_in,
    format_ft_in,
    cm_to_meters,
    meters_to_cm,
    format_surfer_height,
    format_wave_dual_units,
    load_config,
    save_config,
    DEFAULT_CONFIG,
)
from surf_camera.pipeline import PipelineConfig


def test_unit_conversions_board_length():
    # 5'8" is approximately 1.7272 m (1.73m)
    m = parse_ft_in("5'8\"")
    assert pytest.approx(m, 0.01) == 1.73
    assert format_ft_in(1.727) == "5'8\""

    # 6'0" is 1.8288 m
    m6 = parse_ft_in("6'0\"")
    assert pytest.approx(m6, 0.01) == 1.83
    assert format_ft_in(1.829) == "6'0\""

    # 9'2" (longboard) is ~2.79 m
    m9 = parse_ft_in("9'2\"")
    assert pytest.approx(m9, 0.02) == 2.79


def test_unit_conversions_surfer_height():
    assert cm_to_meters(185.0) == 1.85
    assert meters_to_cm(1.85) == 185.0
    text = format_surfer_height(185.0)
    assert "185 cm" in text
    assert "6'1\"" in text


def test_unit_conversions_wave_dual_units():
    text = format_wave_dual_units(1.65)
    assert "1.65 m" in text
    assert "5.4 ft" in text


def test_config_load_and_save(tmp_path):
    tmp_config_file = tmp_path / "test_config.yaml"
    cfg = load_config(str(tmp_config_file))

    assert cfg["surfer"]["height_cm"] == 185.0
    assert cfg["surfboard"]["length_ft_in"] == "5'8\""

    # Modify and save
    cfg["surfer"]["height_cm"] = 180.0
    cfg["surfboard"]["length_ft_in"] = "6'0\""
    save_config(cfg, str(tmp_config_file))

    reloaded = load_config(str(tmp_config_file))
    assert reloaded["surfer"]["height_cm"] == 180.0
    assert reloaded["surfboard"]["length_ft_in"] == "6'0\""
    assert pytest.approx(reloaded["surfer"]["height_m"], 0.01) == 1.80
    assert pytest.approx(reloaded["surfboard"]["length_m"], 0.01) == 1.83


def test_pipeline_config_from_yaml(tmp_path):
    tmp_config_file = tmp_path / "test_config.yaml"
    save_config(DEFAULT_CONFIG, str(tmp_config_file))

    pipeline_cfg = PipelineConfig.from_yaml(str(tmp_config_file))
    assert pipeline_cfg.surfer_height_m == 1.85
    assert pytest.approx(pipeline_cfg.board_length_m, 0.01) == 1.73
    assert pipeline_cfg.aspect_ratio == "9:16"
    assert pipeline_cfg.enable_pose_estimation is True
    assert pipeline_cfg.split_mode == "biomechanical"

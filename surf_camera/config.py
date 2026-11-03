"""
Configuration and Unit Conversion Utilities for Surf-Camera-Vision.
Handles seamless conversion between:
- Surfer height (favoring centimeters: 185 cm <-> 6'1")
- Surfboard length (favoring feet and inches: 5'8" <-> 1.73 m)
- Wave face height (dual display: meters and feet, e.g. 1.65 m / 5.4 ft)
- Metric speeds in km/h

Also manages loading, saving, and defaults from config.yaml.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import re
import copy
import yaml


# --- UNIT CONVERSION HELPERS ---

def cm_to_meters(cm: float) -> float:
    """Converts centimeters to meters."""
    return float(cm) / 100.0


def meters_to_cm(m: float) -> float:
    """Converts meters to centimeters."""
    return float(m) * 100.0


def meters_to_feet(m: float) -> float:
    """Converts meters to decimal feet."""
    return float(m) * 3.28084


def feet_to_meters(ft: float) -> float:
    """Converts decimal feet to meters."""
    return float(ft) / 3.28084


def parse_ft_in(text: str) -> float:
    """
    Parses a string formatted like "5'8\"", "5'8", "5ft 8in", or "1.73" into meters.
    Returns meters as float.
    """
    if isinstance(text, (int, float)):
        # If already a number, if > 3.5 likely in feet, if <= 3.0 likely in meters
        val = float(text)
        return val if val <= 3.0 else feet_to_meters(val)

    text = str(text).strip()

    # Pattern: 5'8" or 5'8 or 5ft 8
    match = re.search(r"(\d+)\s*(?:'|ft|feet)\s*(\d+(?:\.\d+)?)\s*(?:\"|in|inches)?", text, re.IGNORECASE)
    if match:
        feet = float(match.group(1))
        inches = float(match.group(2))
        total_inches = feet * 12.0 + inches
        return total_inches * 0.0254

    # Pattern: just 5' or 6ft
    match_single = re.search(r"^(\d+(?:\.\d+)?)\s*(?:'|ft|feet)$", text, re.IGNORECASE)
    if match_single:
        feet = float(match_single.group(1))
        return feet * 0.3048

    # Fallback to direct float in meters
    try:
        val = float(text.replace(",", ".").replace("m", "").strip())
        return val
    except ValueError:
        return 1.73  # Standard 5'8" default


def format_ft_in(meters: float) -> str:
    """
    Converts meters to surfing standard feet and inches string, e.g. 1.73m -> "5'8\"".
    """
    total_inches = meters / 0.0254
    feet = int(total_inches // 12)
    inches = int(round(total_inches % 12))
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}'{inches}\""


def format_surfer_height(cm: float) -> str:
    """Formats surfer height in cm with ft/in equivalent, e.g. "185 cm (6'1\")"."""
    m = cm_to_meters(cm)
    ft_in = format_ft_in(m)
    return f"{int(round(cm))} cm ({ft_in})"


def format_wave_dual_units(meters: float) -> str:
    """Formats wave height with both meters and feet, e.g. "1.65 m (5.4 ft)"."""
    feet = meters_to_feet(meters)
    return f"{meters:.2f} m ({feet:.1f} ft)"


# --- DEFAULT CONFIGURATION DICTIONARY ---

DEFAULT_CONFIG: Dict[str, Any] = {
    "profile_name": "Eliott Session (Huntington Beach)",
    "surfer": {
        "height_cm": 185.0,        # 185 cm (favorisé) -> 6'1"
        "height_m": 1.85,
    },
    "surfboard": {
        "length_ft_in": "5'8\"",   # 5'8" (favorisé) -> 1.73 m
        "length_m": 1.73,
    },
    "cameraman": {
        "aspect_ratio": "9:16",     # '9:16' (Reels/TikTok) ou '16:9'
        "zoom_scale": 4.0,          # 4.0x taille surfeur
        "smoothing_factor": 0.10,   # Amorti EMA
        "lead_room_factor": 0.35,   # Anticipation règle des tiers
        "enable_split_screen": True # Double écran 1080p
    },
    "detection": {
        "model_name": "yolov8n-seg.pt",
        "conf_threshold": 0.20,
        "enable_segmentation": True,
        "enable_hud": True,
    },
    "state_machine": {
        "min_ride_duration_sec": 1.2,
        "speed_takeoff_thresh": 3.5,
        "max_merge_gap_sec": 2.0,
        "consolidate_rides": True,
    },
    "wave_metrics": {
        "enable": True,
        "units": "dual_metric_imperial",  # mètres et feet
    }
}

PRESETS = {
    "Eliott Session (Huntington Beach)": {
        "surfer_height_cm": 185.0,
        "board_length_ft_in": "5'8\"",
        "aspect_ratio": "9:16",
        "zoom_scale": 4.0,
        "lead_room_factor": 0.35,
        "smoothing_factor": 0.10,
    },
    "Shortboard Dynamique (Petites Vagues)": {
        "surfer_height_cm": 178.0,
        "board_length_ft_in": "5'6\"",
        "aspect_ratio": "9:16",
        "zoom_scale": 4.5,
        "lead_room_factor": 0.40,
        "smoothing_factor": 0.12,
    },
    "Longboard Classique (Lignes Fluides)": {
        "surfer_height_cm": 180.0,
        "board_length_ft_in": "9'2\"",
        "aspect_ratio": "16:9",
        "zoom_scale": 3.2,
        "lead_room_factor": 0.25,
        "smoothing_factor": 0.08,
    },
    "Gros Surf / Tube / Reef": {
        "surfer_height_cm": 185.0,
        "board_length_ft_in": "6'2\"",
        "aspect_ratio": "16:9",
        "zoom_scale": 3.5,
        "lead_room_factor": 0.45,
        "smoothing_factor": 0.09,
    }
}


def get_config_path(custom_path: Optional[str] = None) -> Path:
    """Resolves config.yaml path in workspace root."""
    if custom_path:
        return Path(custom_path)
    root = Path(__file__).resolve().parent.parent
    return root / "config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration from config.yaml, merging safely with DEFAULT_CONFIG.
    Creates config.yaml with defaults if it doesn't exist.
    """
    path = get_config_path(config_path)
    if not path.exists():
        save_config(copy.deepcopy(DEFAULT_CONFIG), path)
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture de {path}: {e}. Utilisation des valeurs par défaut.")
        return copy.deepcopy(DEFAULT_CONFIG)

    # Merge user_cfg onto DEFAULT_CONFIG using deepcopy to avoid mutating globals
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in user_cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k].update(v)
        else:
            merged[k] = v

    # Ensure consistency between cm <-> m and ft_in <-> m
    if "surfer" in merged:
        h_cm = merged["surfer"].get("height_cm", 185.0)
        merged["surfer"]["height_m"] = cm_to_meters(h_cm)
    if "surfboard" in merged:
        b_str = merged["surfboard"].get("length_ft_in", "5'8\"")
        merged["surfboard"]["length_m"] = parse_ft_in(b_str)

    return merged


def save_config(config_data: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """Saves the configuration dictionary to config.yaml."""
    path = get_config_path(config_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"❌ Impossible d'enregistrer {path}: {e}")
        return False

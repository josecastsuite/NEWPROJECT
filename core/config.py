"""Runtime configuration helpers for JoséCast.

Configuration files are optional; missing keys fall back to the built-in
constants so the application keeps working out of the box.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AnimationConfig:
    """User-tunable limits for the flow animator."""

    max_anim_cells: int = 120_000
    max_frames: int = 1350
    min_fill_frames: int = 1200
    phi_sigma: float = 1.2
    decimate_target: float = 0.5
    max_streamlines: int = 20
    max_steps: int = 2000
    cfl_fraction: float = 0.5
    pore_rise_speed_m_s: float = 0.05  # upward pore drift against gravity

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AnimationConfig":
        if not data:
            return cls()
        # Allow both snake_case and the original CONSTANT-style keys.
        mapped: Dict[str, Any] = {}
        for k, v in data.items():
            key = k.lower().replace("-", "_")
            mapped[key] = v
        return cls(
            max_anim_cells=int(mapped.get("max_anim_cells", cls.max_anim_cells)),
            max_frames=int(mapped.get("max_frames", cls.max_frames)),
            min_fill_frames=int(mapped.get("min_fill_frames", cls.min_fill_frames)),
            phi_sigma=float(mapped.get("phi_sigma", cls.phi_sigma)),
            decimate_target=float(mapped.get("decimate_target", cls.decimate_target)),
            max_streamlines=int(mapped.get("max_streamlines", cls.max_streamlines)),
            max_steps=int(mapped.get("max_steps", cls.max_steps)),
            cfl_fraction=float(mapped.get("cfl_fraction", cls.cfl_fraction)),
            pore_rise_speed_m_s=float(
                mapped.get("pore_rise_speed_m_s", cls.pore_rise_speed_m_s)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_anim_cells": self.max_anim_cells,
            "max_frames": self.max_frames,
            "min_fill_frames": self.min_fill_frames,
            "phi_sigma": self.phi_sigma,
            "decimate_target": self.decimate_target,
            "max_streamlines": self.max_streamlines,
            "max_steps": self.max_steps,
            "cfl_fraction": self.cfl_fraction,
            "pore_rise_speed_m_s": self.pore_rise_speed_m_s,
        }


def _default_config_path() -> Path:
    env = os.environ.get("JOSECAST_ANIMATION_CONFIG")
    if env:
        return Path(env)
    # Prefer repo-local config/animation_config.json; fall back to user config.
    repo = Path(__file__).resolve().parent.parent / "config"
    if (repo / "animation_config.json").exists():
        return repo / "animation_config.json"
    home = Path.home() / ".config" / "josecast"
    return home / "animation_config.json"


def load_animation_config(path: Optional[Path] = None) -> AnimationConfig:
    """Load animation limits from a JSON file or return defaults."""
    cfg_path = path or _default_config_path()
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return AnimationConfig.from_dict(data)
        except Exception:
            return AnimationConfig()
    return AnimationConfig()


def save_animation_config(
    config: AnimationConfig, path: Optional[Path] = None
) -> Path:
    """Persist animation limits to a JSON file."""
    cfg_path = path or _default_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return cfg_path

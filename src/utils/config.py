"""Konfigürasyon yardımcıları — YAML dosyalarını yükle, proje root yolunu çöz."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """EUYZ_RL_Dashboard kök dizini (bu dosyaya göre)."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """config/<name>.yaml veya tam isim verilirse onu yükler."""
    path = config_dir() / name
    if not path.exists() and not name.endswith(".yaml"):
        path = config_dir() / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs() -> dict[str, dict]:
    """Tüm config dosyalarını tek sözlükte döndür."""
    cfg = {}
    for name in ["paths", "battery", "supercap", "converter", "reward", "training"]:
        cfg[name] = load_yaml(name)
    return cfg


def dataset_root() -> Path:
    """Veri seti kök yolu."""
    return Path(load_yaml("paths")["dataset_root"])

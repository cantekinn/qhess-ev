"""Kompakt EV longitudinal araç dinamiği.

P_pack(t) hesabı (motoring):
    F_total = m·a + m·g·C_r·cos(θ) + 0.5·ρ·C_d·A·v² + m·g·sin(θ)
    P_wheel = F_total · v
    P_pack  = P_wheel / η_drive            (motoring, P>0)
    P_pack  = P_wheel · η_regen            (regen,    P<0)

UI'da P_pack kW olarak gösterilir. Hücre fiziği için P_cell = P_pack / pack.factor.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
import pandas as pd


@dataclass
class VehicleParams:
    name: str = "Kompakt EV"
    mass_kg: float = 1300.0
    frontal_area_m2: float = 2.2
    drag_coef_cd: float = 0.30
    rolling_coef: float = 0.012
    drivetrain_eff: float = 0.92
    regen_eff: float = 0.65
    gravity: float = 9.81
    air_density: float = 1.225
    peak_power_kW: float = 45.0
    max_speed_kmh: float = 160.0
    pack_factor: float = 5500.0      # cell-level downscale

    @classmethod
    def from_config(cls, cfg_vehicle: dict) -> "VehicleParams":
        v = cfg_vehicle["vehicle"]
        return cls(
            name=v["name"],
            mass_kg=v["mass_kg"],
            frontal_area_m2=v["frontal_area_m2"],
            drag_coef_cd=v["drag_coef_cd"],
            rolling_coef=v["rolling_coef"],
            drivetrain_eff=v["drivetrain_eff"],
            regen_eff=v["regen_eff"],
            gravity=v["gravity"],
            air_density=v["air_density"],
            peak_power_kW=v["peak_power_kW"],
            max_speed_kmh=v["max_speed_kmh"],
            pack_factor=cfg_vehicle["pack"]["factor"],
        )


def power_demand_W(
    speed_mps: np.ndarray,
    accel_mps2: np.ndarray,
    p: VehicleParams,
    slope_deg: float = 0.0,
) -> np.ndarray:
    """Hız ve ivme profilinden pack-seviyesi güç talebini hesapla (W). +deşarj, -rejen."""
    v = np.asarray(speed_mps, dtype=float)
    a = np.asarray(accel_mps2, dtype=float)
    theta = radians(slope_deg)

    F_inertia = p.mass_kg * a
    F_rolling = p.mass_kg * p.gravity * p.rolling_coef * cos(theta)
    F_drag = 0.5 * p.air_density * p.drag_coef_cd * p.frontal_area_m2 * v * v
    F_grade = p.mass_kg * p.gravity * sin(theta)

    F_total = F_inertia + F_rolling + F_drag + F_grade
    P_wheel = F_total * v

    P_pack = np.where(P_wheel >= 0, P_wheel / p.drivetrain_eff, P_wheel * p.regen_eff)
    # Peak güç sınırı
    P_max = p.peak_power_kW * 1000.0
    P_pack = np.clip(P_pack, -P_max, P_max)
    return P_pack


def assemble_profile(
    time_s: np.ndarray,
    speed_kmh: np.ndarray,
    p: VehicleParams,
    slope_deg: float = 0.0,
) -> pd.DataFrame:
    """Zaman + hız profili → güç DataFrame."""
    dt = float(np.mean(np.diff(time_s))) if len(time_s) > 1 else 1.0
    v_mps = speed_kmh / 3.6
    a_mps2 = np.gradient(v_mps, dt)
    # Düşük geçiren filtre (acceleration smoothing) — yumuşak görselleştirme için
    if len(a_mps2) >= 5:
        kernel = np.ones(5) / 5.0
        a_mps2 = np.convolve(a_mps2, kernel, mode="same")

    P_pack = power_demand_W(v_mps, a_mps2, p, slope_deg=slope_deg)
    P_cell = P_pack / p.pack_factor
    regen = P_pack < 0

    return pd.DataFrame(
        {
            "time_s": time_s,
            "speed_kmh": speed_kmh,
            "speed_mps": v_mps,
            "accel_mps2": a_mps2,
            "p_pack_W": P_pack,
            "p_pack_kW": P_pack / 1000.0,
            "p_cell_W": P_cell,
            "p_load_W": np.abs(P_cell),       # env için magnitude (motoring + regen)
            "regen_flag": regen,
            "slope_deg": slope_deg,
        }
    )

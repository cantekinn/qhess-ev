"""Test ve eğitim senaryoları — hız profillerinden araç yük profili üretir.

5 senaryo:
  - city_stop_go
  - overtaking
  - highway_cruise
  - mixed_urban
  - mountain_climb

Tümü `make_scenario(name)` ile çağrılır, `pd.DataFrame(time_s, speed_kmh, p_*, regen_flag)` döner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.vehicle import VehicleParams, assemble_profile


def _smooth_step(t: np.ndarray, t0: float, t1: float, v0: float, v1: float) -> np.ndarray:
    """t0..t1 arasında v0'dan v1'e yumuşak (cosine ease) geçiş."""
    out = np.where(t < t0, v0, v1)
    mask = (t >= t0) & (t <= t1)
    if mask.any() and t1 > t0:
        x = (t[mask] - t0) / (t1 - t0)
        ease = 0.5 - 0.5 * np.cos(np.pi * x)
        out = out.astype(float)
        out[mask] = v0 + (v1 - v0) * ease
    return out


def _speed_city_stop_go(cfg: dict, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    duration = float(cfg["duration_s"])
    v_max = float(cfg["v_max_kmh"])
    t_a = cfg["accel_phase_s"]
    t_c = cfg["cruise_phase_s"]
    t_b = cfg["brake_phase_s"]
    t_i = cfg["idle_phase_s"]
    cycle = t_a + t_c + t_b + t_i
    t = np.arange(0, duration + dt, dt)
    v = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = ti % cycle
        if phase < t_a:
            v[i] = v_max * phase / t_a
        elif phase < t_a + t_c:
            v[i] = v_max
        elif phase < t_a + t_c + t_b:
            v[i] = v_max * (1.0 - (phase - t_a - t_c) / t_b)
        else:
            v[i] = 0.0
    return t, v


def _speed_overtaking(cfg: dict, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    duration = float(cfg["duration_s"])
    t = np.arange(0, duration + dt, dt)
    v_cruise = float(cfg["v_cruise_kmh"])
    v_peak = float(cfg["v_peak_kmh"])
    v_after = float(cfg["v_after_kmh"])
    ts = float(cfg["peak_start_s"])
    ta = float(cfg["peak_accel_s"])
    th = float(cfg["peak_hold_s"])
    tb = float(cfg["decel_back_s"])
    v = np.full_like(t, v_cruise, dtype=float)
    # Cruise → peak
    v = _smooth_step(t, ts, ts + ta, v_cruise, v_peak)
    # Peak hold
    mask = (t >= ts + ta) & (t < ts + ta + th)
    v = np.where(mask, v_peak, v)
    # Peak → after
    v = np.where(
        t >= ts + ta + th,
        _smooth_step(t, ts + ta + th, ts + ta + th + tb, v_peak, v_after),
        v,
    )
    # Sabit after
    v = np.where(t >= ts + ta + th + tb, v_after, v)
    # Önce cruise sabit kalsın
    v = np.where(t < ts, v_cruise, v)
    return t, v


def _speed_cruise(cfg: dict, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    duration = float(cfg["duration_s"])
    t = np.arange(0, duration + dt, dt)
    v_target = float(cfg["v_kmh"])
    ramp = float(cfg.get("ramp_up_s", 5.0))
    v = _smooth_step(t, 0.0, ramp, 0.0, v_target)
    return t, v


def _speed_mixed(cfg: dict, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Segment segment birleştir."""
    ts: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    t0 = 0.0
    v_prev = 0.0
    for seg in cfg["segments"]:
        dur = float(seg["dur_s"])
        if seg["type"] == "stop_go":
            sub = {
                "duration_s": dur,
                "v_max_kmh": seg["v_max_kmh"],
                "accel_phase_s": 6, "cruise_phase_s": 6, "brake_phase_s": 4, "idle_phase_s": 4,
            }
            t_s, v_s = _speed_city_stop_go(sub, dt=dt)
        elif seg["type"] == "cruise":
            sub = {"duration_s": dur, "v_kmh": seg["v_kmh"], "ramp_up_s": 5.0}
            t_s, v_s = _speed_cruise(sub, dt=dt)
        elif seg["type"] == "overtaking":
            sub = {
                "duration_s": dur,
                "v_cruise_kmh": v_prev if v_prev > 0 else seg.get("v_after_kmh", 60.0),
                "v_peak_kmh": seg["v_peak_kmh"],
                "v_after_kmh": seg["v_after_kmh"],
                "peak_start_s": 4, "peak_accel_s": 5, "peak_hold_s": 6, "decel_back_s": 5,
            }
            t_s, v_s = _speed_overtaking(sub, dt=dt)
        else:
            t_s = np.arange(0, dur + dt, dt)
            v_s = np.zeros_like(t_s)
        # İlk hız önceki son hızla başlasın
        if vs:
            shift = vs[-1][-1] - v_s[0]
            # smooth tek-step: ilk 3 adım blend
            blend = min(3, len(v_s))
            if blend > 1:
                w = np.linspace(1.0, 0.0, blend)
                v_s = v_s.copy()
                v_s[:blend] = w * (v_s[:blend] + shift) + (1 - w) * v_s[:blend]
        ts.append(t_s + t0)
        vs.append(v_s)
        t0 += t_s[-1] + dt
        v_prev = v_s[-1]
    t = np.concatenate(ts)
    v = np.concatenate(vs)
    return t, v


_SPEED_FNS = {
    "stop_go": _speed_city_stop_go,
    "overtaking": _speed_overtaking,
    "cruise": _speed_cruise,
    "mixed": _speed_mixed,
}


def list_scenarios() -> list[str]:
    cfg = load_yaml("scenarios")
    return list(cfg.keys())


def scenario_meta(name: str) -> dict:
    cfg = load_yaml("scenarios")[name]
    return {"title": cfg["title"], "description": cfg["description"], "duration_s": cfg["duration_s"]}


def make_scenario(name: str, vehicle: VehicleParams, dt: float = 1.0) -> pd.DataFrame:
    """Senaryo adı → araç güç profili (pd.DataFrame)."""
    cfg = load_yaml("scenarios")[name]
    pattern = cfg["pattern"]
    fn = _SPEED_FNS.get(pattern)
    if fn is None:
        raise ValueError(f"Bilinmeyen pattern: {pattern}")
    t, v_kmh = fn(cfg, dt=dt)
    df = assemble_profile(t, v_kmh, vehicle, slope_deg=float(cfg.get("slope_deg", 0.0)))
    df["scenario"] = name
    return df

"""KPI hesaplamaları — bir episode history'sinden (list[dict]) çıkar.

Plan Rev 1 ile birlikte:
  - compute_kpis: temel metrikler
  - success_score: senaryo bazlı 0-100 başarı yüzdesi
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kpis(history: list[dict], dt_s: float = 1.0,
                 i_bat_max: float = 2.2) -> dict:
    if not history:
        return {}
    df = pd.DataFrame(history)
    p_load = df["p_load"].values
    p_supplied = df["p_supplied"].values
    p_loss = df["p_loss_total"].values
    i_bat = df["i_bat"].values

    mask = p_load > 1e-6
    if mask.any():
        supply_ratio = float(np.mean(np.clip(p_supplied[mask] / p_load[mask], 0, 1.5)))
    else:
        supply_ratio = 1.0

    # Süperkap kullanım: action >= 3 olan adımların oranı (sadece motoring)
    motoring = ~df.get("regen", pd.Series([False] * len(df))).astype(bool).values
    high_demand = df["demand"].values >= 3
    low_demand = df["demand"].values <= 1
    if (high_demand & motoring).any():
        sc_usage_peak = float(np.mean(df["action"].values[high_demand & motoring] >= 3))
    else:
        sc_usage_peak = 0.0
    if (low_demand & motoring).any():
        bat_usage_low = float(np.mean(df["action"].values[low_demand & motoring] <= 1))
    else:
        bat_usage_low = 0.0
    # Regen → süperkap akımı negatif (şarj)
    if "regen" in df.columns:
        regen_steps = df["regen"].astype(bool).values
        if regen_steps.any():
            sc_regen_charge = float(np.mean(df["i_sc"].values[regen_steps] < 0))
        else:
            sc_regen_charge = 1.0  # rejen yoksa nötr
    else:
        sc_regen_charge = 1.0

    n = max(len(df), 1)
    return {
        "total_reward": float(df["reward"].sum()),
        "mean_reward": float(df["reward"].mean()),
        "cumulative_loss_Wh": float(np.sum(p_loss) * dt_s / 3600.0),
        "peak_i_bat_A": float(np.max(np.abs(i_bat))),
        "rms_i_bat_A": float(np.sqrt(np.mean(i_bat ** 2))),
        "ah_throughput_bat": float(np.sum(np.abs(i_bat)) * dt_s / 3600.0),
        "violations": int(df["violation"].sum()),
        "violation_rate": float(df["violation"].sum() / n),
        "mean_supply_ratio": supply_ratio,
        "soc_bat_final": float(df["soc_bat"].iloc[-1]),
        "soc_sc_final": float(df["soc_sc"].iloc[-1]),
        "supercap_usage_peak": sc_usage_peak,
        "battery_usage_low": bat_usage_low,
        "supercap_regen_charge_rate": sc_regen_charge,
        "n_steps": n,
        "peak_i_bat_norm": float(np.max(np.abs(i_bat)) / max(i_bat_max, 1e-6)),
    }


def success_score(kpis: dict, i_bat_max: float = 2.2) -> dict:
    """Başarı skorunu 0-100 arasında hesapla.

    Ağırlıklar:
      0.30 — pik batarya akımı I_max'in %80'i altında
      0.25 — yüksek talepte süperkap kullanım oranı
      0.20 — rejen sırasında süperkap şarj oranı
      0.15 — kısıt ihlali yok
      0.10 — talep karşılanma oranı
    """
    if not kpis:
        return {"score": 0.0, "components": {}}

    peak_norm = kpis.get("peak_i_bat_norm", 1.0)
    # Smooth: 0.0 (peak>=I_max) → 1.0 (peak<=0.5*I_max)
    s_peak = float(np.clip(1.0 - max(0.0, peak_norm - 0.5) / 0.5, 0.0, 1.0))
    s_sc_peak = float(np.clip(kpis.get("supercap_usage_peak", 0.0), 0.0, 1.0))
    s_sc_regen = float(np.clip(kpis.get("supercap_regen_charge_rate", 1.0), 0.0, 1.0))
    s_viol = float(np.clip(1.0 - kpis.get("violation_rate", 0.0), 0.0, 1.0))
    s_supply = float(np.clip(kpis.get("mean_supply_ratio", 0.0), 0.0, 1.0))

    score = (
        0.30 * s_peak
        + 0.25 * s_sc_peak
        + 0.20 * s_sc_regen
        + 0.15 * s_viol
        + 0.10 * s_supply
    ) * 100.0

    return {
        "score": float(score),
        "components": {
            "peak_protection": s_peak,
            "supercap_at_peak": s_sc_peak,
            "supercap_regen": s_sc_regen,
            "no_violation": s_viol,
            "supply_quality": s_supply,
        },
    }


def kpi_table(results: dict[str, list[dict]], dt_s: float = 1.0,
              i_bat_max: float = 2.2) -> pd.DataFrame:
    rows = []
    for name, h in results.items():
        k = compute_kpis(h, dt_s=dt_s, i_bat_max=i_bat_max)
        k["policy"] = name
        rows.append(k)
    df = pd.DataFrame(rows)
    cols = ["policy"] + [c for c in df.columns if c != "policy"]
    return df[cols]

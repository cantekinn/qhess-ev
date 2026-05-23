"""Süperkapasitör klasik RC + ESR modeli.

Denklemler:
    dV_sc/dt = -I_sc / C_sc          # I_sc > 0 deşarj
    V_term   = V_sc - I_sc * ESR_sc  # terminal gerilim
    E_sc     = 0.5 * C_sc * V_sc^2   # depolanan enerji

SOC tanımı:
  - "voltage" : SOC = (V - V_min) / (V_max - V_min)
  - "energy"  : SOC = (V^2 - V_min^2) / (V_max^2 - V_min^2)   # daha doğru, %75 enerji V>V_max/2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SupercapParams:
    C_F: float = 100.0
    ESR_ohm: float = 0.012
    V_max: float = 2.7
    V_min: float = 1.35
    I_max_A: float = 50.0
    soc_mode: str = "energy"   # "voltage" veya "energy"
    soc_min: float = 0.05
    soc_max: float = 1.0


@dataclass
class SupercapState:
    v_sc: float = 2.7


class SupercapModel:
    def __init__(self, params: SupercapParams, dt_s: float = 1.0):
        self.p = params
        self.dt = float(dt_s)
        self.state = SupercapState(v_sc=params.V_max)

    def reset(self, soc_init: float = 1.0):
        # SOC -> V_sc
        self.state.v_sc = self.soc_to_v(float(np.clip(soc_init, 0.0, 1.0)))

    # --- SOC <-> V dönüşümü ----------------------------------------------------
    def v_to_soc(self, v: float | None = None) -> float:
        v = self.state.v_sc if v is None else v
        if self.p.soc_mode == "voltage":
            return float((v - self.p.V_min) / (self.p.V_max - self.p.V_min))
        # energy
        num = v * v - self.p.V_min * self.p.V_min
        den = self.p.V_max * self.p.V_max - self.p.V_min * self.p.V_min
        return float(num / den)

    def soc_to_v(self, soc: float) -> float:
        soc = float(np.clip(soc, 0.0, 1.0))
        if self.p.soc_mode == "voltage":
            return self.p.V_min + soc * (self.p.V_max - self.p.V_min)
        v2 = self.p.V_min ** 2 + soc * (self.p.V_max ** 2 - self.p.V_min ** 2)
        return float(np.sqrt(max(v2, 0.0)))

    @property
    def soc(self) -> float:
        return self.v_to_soc()

    @property
    def energy_Wh(self) -> float:
        return 0.5 * self.p.C_F * self.state.v_sc ** 2 / 3600.0

    def step(self, i_sc_A: float) -> dict:
        """i_sc > 0 deşarj. Coulomb counting + ESR."""
        i = float(i_sc_A)
        # V_sc güncelle: dV = -I*dt/C
        self.state.v_sc -= i * self.dt / self.p.C_F
        self.state.v_sc = float(np.clip(self.state.v_sc, 0.0, self.p.V_max))
        v_term = self.state.v_sc - i * self.p.ESR_ohm
        p_loss = (i ** 2) * self.p.ESR_ohm
        return {
            "v_sc": self.state.v_sc,
            "v_term": v_term,
            "p_loss_esr": p_loss,
            "soc": self.soc,
            "i_sc": i,
            "v_min_violation": self.state.v_sc < self.p.V_min,
            "i_violation": abs(i) > self.p.I_max_A,
        }


def make_supercap(cfg_supercap: dict, dt_s: float = 1.0) -> SupercapModel:
    c = cfg_supercap
    p = SupercapParams(
        C_F=c["cell"]["C_F"],
        ESR_ohm=c["cell"]["ESR_ohm"],
        V_max=c["cell"]["V_max"],
        V_min=c["cell"]["V_min"],
        I_max_A=c["cell"]["I_max_A"],
        soc_mode=c["soc_definition"]["mode"],
        soc_min=c["soc_definition"]["soc_min"],
        soc_max=c["soc_definition"]["soc_max"],
    )
    return SupercapModel(p, dt_s=dt_s)

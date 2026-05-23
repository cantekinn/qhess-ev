"""Batarya 2RC Thevenin Equivalent Circuit Model.

Denklemler:
    V_bat(t) = OCV(SOC(t)) - I_bat(t) * R0 - V_RC1(t) - V_RC2(t)
    dV_RC1/dt = -V_RC1/(R1*C1) + I_bat/C1
    dV_RC2/dt = -V_RC2/(R2*C2) + I_bat/C2
    dSOC/dt  = -I_bat / (3600 * Q_nom)        # Coulomb counting

I_bat: deşarjda pozitif kabul ediliyor (akım hücreden dışarı). Bu konvansiyon
ile V_bat = OCV - IR olur (Pozar/Plett konvansiyonu).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.loader import OcvSocCurve


@dataclass
class BatteryParams:
    Q_nom_Ah: float = 1.1
    R0_ohm: float = 0.085
    R1_ohm: float = 0.045
    C1_F: float = 2200.0
    R2_ohm: float = 0.030
    C2_F: float = 18000.0
    V_max: float = 4.2
    V_min: float = 3.0
    I_max_dis_A: float = 2.2
    I_max_chg_A: float = 1.1
    soc_min: float = 0.10
    soc_max: float = 0.95


@dataclass
class BatteryState:
    soc: float = 0.9
    v_rc1: float = 0.0
    v_rc2: float = 0.0


class BatteryECM:
    """2RC Thevenin batarya modeli (forward-Euler entegrasyon)."""

    def __init__(self, params: BatteryParams, ocv_curve: OcvSocCurve, dt_s: float = 1.0):
        self.p = params
        self.ocv = ocv_curve
        self.dt = float(dt_s)
        self.state = BatteryState(soc=0.9)
        # Ah-throughput (cycle life proxy)
        self.ah_throughput = 0.0
        self.peak_current = 0.0

    def reset(self, soc_init: float = 0.9):
        self.state = BatteryState(soc=float(np.clip(soc_init, 0.0, 1.0)))
        self.ah_throughput = 0.0
        self.peak_current = 0.0

    def terminal_voltage(self) -> float:
        """V_bat = OCV(SOC) - V_RC1 - V_RC2 (akım R0 üzerinden ayrıca düşer)."""
        return float(self.ocv(self.state.soc)) - self.state.v_rc1 - self.state.v_rc2

    def step(self, i_bat_A: float) -> dict:
        """Bir adım simüle et. i_bat > 0 deşarj, i_bat < 0 şarj.

        Döndürür: dict( v_bat, v_oc, soc, p_loss_ohmic, p_terminal, kısıt bayrakları )
        """
        i = float(i_bat_A)
        dt = self.dt
        tau1 = self.p.R1_ohm * self.p.C1_F
        tau2 = self.p.R2_ohm * self.p.C2_F
        # Exact exponential update — Plett çözümü:
        #   V_RC[k+1] = V_RC[k]*exp(-dt/tau) + i*R*(1 - exp(-dt/tau))
        a1 = float(np.exp(-dt / tau1))
        a2 = float(np.exp(-dt / tau2))
        self.state.v_rc1 = self.state.v_rc1 * a1 + i * self.p.R1_ohm * (1.0 - a1)
        self.state.v_rc2 = self.state.v_rc2 * a2 + i * self.p.R2_ohm * (1.0 - a2)

        # SOC (Coulomb counting)
        self.state.soc -= i * dt / (3600.0 * self.p.Q_nom_Ah)
        self.state.soc = float(np.clip(self.state.soc, 0.0, 1.0))

        # Terminal voltaj
        v_oc = float(self.ocv(self.state.soc))
        v_bat = v_oc - i * self.p.R0_ohm - self.state.v_rc1 - self.state.v_rc2

        # Kayıplar
        p_loss_R0 = (i ** 2) * self.p.R0_ohm
        p_loss_R1 = (self.state.v_rc1 ** 2) / max(self.p.R1_ohm, 1e-9)
        p_loss_R2 = (self.state.v_rc2 ** 2) / max(self.p.R2_ohm, 1e-9)

        # Cycle life proxy
        self.ah_throughput += abs(i) * dt / 3600.0
        self.peak_current = max(self.peak_current, abs(i))

        return {
            "v_bat": v_bat,
            "v_oc": v_oc,
            "v_rc1": self.state.v_rc1,
            "v_rc2": self.state.v_rc2,
            "soc": self.state.soc,
            "p_loss_ohmic": p_loss_R0 + p_loss_R1 + p_loss_R2,
            "p_terminal": v_bat * i,
            "i_bat": i,
            "v_min_violation": v_bat < self.p.V_min,
            "v_max_violation": v_bat > self.p.V_max,
            "soc_min_violation": self.state.soc < self.p.soc_min,
            "soc_max_violation": self.state.soc > self.p.soc_max,
        }


def make_battery(cfg_battery: dict, ocv_curve: OcvSocCurve, dt_s: float = 1.0) -> BatteryECM:
    """`config/battery.yaml` sözlüğünden BatteryECM üret."""
    c = cfg_battery
    p = BatteryParams(
        Q_nom_Ah=c["cell"]["Q_nom_Ah"],
        R0_ohm=c["ecm"]["R0_ohm"],
        R1_ohm=c["ecm"]["R1_ohm"],
        C1_F=c["ecm"]["C1_F"],
        R2_ohm=c["ecm"]["R2_ohm"],
        C2_F=c["ecm"]["C2_F"],
        V_max=c["cell"]["V_max"],
        V_min=c["cell"]["V_min"],
        I_max_dis_A=c["cell"]["I_max_dis_A"],
        I_max_chg_A=c["cell"]["I_max_chg_A"],
        soc_min=c["soc_limits"]["soc_min"],
        soc_max=c["soc_limits"]["soc_max"],
    )
    return BatteryECM(p, ocv_curve, dt_s=dt_s)

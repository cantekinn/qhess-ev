"""Bidirectional DC-DC converter — basit yük-bağımlı verim modeli.

eta(P) = eta_nominal - (eta_nominal - eta_min) * exp(-|P|/P_rated)

  P=0    -> eta_min
  P>>P_r -> eta_nominal

Konvansiyon:
  - "discharge" yönde: P_out_bus = P_cell * eta   (kayıp düşer)
  - "charge"    yönde: P_out_cell = P_bus * eta   (yine kayıp düşer)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConverterParams:
    eta_nominal: float = 0.94
    eta_min: float = 0.85
    P_rated_W: float = 25.0


class Converter:
    def __init__(self, params: ConverterParams):
        self.p = params

    def eta(self, p_W: float) -> float:
        from math import exp
        x = abs(p_W) / max(self.p.P_rated_W, 1e-6)
        # eta_min düşük yükte; eta_nominal asıl rate civarında
        return self.p.eta_min + (self.p.eta_nominal - self.p.eta_min) * (1 - exp(-x * 2.0))

    def to_bus(self, p_cell_W: float) -> tuple[float, float]:
        """Hücreden bara'ya transfer. Döndürür (P_bus, P_loss)."""
        e = self.eta(p_cell_W)
        if p_cell_W >= 0:  # deşarj
            p_bus = p_cell_W * e
            p_loss = p_cell_W - p_bus
        else:               # şarj — bus'tan hücreye negatif
            p_bus = p_cell_W / e  # hücreye gitmesi için daha fazla bara gücü gerekir
            p_loss = p_bus - p_cell_W
        return p_bus, abs(p_loss)

    def to_cell(self, p_bus_W: float) -> tuple[float, float]:
        """Bara'dan hücreye transfer (referans). Döndürür (P_cell, P_loss)."""
        e = self.eta(p_bus_W)
        if p_bus_W >= 0:
            p_cell = p_bus_W * e
            p_loss = p_bus_W - p_cell
        else:
            p_cell = p_bus_W / e
            p_loss = p_cell - p_bus_W
        return p_cell, abs(p_loss)


def make_converters(cfg_conv: dict) -> tuple[Converter, Converter]:
    """battery_converter ve supercap_converter olarak iki ayrı Converter döndür."""
    b = cfg_conv["battery_converter"]
    s = cfg_conv["supercap_converter"]
    bat = Converter(ConverterParams(eta_nominal=b["eta_nominal"], eta_min=b["eta_min"], P_rated_W=b["P_rated_W"]))
    sc = Converter(ConverterParams(eta_nominal=s["eta_nominal"], eta_min=s["eta_min"], P_rated_W=s["P_rated_W"]))
    return bat, sc

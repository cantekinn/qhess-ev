"""Plotly figür üreticileri — RL test/sim ve veri keşfi için."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def time_series(df: pd.DataFrame, x: str, ys: list[str], title: str = "") -> go.Figure:
    fig = go.Figure()
    for y in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title=x, hovermode="x unified", height=380)
    return fig


def histogram(values, title: str = "", nbins: int = 30) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=nbins)])
    fig.update_layout(title=title, height=320)
    return fig


def ocv_soc_curve(soc: np.ndarray, ocv: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Scatter(x=soc, y=ocv, mode="lines+markers", name="OCV(SOC)"))
    fig.update_layout(
        title="OCV–SOC Eğrisi (CS2)", xaxis_title="SOC [-]", yaxis_title="OCV [V]", height=380
    )
    return fig

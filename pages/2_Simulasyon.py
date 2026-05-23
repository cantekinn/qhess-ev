"""Sayfa 2 — Simülasyon (animasyonlu).

Eğitilmiş Q-Learning ajanı seçilen senaryoda kare-kare canlı oynatır.
Anlık: araç SVG + hız gauge + güç paylaşım donut + 2 SOC bar
     + güç akış oku + mod pill + 3'lü zaman serisi + başarı %.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.agents.q_learning import QLearningAgent
from src.agents.runner import evaluate_policy
from src.data.loader import load_ocv_soc_curve
from src.env.hess_env import make_env_from_config
from src.metrics.kpi import compute_kpis, success_score
from src.scenarios import list_scenarios, make_scenario, scenario_meta
from src.utils.config import load_all_configs, load_yaml, project_root
from src.vehicle import VehicleParams

st.set_page_config(page_title="Simülasyon — HESS", page_icon="🚗", layout="wide",
                   initial_sidebar_state="expanded", menu_items={})
st.markdown(
    """<style>
#MainMenu, footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stSidebarNav"] ul li:first-child a span { display: none; }
[data-testid="stSidebarNav"] ul li:first-child a::after {
    content: "Ana Sayfa"; color: inherit; font: inherit;
}
.block-container {padding-top: 1rem; max-width: 1500px;}
h1 { color: #4FC3F7; font-weight: 800; }
.mode-pill {
    display:inline-block; padding: 8px 18px; border-radius: 999px;
    font-weight: 700; font-size: 1.05rem; color: white;
}
.mode-cruise { background: #2E7D32; }
.mode-peak { background: #E53935; }
.mode-regen { background: #1976D2; }
.mode-limit { background: #F57C00; }
.mode-idle { background: #455A64; }
.big-num { font-size: 2.6rem; font-weight: 900; color: #4FC3F7; }
.label { color: #90A4AE; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
.success-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.4rem;
}
.bg-good { background: linear-gradient(135deg, #2E7D32, #66BB6A); color: white; }
.bg-mid  { background: linear-gradient(135deg, #F57C00, #FFB74D); color: white; }
.bg-bad  { background: linear-gradient(135deg, #C62828, #EF5350); color: white; }
</style>""", unsafe_allow_html=True,
)

st.title("HESS Simülasyon ve Test")

cfg = load_all_configs()
cfg["vehicle"] = load_yaml("vehicle")
ocv = load_ocv_soc_curve()
veh = VehicleParams.from_config(cfg["vehicle"])

# ============================================================ Üst seçim bar
top_l, top_m = st.columns([2, 2])

with top_l:
    scen_names = list_scenarios()
    sc_idx = st.selectbox(
        "Senaryo",
        range(len(scen_names)),
        format_func=lambda i: scenario_meta(scen_names[i])["title"],
        index=1,
    )
    sc_name = scen_names[sc_idx]
    st.caption(scenario_meta(sc_name)["description"])

with top_m:
    models_dir = project_root() / cfg["paths"]["models_dir"]
    models = sorted(models_dir.glob("ql_*.pkl"))
    if not models:
        st.warning("Henüz eğitilmiş model yok. Önce Eğitim sayfasında bir model oluştur.")
        st.stop()
    model_idx = st.selectbox(
        "Eğitilmiş ajan",
        range(len(models)),
        format_func=lambda i: models[i].name,
        index=len(models) - 1,
    )
    model_path = models[model_idx]

# ============================================================ Simülasyon koşumu (önbelleğe alınır)
sim_key = (sc_name, model_path.name)
need_recompute = (
    "sim_key" not in st.session_state
    or st.session_state.get("sim_key") != sim_key
    or "sim_df" not in st.session_state
)

if need_recompute or st.button("Simülasyonu Yeniden Hesapla"):
    with st.spinner("Senaryo koşuluyor..."):
        profile = make_scenario(sc_name, veh)
        env = make_env_from_config(cfg, ocv, profile_df=profile, seed=42)
        env.cfg.max_steps = len(profile) - 1

        agent = QLearningAgent(
            n_actions=env.n_actions,
            bins_p=cfg["training"]["q_learning"]["state_bins"]["p_load"],
            bins_soc_bat=cfg["training"]["q_learning"]["state_bins"]["soc_bat"],
            bins_soc_sc=cfg["training"]["q_learning"]["state_bins"]["soc_sc"],
        )
        agent.load(str(model_path))
        agent.epsilon = 0.0

        history = evaluate_policy(env, agent, start_idx=0, max_steps=env.cfg.max_steps,
                                  soc_bat_init=0.85, soc_sc_init=0.70)
        df = pd.DataFrame(history)
        df["t_s"] = np.arange(len(df)) * env.cfg.dt_s
        df["speed_kmh"] = profile["speed_kmh"].iloc[:len(df)].values
        df["accel_mps2"] = profile["accel_mps2"].iloc[:len(df)].values
        df["p_pack_kW"] = profile["p_pack_kW"].iloc[:len(df)].values
        actions_tbl = env.cfg.actions
        df["bat_ratio"] = df["action"].map(lambda a: actions_tbl[int(a)][0])
        df["sc_ratio"] = df["action"].map(lambda a: actions_tbl[int(a)][1])
        pf = veh.pack_factor
        df["p_load_kW"] = df["p_load"] * pf / 1000.0
        df["p_bat_kW"] = df["bat_ratio"] * df["p_load_kW"] * np.where(df["regen"], -1.0, 1.0)
        df["p_sc_kW"] = df["sc_ratio"] * df["p_load_kW"] * np.where(df["regen"], -1.0, 1.0)
        I_max = cfg["battery"]["cell"]["I_max_dis_A"]
        def _mode(row):
            if row["regen"]:
                return ("REJEN — Süperkap Şarj", "regen", "#1976D2")
            if row["p_load"] < 0.05:
                return ("Boşta", "idle", "#455A64")
            if abs(row["i_bat"]) > 0.85 * I_max:
                return ("Limit — Süperkap Destek", "limit", "#F57C00")
            if row["sc_ratio"] >= 0.5:
                return ("Yüksek Talep — Süperkap Aktif", "peak", "#E53935")
            return ("Cruise — Batarya", "cruise", "#43A047")
        modes = df.apply(_mode, axis=1, result_type="expand")
        df["mode_text"] = modes[0]
        df["mode_class"] = modes[1]
        df["mode_color"] = modes[2]

        st.session_state["sim_df"] = df
        st.session_state["sim_history"] = history
        st.session_state["sim_key"] = sim_key
        st.session_state["anim_idx"] = 0   # animasyon başlangıç

df: pd.DataFrame = st.session_state["sim_df"]
history = st.session_state["sim_history"]
N = len(df)

# ============================================================ Başarı banner'ı (sabit)
kpis = compute_kpis(history, dt_s=1.0, i_bat_max=cfg["battery"]["cell"]["I_max_dis_A"])
ss = success_score(kpis, i_bat_max=cfg["battery"]["cell"]["I_max_dis_A"])
score = ss["score"]
score_class = "bg-good" if score >= 75 else ("bg-mid" if score >= 50 else "bg-bad")
st.markdown(
    f"<div style='display:flex; gap:14px; align-items:center; margin-top:6px; flex-wrap:wrap;'>"
    f"<span class='success-badge {score_class}'>Senaryo Başarısı: %{score:.1f}</span>"
    f"<span class='label'>peak korunma</span> <b>%{ss['components']['peak_protection']*100:.0f}</b>"
    f"<span class='label'>peak'te süperkap</span> <b>%{ss['components']['supercap_at_peak']*100:.0f}</b>"
    f"<span class='label'>rejen şarj</span> <b>%{ss['components']['supercap_regen']*100:.0f}</b>"
    f"<span class='label'>ihlal yok</span> <b>%{ss['components']['no_violation']*100:.0f}</b>"
    f"<span class='label'>talep karşılanma</span> <b>%{ss['components']['supply_quality']*100:.0f}</b>"
    f"</div>",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================ Oynatma kontrolleri
ctrl_play, ctrl_reset, ctrl_speed, ctrl_slider = st.columns([1, 1, 1, 4])
with ctrl_play:
    play_btn = st.button("▶ Oynat", use_container_width=True, type="primary")
with ctrl_reset:
    if st.button("⏮ Başa Al", use_container_width=True):
        st.session_state["anim_idx"] = 0
with ctrl_speed:
    speed = st.selectbox("Hız", ["0.5x", "1x", "2x", "5x", "10x"], index=2)
with ctrl_slider:
    manual_idx = st.slider("Adım", 0, N - 1, st.session_state.get("anim_idx", N - 1), step=1, key="manual_slider")

speed_delay = {"0.5x": 0.20, "1x": 0.10, "2x": 0.05, "5x": 0.02, "10x": 0.01}[speed]
speed_step = {"0.5x": 1, "1x": 1, "2x": 1, "5x": 2, "10x": 4}[speed]

# ============================================================ Panel placeholder'ları
col_v, col_h, col_t = st.columns([1.1, 1.4, 1.5])
vehicle_ph = col_v.empty()
hess_ph = col_h.empty()
metrics_ph = col_t.empty()
st.divider()
ts_ph = st.empty()

I_max = cfg["battery"]["cell"]["I_max_dis_A"]

# Her render çağrısında artan benzersiz counter — plotly_chart key çakışmasını önler
if "render_seq" not in st.session_state:
    st.session_state["render_seq"] = 0


def _nk(prefix: str) -> str:
    st.session_state["render_seq"] += 1
    return f"{prefix}_{st.session_state['render_seq']}"


def render_vehicle(idx: int):
    row = df.iloc[idx]
    v_kmh = float(row["speed_kmh"])
    a_mps2 = float(row["accel_mps2"])
    p_kw = float(row["p_pack_kW"])
    mode_text = row["mode_text"]
    mode_class = f"mode-{row['mode_class']}"
    car_color = row["mode_color"]

    pos_x = 30 + (v_kmh / 160.0) * 200
    svg = f"""
    <svg viewBox='0 0 320 130' xmlns='http://www.w3.org/2000/svg' style='width:100%;background:linear-gradient(180deg,#0f1419 60%,#1d2530 100%);border-radius:8px;'>
      <line x1='0' y1='110' x2='320' y2='110' stroke='#37474F' stroke-width='2' stroke-dasharray='8,6'/>
      <g transform='translate({pos_x},48)'>
        <rect x='0' y='14' width='70' height='22' rx='6' fill='{car_color}' />
        <rect x='12' y='4' width='42' height='18' rx='5' fill='{car_color}' opacity='0.85'/>
        <rect x='16' y='6' width='16' height='14' fill='#B0BEC5' opacity='0.7'/>
        <rect x='34' y='6' width='16' height='14' fill='#B0BEC5' opacity='0.7'/>
        <circle cx='14' cy='38' r='6' fill='#212121'/>
        <circle cx='56' cy='38' r='6' fill='#212121'/>
        <circle cx='14' cy='38' r='2.5' fill='#757575'/>
        <circle cx='56' cy='38' r='2.5' fill='#757575'/>
      </g>
      <text x='160' y='22' text-anchor='middle' fill='#4FC3F7' font-size='14' font-weight='bold'>{v_kmh:.0f} km/h</text>
      <text x='160' y='38' text-anchor='middle' fill='#90A4AE' font-size='11'>{a_mps2:+.2f} m/s²</text>
    </svg>
    """

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v_kmh,
        gauge={
            "axis": {"range": [0, 160], "tickcolor": "#90A4AE"},
            "bar": {"color": car_color},
            "steps": [
                {"range": [0, 50], "color": "#1a3a1a"},
                {"range": [50, 100], "color": "#3a3a1a"},
                {"range": [100, 160], "color": "#3a1a1a"},
            ],
            "bgcolor": "#0e1117",
        },
        number={"font": {"color": "#4FC3F7", "size": 32}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    gauge.update_layout(template="plotly_dark", height=170, margin=dict(l=10, r=10, t=10, b=10))

    with vehicle_ph.container():
        st.markdown("##### Araç Durumu")
        st.markdown(svg, unsafe_allow_html=True)
        st.markdown(f"<div class='mode-pill {mode_class}'>{mode_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='label'>Anlık Güç Talebi</div>"
                    f"<div class='big-num'>{p_kw:+.1f} <span style='font-size:1.2rem'>kW</span></div>",
                    unsafe_allow_html=True)
        st.plotly_chart(gauge, use_container_width=True, key=_nk("gauge"))


def render_hess(idx: int):
    row = df.iloc[idx]
    bat_pct = float(row["bat_ratio"]) * 100.0
    sc_pct = float(row["sc_ratio"]) * 100.0

    donut = go.Figure(go.Pie(
        labels=["Batarya", "Süperkap"],
        values=[max(bat_pct, 0.001), max(sc_pct, 0.001)],
        hole=0.65,
        marker=dict(colors=["#43A047", "#1E88E5"], line=dict(color="#0e1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
        sort=False,
    ))
    action_labels = ["%100\nBAT", "%75 BAT\n%25 SC", "%50/%50", "%25 BAT\n%75 SC", "%100\nSC"]
    donut.add_annotation(
        text=f"<b>{action_labels[int(row['action'])]}</b>".replace("\n", "<br>"),
        x=0.5, y=0.5, font=dict(size=15, color="#4FC3F7"), showarrow=False,
    )
    donut.update_layout(template="plotly_dark", height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)

    soc_b = float(row["soc_bat"]) * 100.0
    soc_s = float(row["soc_sc"]) * 100.0
    bar_b_color = "#43A047" if soc_b > 50 else ("#FB8C00" if soc_b > 20 else "#E53935")
    bar_s_color = "#1E88E5" if soc_s > 30 else ("#FB8C00" if soc_s > 10 else "#E53935")
    soc_fig = go.Figure()
    soc_fig.add_trace(go.Bar(
        y=["Batarya SOC", "Süperkap SOC"],
        x=[soc_b, soc_s],
        orientation="h",
        marker=dict(color=[bar_b_color, bar_s_color]),
        text=[f"{soc_b:.1f}%", f"{soc_s:.1f}%"],
        textposition="inside",
        textfont=dict(size=16, color="white", family="Arial Black"),
        width=0.6,
    ))
    soc_fig.update_layout(
        template="plotly_dark",
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#37474F"),
        yaxis=dict(tickfont=dict(size=14)),
        height=150, margin=dict(l=10, r=20, t=10, b=10), showlegend=False,
    )

    p_b = float(row["p_bat_kW"])
    p_s = float(row["p_sc_kW"])
    arrow_w_b = max(2, min(12, abs(p_b) * 1.0))
    arrow_w_s = max(2, min(12, abs(p_s) * 1.0))
    color_b = "#43A047" if p_b >= 0 else "#FFB74D"
    color_s = "#1E88E5" if p_s >= 0 else "#FFB74D"
    flow_svg = f"""
    <svg viewBox='0 0 340 140' xmlns='http://www.w3.org/2000/svg' style='width:100%;background:#0f1419;border-radius:8px;'>
      <rect x='10' y='25' width='70' height='35' rx='5' fill='#1B5E20' stroke='#43A047' stroke-width='2'/>
      <text x='45' y='47' text-anchor='middle' fill='white' font-size='12' font-weight='bold'>BATARYA</text>
      <rect x='10' y='80' width='70' height='35' rx='5' fill='#0D47A1' stroke='#1E88E5' stroke-width='2'/>
      <text x='45' y='102' text-anchor='middle' fill='white' font-size='11' font-weight='bold'>SÜPERKAP</text>
      <rect x='150' y='52' width='40' height='36' rx='4' fill='#263238' stroke='#90A4AE' stroke-width='2'/>
      <text x='170' y='73' text-anchor='middle' fill='#90A4AE' font-size='10' font-weight='bold'>DC BUS</text>
      <rect x='250' y='52' width='80' height='36' rx='4' fill='#311B92' stroke='#7E57C2' stroke-width='2'/>
      <text x='290' y='75' text-anchor='middle' fill='white' font-size='11' font-weight='bold'>MOTOR</text>
      <line x1='80' y1='42' x2='150' y2='62' stroke='{color_b}' stroke-width='{arrow_w_b}' marker-end='url(#aa1_{idx})'/>
      <line x1='80' y1='97' x2='150' y2='80' stroke='{color_s}' stroke-width='{arrow_w_s}' marker-end='url(#aa2_{idx})'/>
      <line x1='190' y1='70' x2='250' y2='70' stroke='#7E57C2' stroke-width='5' marker-end='url(#aa3_{idx})'/>
      <text x='115' y='35' fill='{color_b}' font-size='12' font-weight='bold'>{p_b:+.2f} kW</text>
      <text x='115' y='115' fill='{color_s}' font-size='12' font-weight='bold'>{p_s:+.2f} kW</text>
      <defs>
        <marker id='aa1_{idx}' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'>
          <path d='M0,0 L6,3 L0,6 z' fill='{color_b}'/></marker>
        <marker id='aa2_{idx}' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'>
          <path d='M0,0 L6,3 L0,6 z' fill='{color_s}'/></marker>
        <marker id='aa3_{idx}' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'>
          <path d='M0,0 L6,3 L0,6 z' fill='#7E57C2'/></marker>
      </defs>
    </svg>
    """

    with hess_ph.container():
        st.markdown("##### HESS Güç Paylaşımı")
        st.plotly_chart(donut, use_container_width=True, key=_nk("donut"))
        st.plotly_chart(soc_fig, use_container_width=True, key=_nk("soc"))
        st.markdown(flow_svg, unsafe_allow_html=True)


def render_metrics(idx: int):
    distance_km = float(np.trapz(df["speed_kmh"].iloc[:idx + 1] / 3.6, dx=1.0) / 1000.0)
    avg_p_kw = float(df["p_pack_kW"].iloc[:idx + 1].abs().mean())
    peak_i_bat = float(df["i_bat"].iloc[:idx + 1].abs().max())
    sc_usage = float((df["sc_ratio"].iloc[:idx + 1] > 0.25).mean()) * 100.0
    regen_count = int(df["regen"].iloc[:idx + 1].sum())
    viol_now = int(df["violation"].iloc[:idx + 1].sum())
    t_now = float(df["t_s"].iloc[idx])

    with metrics_ph.container():
        st.markdown("##### Metrikler")
        st.markdown(f"<div class='label'>Zaman</div><div class='big-num' style='font-size:1.8rem'>{t_now:.0f} s · {idx+1}/{N}</div>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        mc1.metric("Mesafe", f"{distance_km:.2f} km")
        mc1.metric("Ortalama Güç", f"{avg_p_kw:.1f} kW")
        mc1.metric("Süperkap kullanım", f"%{sc_usage:.0f}")
        mc2.metric("Pik I_bat", f"{peak_i_bat:.2f} A",
                   delta=f"%{peak_i_bat/I_max*100:.0f} of I_max")
        mc2.metric("Rejen adımı", f"{regen_count}")
        mc2.metric("İhlal", f"{viol_now}")


def render_ts(idx: int):
    sub_t = df.iloc[:idx + 1]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Güç [kW]", "SOC [%]", "Aksiyon"),
    )
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["p_load_kW"], name="P_load", line=dict(color="#E53935", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["p_bat_kW"], name="P_bat", line=dict(color="#43A047", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["p_sc_kW"], name="P_sc", line=dict(color="#1E88E5", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["soc_bat"] * 100, name="SOC_bat", line=dict(color="#43A047", width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["soc_sc"] * 100, name="SOC_sc", line=dict(color="#1E88E5", width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=sub_t["t_s"], y=sub_t["action"], name="action", mode="lines+markers",
                             line=dict(color="#FFB300", width=1), marker=dict(size=4)), 3, 1)
    # Şu an imleci
    if len(sub_t) > 0:
        fig.add_vline(x=float(sub_t["t_s"].iloc[-1]), line=dict(color="#FFFFFF", width=1, dash="dot"), opacity=0.5)
    # X eksenini tüm senaryoya sabitle (zoom etmesin)
    t_max = float(df["t_s"].iloc[-1])
    fig.update_xaxes(range=[0, t_max])
    fig.update_xaxes(title_text="t [s]", row=3, col=1)
    # Y eksenleri
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1, range=[0, 100])
    # Aksiyon etiketleri: 0=Bat100, 1=75/25, 2=50/50, 3=25/75, 4=SC100
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4],
        ticktext=["0 · %100 BAT", "1 · %75 BAT", "2 · %50/50", "3 · %75 SC", "4 · %100 SC"],
        row=3, col=1, range=[-0.3, 4.3],
    )
    fig.update_layout(template="plotly_dark", height=620, hovermode="x unified", showlegend=True, margin=dict(t=40, b=40))

    with ts_ph.container():
        st.markdown("##### Zaman Serileri")
        st.plotly_chart(fig, use_container_width=True, key=_nk("ts"))


def render_frame(idx: int):
    render_vehicle(idx)
    render_hess(idx)
    render_metrics(idx)
    render_ts(idx)


# ============================================================ Çalıştırma
if play_btn:
    # Slider'da gösterilen pozisyondan başla; sona kadar oyna
    start_idx = int(st.session_state.get("manual_slider", 0))
    if start_idx >= N - 1:
        start_idx = 0
    for i in range(start_idx, N, speed_step):
        st.session_state["anim_idx"] = i
        render_frame(i)
        time.sleep(speed_delay)
    # Son kareyi göster
    st.session_state["anim_idx"] = N - 1
    render_frame(N - 1)
else:
    # Statik: slider değerine göre tek kare
    render_frame(int(manual_idx))
    st.session_state["anim_idx"] = int(manual_idx)

# ============================================================ RL bilgi paneli (alt)
with st.expander("RL nasıl çalışıyor? · State, Action, Policy, Q, Environment"):
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.markdown("**State (s)**\n\n8-D vektör:\n\n"
                    "`[P_load, SOC_bat, SOC_sc, V_bat, V_sc, I_bat_prev, I_sc_prev, demand]`\n\n"
                    "Demand 0-4 talep seviyesi.")
    with e2:
        st.markdown("**Action (a)**\n\n5 ayrık paylaşım:\n\n"
                    "`a=0: %100 bat`\n"
                    "`a=1: 75/25`\n"
                    "`a=2: 50/50`\n"
                    "`a=3: 25/75`\n"
                    "`a=4: %100 sc`")
    with e3:
        st.markdown("**Policy (π)**\n\nε-greedy:\n\n"
                    "Eğitimde ε ile rastgele (keşif), `1-ε` ile `argmax Q(s,a)` (sömürü).\n\n"
                    "Testte ε=0.")
    with e4:
        st.markdown("**Q-table**\n\nDiscretize state için 5 aksiyonun Q değeri.\n\n"
                    "$Q ← Q + α[r + γ·\\max Q' − Q]$\n\n"
                    "α=0.15, γ=0.95")
    with e5:
        st.markdown("**Environment**\n\n2RC Thevenin batarya + RC+ESR süperkap + DC-DC η.\n\n"
                    "Aksiyon → akımlar → fizik → reward → next state")

# ============================================================ Toplu senaryo raporu
st.divider()
st.markdown("### Tüm Senaryolarda Performans (sunum için toplu rapor)")

if st.button("Tüm 5 senaryoda koş ve karşılaştır"):
    rep_rows = []
    rep_radar = {}
    bar = st.progress(0.0)
    for i, sn in enumerate(scen_names):
        prof = make_scenario(sn, veh)
        env_r = make_env_from_config(cfg, ocv, profile_df=prof, seed=42)
        env_r.cfg.max_steps = len(prof) - 1
        ag = QLearningAgent(
            n_actions=env_r.n_actions,
            bins_p=cfg["training"]["q_learning"]["state_bins"]["p_load"],
            bins_soc_bat=cfg["training"]["q_learning"]["state_bins"]["soc_bat"],
            bins_soc_sc=cfg["training"]["q_learning"]["state_bins"]["soc_sc"],
        )
        ag.load(str(model_path))
        ag.epsilon = 0.0
        hist = evaluate_policy(env_r, ag, start_idx=0, max_steps=env_r.cfg.max_steps,
                               soc_bat_init=0.85, soc_sc_init=0.70)
        k = compute_kpis(hist, dt_s=1.0, i_bat_max=I_max)
        s = success_score(k, i_bat_max=I_max)
        rep_rows.append({
            "Senaryo": scenario_meta(sn)["title"],
            "Başarı %": round(s["score"], 1),
            "Pik I_bat (A)": round(k["peak_i_bat_A"], 2),
            "Süperkap kullanımı": f"%{k['supercap_usage_peak']*100:.0f}",
            "Rejen şarj": f"%{k['supercap_regen_charge_rate']*100:.0f}",
            "İhlal": int(k["violations"]),
            "Mesafe (km)": round(np.trapz(pd.DataFrame(hist).get("speed_kmh", pd.Series([0]*len(hist))) / 3.6) / 1000.0, 2) if False else round(float(prof["speed_kmh"].sum() / 3.6 / 1000), 2),
            "Toplam Reward": round(k["total_reward"], 1),
        })
        rep_radar[scenario_meta(sn)["title"]] = s["components"]
        bar.progress((i + 1) / len(scen_names))

    rep_df = pd.DataFrame(rep_rows)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

    # Radar chart — her senaryo için 5 boyut
    radar_dims = ["peak_protection", "supercap_at_peak", "supercap_regen", "no_violation", "supply_quality"]
    radar_labels = ["Peak Korunma", "Süperkap @ Peak", "Rejen Şarj", "İhlal Yok", "Talep Karşılanma"]
    fig_radar = go.Figure()
    colors = ["#4FC3F7", "#E53935", "#FFB300", "#43A047", "#AB47BC"]
    for (name, comps), col in zip(rep_radar.items(), colors):
        vals = [comps[d] * 100 for d in radar_dims]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            name=name,
            line=dict(color=col, width=2),
            fill="toself",
            opacity=0.3,
        ))
    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
        title="Senaryo Bazlı Başarı Bileşenleri (%)",
    )
    st.plotly_chart(fig_radar, use_container_width=True, key=_nk("radar"))

    st.success(f"Ortalama başarı: %{rep_df['Başarı %'].mean():.1f}  ·  En zorlu senaryo: **{rep_df.loc[rep_df['Başarı %'].idxmin(), 'Senaryo']}** (%{rep_df['Başarı %'].min():.1f})")

    st.download_button(
        "Toplu raporu CSV olarak indir",
        rep_df.to_csv(index=False).encode("utf-8"),
        file_name=f"toplu_rapor_{model_path.stem}.csv",
        mime="text/csv",
    )

# ============================================================ CSV indir
st.divider()
st.download_button(
    "Mevcut senaryo verisini CSV olarak indir",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"sim_{sc_name}_{model_path.stem}.csv",
    mime="text/csv",
)

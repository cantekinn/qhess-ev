"""Sayfa 1 — Eğitim (Q-Learning only).

Senaryo seç → Q-Learning ajanı eğit → canlı reward eğrisi → model kaydet.
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
import streamlit as st

from src.agents.q_learning import QLearningAgent
from src.agents.runner import train_one_episode
from src.data.loader import load_ocv_soc_curve
from src.env.hess_env import make_env_from_config
from src.scenarios import list_scenarios, scenario_meta, make_scenario
from src.utils.config import load_all_configs, load_yaml, project_root
from src.vehicle import VehicleParams

st.set_page_config(page_title="Eğitim — HESS", page_icon="🎓", layout="wide", initial_sidebar_state="expanded", menu_items={})
st.markdown(
    """<style>
#MainMenu, footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stSidebarNav"] ul li:first-child a span { display: none; }
[data-testid="stSidebarNav"] ul li:first-child a::after {
    content: "Ana Sayfa"; color: inherit; font: inherit;
}
.block-container {padding-top: 1.5rem; max-width: 1400px;}
h1 { color: #4FC3F7; font-weight: 800; }
.metric-card { background: rgba(79, 195, 247, 0.08); border-left: 4px solid #4FC3F7; padding: 10px 14px; border-radius: 8px; }
</style>""", unsafe_allow_html=True,
)

st.title("Q-Learning Eğitimi")
st.caption("Tabular Q-Learning · 4-D ayrıklaştırılmış state · 5 ayrık aksiyon · ε-greedy + episode-end replay")

cfg = load_all_configs()
cfg["vehicle"] = load_yaml("vehicle")
ocv = load_ocv_soc_curve()
veh = VehicleParams.from_config(cfg["vehicle"])

# --- Senaryo seçimi
col_l, col_r = st.columns([1, 2])
with col_l:
    st.subheader("Senaryo")
    scen_names = list_scenarios()
    chosen = st.multiselect(
        "Eğitim senaryoları (birden çok seçince her epizot rastgele biri)",
        scen_names,
        default=["mixed_urban", "overtaking", "city_stop_go"],
        format_func=lambda n: scenario_meta(n)["title"],
    )
    if not chosen:
        st.warning("En az bir senaryo seç.")
        st.stop()

    st.subheader("Hiperparametreler")
    episodes = st.slider("Episode sayısı", 20, 1000, 150, step=10)
    max_steps = st.slider("Episode başına maks adım", 50, 600, 200, step=50)
    alpha = st.number_input("α (öğrenme oranı)", 0.01, 1.0, float(cfg["training"]["q_learning"]["learning_rate"]), step=0.01)
    gamma = st.number_input("γ (indirim)", 0.5, 0.999, float(cfg["training"]["q_learning"]["discount_factor"]), step=0.01)
    eps_decay = st.number_input("ε decay", 0.90, 0.999, float(cfg["training"]["q_learning"]["epsilon_decay"]), step=0.001, format="%.4f")

with col_r:
    st.subheader("Seçilen Senaryolar")
    for n in chosen:
        m = scenario_meta(n)
        st.markdown(f"**{m['title']}** ({m['duration_s']}s) — {m['description']}")

st.divider()

start_col, status_col = st.columns([1, 3])
with start_col:
    start = st.button("Eğitimi Başlat", type="primary", use_container_width=True)

if start:
    # Her senaryo için profil önceden üret
    scen_profiles = {n: make_scenario(n, veh) for n in chosen}

    # İlk senaryoyla env'i kur (sonra her ep başında değiştirilebilir)
    env = make_env_from_config(cfg, ocv, profile_df=scen_profiles[chosen[0]], seed=cfg["training"]["common"]["seed"])
    env.cfg.max_steps = int(max_steps)

    agent = QLearningAgent(
        n_actions=env.n_actions,
        bins_p=cfg["training"]["q_learning"]["state_bins"]["p_load"],
        bins_soc_bat=cfg["training"]["q_learning"]["state_bins"]["soc_bat"],
        bins_soc_sc=cfg["training"]["q_learning"]["state_bins"]["soc_sc"],
        learning_rate=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=cfg["training"]["q_learning"]["epsilon_min"],
        epsilon_decay=eps_decay,
        init_battery_bias=cfg["training"]["q_learning"]["init_battery_bias"],
        replay_last_n=cfg["training"]["q_learning"]["replay_last_n"],
    )

    progress = st.progress(0.0)
    metric_row = st.empty()
    fig_holder = st.empty()

    rewards, eps_hist, viol_hist, scen_hist = [], [], [], []
    t0 = time.time()
    rng = np.random.default_rng(cfg["training"]["common"]["seed"])

    for ep in range(int(episodes)):
        # Her episode'da rastgele senaryo
        sc_name = chosen[int(rng.integers(0, len(chosen)))]
        env.profile_df = scen_profiles[sc_name].reset_index(drop=True)
        env.peak_load_W = max(float(env.profile_df["p_load_W"].max()), 1e-6)
        result = train_one_episode(env, agent)
        rewards.append(result["total_reward"])
        eps_hist.append(agent.epsilon)
        viol_hist.append(result["violations"])
        scen_hist.append(sc_name)
        progress.progress((ep + 1) / episodes)

        if (ep + 1) % max(1, episodes // 30) == 0 or ep == episodes - 1:
            recent = float(np.mean(rewards[-20:])) if rewards else 0.0
            elapsed = time.time() - t0
            with metric_row.container():
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Episode", f"{ep+1}/{episodes}")
                mc2.metric("Son 20 ortalama", f"{recent:.1f}")
                mc3.metric("ε", f"{agent.epsilon:.3f}")
                mc4.metric("Q-state sayısı", len(agent.q))
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rewards, mode="lines", name="reward", line=dict(color="#4FC3F7", width=1.5)))
            if len(rewards) >= 20:
                mov = np.convolve(rewards, np.ones(20)/20, mode="valid")
                fig.add_trace(go.Scatter(y=mov, mode="lines", name="MA(20)", line=dict(color="#FFB300", width=3)))
            fig.update_layout(
                title=f"Eğitim Reward Eğrisi  ·  {elapsed:.1f}s",
                xaxis_title="episode", yaxis_title="toplam reward",
                template="plotly_dark", height=420, hovermode="x unified",
            )
            fig_holder.plotly_chart(fig, use_container_width=True)

    # Eğitim sonu — kaydet
    out_dir = project_root() / cfg["paths"]["models_dir"]
    out_dir.mkdir(exist_ok=True)
    fname = f"ql_{int(time.time())}.pkl"
    fpath = out_dir / fname
    agent.save(str(fpath))
    st.success(f"Eğitim tamamlandı. Model kaydedildi: `{fname}` · Q-tablosunda {len(agent.q)} state · son 20 ep ortalama reward {np.mean(rewards[-20:]):.1f}")

    # Log kaydet
    log_dir = project_root() / cfg["paths"]["logs_dir"]
    log_dir.mkdir(exist_ok=True)
    log_df = pd.DataFrame({
        "episode": np.arange(len(rewards)),
        "reward": rewards,
        "epsilon": eps_hist,
        "violations": viol_hist,
        "scenario": scen_hist,
    })
    log_df.to_csv(log_dir / f"ql_{int(time.time())}.csv", index=False)

    # Reward dağılımı (senaryo başına)
    st.subheader("Senaryo başına ortalama reward (son 50 epizot)")
    last = log_df.tail(50)
    by_sc = last.groupby("scenario")["reward"].agg(["mean", "std", "count"]).round(2)
    by_sc.columns = ["ortalama", "std", "sayı"]
    st.dataframe(by_sc, use_container_width=True)

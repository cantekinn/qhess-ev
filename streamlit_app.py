"""HESS Akıllı Güç Yöneticisi — Ana Sayfa.

Q-Learning tabanlı batarya + süperkapasitör hibrit güç paylaşım kontrolü.
Sadece 2 sayfa: Eğitim ve Simülasyon.

Çalıştırma:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.utils.config import load_all_configs, load_yaml

st.set_page_config(
    page_title="HESS Akıllı Güç Yöneticisi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# Streamlit default UI'yi gizle + tema
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
/* Sidebar nav — ilk girdi "streamlit app" yerine "Ana Sayfa" */
[data-testid="stSidebarNav"] ul li:first-child a span { display: none; }
[data-testid="stSidebarNav"] ul li:first-child a::after {
    content: "Ana Sayfa";
    color: inherit;
    font: inherit;
}
.block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px;}
.stApp {background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%);}
h1 { color: #4FC3F7; font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { color: #B0BEC5; }
.metric-card {
    background: rgba(79, 195, 247, 0.08);
    border-left: 4px solid #4FC3F7;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.scenario-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}
.big-title { font-size: 2.4rem; font-weight: 900; }
.subtle { color: #78909C; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='big-title'>HESS Akıllı Güç Yöneticisi</div>"
    "<div class='subtle'>Batarya + Süperkapasitör hibrit enerji depolama sistemi · "
    "Q-Learning tabanlı dinamik güç paylaşım kontrolü</div>",
    unsafe_allow_html=True,
)
st.write("")

# Quick stats
try:
    cfg = load_all_configs()
    veh = load_yaml("vehicle")
    cfg_ok = True
except Exception as e:
    st.error(f"Konfigürasyon hatası: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='metric-card'><b>Araç</b><br>{veh['vehicle']['name']}<br>"
        f"<span class='subtle'>{veh['vehicle']['mass_kg']} kg · {veh['vehicle']['peak_power_kW']} kW</span></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card'><b>Batarya</b><br>CS2 LiCoO₂<br>"
        f"<span class='subtle'>{cfg['battery']['cell']['Q_nom_Ah']} Ah · I_max {cfg['battery']['cell']['I_max_dis_A']} A</span></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='metric-card'><b>Süperkapasitör</b><br>EDLC<br>"
        f"<span class='subtle'>{cfg['supercap']['cell']['C_F']} F · {cfg['supercap']['cell']['V_max']} V</span></div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='metric-card'><b>RL Ajan</b><br>Tabular Q-Learning<br>"
        f"<span class='subtle'>α={cfg['training']['q_learning']['learning_rate']} · γ={cfg['training']['q_learning']['discount_factor']}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### Çalışma Akışı")
st.markdown(
    """
**1 · Eğitim**: Ajanı seçtiğin senaryolarda Q-tablosu yakınsayana kadar eğit. Reward eğrisi canlı izlenir, model otomatik kaydedilir.

**2 · Simülasyon**: Eğitilmiş ajanı bir test senaryosunda koştur. Araç animasyonu, anlık SOC barları, güç paylaşım donut'u ve canlı zaman serileriyle davranışı izle.
"""
)

st.markdown("### Senaryolar")
sc = load_yaml("scenarios")
cards = st.columns(len(sc))
for i, (name, meta) in enumerate(sc.items()):
    with cards[i]:
        st.markdown(
            f"<div class='scenario-card'>"
            f"<b>{meta['title']}</b><br>"
            f"<span class='subtle'>{meta['duration_s']}s</span><br><br>"
            f"<span style='font-size:0.88rem'>{meta['description']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<center><span class='subtle'>Sol kenardaki menüden <b>Eğitim</b> veya <b>Simülasyon</b> sayfasına geçin.</span></center>",
    unsafe_allow_html=True,
)

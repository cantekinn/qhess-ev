<div align="center">

# QHESS-EV

### Q-Learning based Hybrid Energy Storage Management for Electric Vehicles

**Battery + Supercapacitor power-share control via Reinforcement Learning, with live vehicle animation and 5 real-world driving scenarios.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CALCE](https://img.shields.io/badge/Data-CALCE_CS2-2E7D32)](https://calce.umd.edu/battery-data#CS2)
[![License](https://img.shields.io/badge/License-Academic-blue)](#license)
[![Status](https://img.shields.io/badge/Status-Working-success)]()

</div>

---

## What is this?

When an electric vehicle accelerates hard or overtakes another car, the battery experiences a **current spike** that ages it faster. A **supercapacitor** is much better at handling these transient peaks but cannot store enough energy for the entire trip. A **hybrid energy storage system (HESS)** combines both — but **who decides which one to use, when, and how much?**

This project trains a **tabular Q-Learning agent** that learns this power-sharing policy from physics-based interactions, then visualizes its decisions in real time as a virtual car drives through five distinct scenarios.

```
   battery limits exceeded?  →  supercap takes over
   demand normalizes?        →  battery returns
   regen braking?            →  energy recovered into supercap
```

---

## Demo

> *Screenshots/GIFs to be added after final demo run.*

| | |
|---|---|
| **Training page** — live reward curve, scenario picker, Q-table coverage | *[screenshot placeholder]* |
| **Simulation page** — vehicle animation, SOC bars, power-flow diagram | *[screenshot placeholder]* |
| **Power flow** — battery / supercap / DC bus with live kW arrows | *[screenshot placeholder]* |

---

## Quick Start

```bash
git clone https://github.com/cantekinn/qhess-ev.git
cd qhess-ev
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501` automatically.

A pre-trained model is included in `models/`, so **simulation works out of the box** — no training required to see the demo.

### Optional: train your own agent

1. Open the **Training** page
2. Pick scenarios (default: all five)
3. Set hyperparameters (default: 150 episodes, α=0.15, γ=0.95)
4. Click **Start Training** — finishes in ~25 seconds
5. Switch to **Simulation** and select your fresh model

---

## How it works

### Reinforcement Learning

| Concept | This project |
|---|---|
| **State** $s \in \mathbb{R}^8$ | `[P_load, SOC_bat, SOC_sc, V_bat, V_sc, I_bat_prev, I_sc_prev, demand_code]` |
| **Action** $a \in \{0,1,2,3,4\}$ | Battery / supercap power-share ratios: `(1.0/0)`, `(0.75/0.25)`, `(0.5/0.5)`, `(0.25/0.75)`, `(0/1.0)` |
| **Policy** $\pi$ | ε-greedy (ε: 1.0 → 0.05 over training) |
| **Update** | $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)]$ |
| **Reward** | 5-term shaped: $r = R_\text{base} + R_\text{match} + R_\text{supply} - P_\text{loss} - I_\text{stress} - C_\text{violation}$ |

### Physical Models

| Component | Model | Reference |
|---|---|---|
| Battery | 2RC Thevenin ECM (exact exp. solver) | Plett, *BMS Vol. 2* (2015) |
| Supercapacitor | Classic RC + ESR, energy-based SOC | $E = \tfrac{1}{2}CV^2$ |
| DC-DC converter | Load-dependent efficiency | $\eta(P) = \eta_\text{min} + (\eta_\text{nom} - \eta_\text{min})(1 - e^{-2|P|/P_\text{rated}})$ |
| Vehicle | Longitudinal Newtonian dynamics | $F = ma + mgC_r + \tfrac{1}{2}\rho C_d A v^2 + mg\sin\theta$ |

### Driving Scenarios

| Scenario | Duration | Description | Expected behavior |
|---|---|---|---|
| **City stop-go** | 120 s | 0 → 50 km/h repeated, regen active | Moderate demand, frequent regen → supercap charges |
| **Overtaking** | 60 s | 80 → 120 km/h burst, then cruise | Short peak (> 30 kW) → supercap takes load |
| **Highway cruise** | 90 s | 100 km/h steady | Steady moderate demand → mostly battery |
| **Mixed urban** | 180 s | Stop-go + overtake + cruise | Tests adaptation across regimes |
| **Mountain climb** | 150 s | 6% grade, 60 km/h sustained | Continuous high demand → battery stress |

---

## Architecture

```
qhess-ev/
├── streamlit_app.py             Entry point (home page)
├── pages/
│   ├── 1_Egitim.py              Training UI
│   └── 2_Simulasyon.py          Simulation UI with vehicle animation
├── src/
│   ├── data/loader.py           CALCE CS2 loader + OCV-SOC extraction
│   ├── physics/
│   │   ├── battery_ecm.py       2RC Thevenin (Plett solver)
│   │   ├── supercap_model.py    RC + ESR, energy-based SOC
│   │   └── converter.py         Load-dependent η DC-DC
│   ├── vehicle.py               Longitudinal vehicle dynamics
│   ├── scenarios.py             Five driving scenario generators
│   ├── env/hess_env.py          Gym-style HESS environment + reward
│   ├── agents/
│   │   ├── q_learning.py        Tabular Q-Learning + episode replay
│   │   └── runner.py            train_one_episode, evaluate_policy
│   ├── metrics/kpi.py           Compute KPIs + 0–100 success score
│   └── utils/                   Config & plotting helpers
├── config/                      All hyperparameters as YAML
├── models/                      Trained Q-tables (.pkl)
├── logs/                        Training reward CSVs
├── docs/                        Technical report + presentation (LaTeX)
└── requirements.txt
```

---

## Success Metric

Each scenario is scored 0–100 by a weighted sum of five components:

| Weight | Component | Measures |
|---:|---|---|
| 30% | Peak protection | Did peak battery current stay below 80% of $I_\text{max}$? |
| 25% | Supercap at peak | When demand was high, was supercap engaged? |
| 20% | Regen capture | During braking, was energy stored in supercap? |
| 15% | No violations | Did SOC/current limits hold? |
| 10% | Supply quality | Was the requested power actually delivered? |

**Current performance** (using the included pre-trained model):

| Scenario | Success Score |
|---|:---:|
| Overtaking | **98.0%** |
| City stop-go | **98.9%** |
| Mixed urban | **96.2%** |

---

## Configuration

Everything is YAML, no code edits required for tuning:

```
config/
├── paths.yaml         Dataset path (edit this for your machine)
├── battery.yaml       2RC parameters: Q_nom, R0/R1/C1/R2/C2, V_min/V_max
├── supercap.yaml      C, ESR, V_min/V_max, energy/voltage SOC mode
├── converter.yaml     η_min, η_nom, P_rated for battery and supercap branches
├── vehicle.yaml       Mass, drag, rolling, drivetrain efficiency, pack factor
├── scenarios.yaml     Speed profiles for each scenario
├── reward.yaml        Five-term weight tuning
└── training.yaml      α, γ, ε decay, episode count, action table, state bins
```

> **Note on dataset path:** `config/paths.yaml > dataset_root` points to the CALCE CS2 dataset on the original machine. The dataset is **not included** due to size and licensing. The OCV-SOC curve is cached in `data_cache/ocv_soc.pkl`, so the project **runs without the raw dataset**. To regenerate the cache from raw data, download CS2 from [CALCE](https://calce.umd.edu/battery-data#CS2) and update the path.

---

## Tech Stack

- **Python 3.10+** — core
- **Streamlit** — interactive dashboard
- **NumPy, Pandas, SciPy** — numerics
- **Plotly** — live interactive charts
- **PyYAML** — configuration

No heavy ML frameworks (no PyTorch, no TensorFlow) — tabular Q-Learning runs in pure NumPy.

---

## Theory Foundation

This project sits at the intersection of three fields:

- **Reinforcement Learning** — Sutton & Barto, *Reinforcement Learning: An Introduction* (2018)
- **Battery Modeling** — Plett, *Battery Management Systems Vol. 2: Equivalent-Circuit Methods* (2015)
- **Hybrid Energy Storage Control** — Song et al., *Multi-objective optimization of HESS via DP*, J. Power Sources (2014)

The reward shaping approach follows Ng, Harada, Russell, *Policy invariance under reward transformations*, ICML 1999, ensuring the shaped reward preserves the optimal policy.

---

## Limitations & Future Work

| Known limitation | Mitigation path |
|---|---|
| Tabular Q-Learning's state-space coverage ~7% after 150 episodes | Train longer or switch to DQN with function approximation |
| Reward shaping pre-encodes preferred policy structure | Move to pure objective rewards + 5000+ episodes |
| Regen power split is rule-based, not RL-controlled | Extend action space to include regen-specific actions |
| Single battery cell physics, scaled via `pack_factor` | Full pack-level modeling with cell imbalance |
| No thermal model | Add lumped thermal coupling with reward penalty |
| No DP/MPC baseline comparison | Add offline DP solution as upper-bound benchmark |

---

## Documentation

Full technical report and presentation script (LaTeX, both ~25 pages):

- `docs/RAPOR.tex` — line-by-line code walk-through, theory, design rationale
- `docs/SUNUM.tex` — slide-by-slide presentation script with anticipated Q&A
- `docs/PROJE_OZET.md` — high-level Turkish summary

Compile with any LaTeX distribution (TeX Live, MiKTeX) or upload to [Overleaf](https://overleaf.com).

---

## License

Academic use only. CALCE CS2 dataset is subject to the CALCE Battery Data Group's terms of use.

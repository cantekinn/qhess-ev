# HESS Akıllı Güç Yöneticisi

Batarya + Süperkapasitör **hibrit enerji depolama sistemi (HESS)** için **Q-Learning** tabanlı dinamik güç paylaşım kontrolü. CALCE CS2 batarya verisi, 2RC Thevenin batarya ECM, RC+ESR süperkapasitör, longitudinal araç dinamiği ve 5 farklı sürüş senaryosu üzerine kurulu Streamlit dashboard.

**Ders:** Endüstriyel Uygulamalarda Yapay Zeka (EUYZ) — Lisans Bitirme Projesi
**Geliştirici:** Can Tekin

---

## Hızlı Başlangıç

```bash
cd EUYZ_RL_Dashboard
py -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Tarayıcı `http://localhost:8501` adresinde otomatik açılır.

### Konfigürasyon (önemli)

`config/paths.yaml` içindeki `dataset_root` mutlak yol içerir (CALCE veri seti yolu). Kendi sisteminde çalıştırmak için bu yolu kendi makinene göre düzenle:

```yaml
dataset_root: "C:/path/to/Endustriyel Uygulamalarda Yapay Zeka Projesi"
```

> **Not:** CALCE CS2 veri seti büyük boyutlu olduğu için bu repoya dahil edilmedi. Veri seti gerekirse [CALCE Battery Data](https://calce.umd.edu/battery-data#CS2) üzerinden indirilebilir. Repoda hazır eğitilmiş model (`models/ql_*.pkl`) ile dataset olmadan da simülasyon koşturulabilir.

---

## Sayfa Yapısı

| Sayfa | İçerik |
|---|---|
| **Ana Sayfa** | Sistem özeti, senaryo kartları, hızlı bağlantılar |
| **1 · Eğitim** | Q-Learning ajanını seçtiğin senaryolarda eğit, canlı reward eğrisi, model otomatik kaydet |
| **2 · Simülasyon** | Eğitilmiş ajanı bir senaryoda koştur. Animasyonlu araç, anlık SOC barları, güç paylaşım donut, güç akış diyagramı, 3'lü zaman serisi |

---

## Mimari

```
EUYZ_RL_Dashboard/
├── streamlit_app.py            # Ana sayfa (entry point)
├── pages/
│   ├── 1_Egitim.py             # Q-Learning eğitim arayüzü
│   └── 2_Simulasyon.py         # Test/simülasyon arayüzü (canlı animasyon)
├── src/
│   ├── data/loader.py          # CALCE CS2 yükleme + OCV-SOC çıkarımı
│   ├── physics/
│   │   ├── battery_ecm.py      # 2RC Thevenin (exact exp solver)
│   │   ├── supercap_model.py   # RC + ESR, enerji-tabanlı SOC
│   │   └── converter.py        # Yük-bağımlı η DC-DC modeli
│   ├── vehicle.py              # Longitudinal araç dinamiği
│   ├── scenarios.py            # 5 sürüş senaryosu üreticisi
│   ├── env/hess_env.py         # HESS environment + reward
│   ├── agents/
│   │   ├── q_learning.py       # Tabular Q-Learning + replay
│   │   └── runner.py           # train_one_episode, evaluate_policy
│   ├── metrics/kpi.py          # KPI hesaplama + success_score
│   └── utils/                  # Config + plotting yardımcıları
├── config/
│   ├── paths.yaml              # Veri seti yolu (kendine göre düzenle)
│   ├── battery.yaml            # 2RC ECM parametreleri
│   ├── supercap.yaml           # RC+ESR parametreleri
│   ├── converter.yaml          # DC-DC verimleri
│   ├── vehicle.yaml            # Araç (kompakt EV)
│   ├── scenarios.yaml          # Senaryo meta verileri
│   ├── reward.yaml             # Reward ağırlıkları
│   └── training.yaml           # Hiperparametreler
├── models/                     # Eğitilmiş Q-tabloları (.pkl)
├── logs/                       # Eğitim reward CSV'leri
├── docs/                       # RAPOR.tex, SUNUM.tex, vb.
└── requirements.txt
```

---

## Teorik Temel

### Reinforcement Learning
- **MDP:** $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ — state, action, geçiş, ödül, indirim faktörü
- **Q-Learning güncelleme:** $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- **Politika:** ε-greedy ($1-\varepsilon$ ile exploit, $\varepsilon$ ile explore)
- **State (8-D):** `[P_load, SOC_bat, SOC_sc, V_bat, V_sc, I_bat_prev, I_sc_prev, demand_code]`
- **Action (5 ayrık):** batarya/süperkap güç paylaşım oranı `{(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)}`

### Fiziksel Modeller
- **Batarya:** 2RC Thevenin ECM, Plett (2015) standardı, exact exponential solver
- **Süperkap:** Klasik RC + ESR, enerji-tabanlı SOC: $(V^2 - V_{min}^2)/(V_{max}^2 - V_{min}^2)$
- **DC-DC Converter:** Yük-bağımlı verim modeli, $\eta(P) = \eta_{min} + (\eta_{nom} - \eta_{min})(1 - e^{-2|P|/P_{rated}})$
- **Araç dinamiği (Newton):** $F = m\,a + m\,g\,C_r + \tfrac{1}{2}\rho\,C_d\,A\,v^2 + m\,g\sin\theta$

### Senaryolar (5 adet)
| Senaryo | Süre | Tasvir |
|---|---|---|
| `city_stop_go` | 120 s | Trafikte 0-50 km/h tekrarlı, regen aktif |
| `overtaking` | 60 s | 80 → 120 km/h kısa süreli PEAK |
| `highway_cruise` | 90 s | 100 km/h sabit |
| `mixed_urban` | 180 s | Dur-kalk + sollama + cruise karışımı |
| `mountain_climb` | 150 s | %6 eğim, sürekli yüksek talep |

### Reward Fonksiyonu (5 terim)
$$r_t = R_{base} + R_{match\_demand} + R_{supply\_quality} - P_{loss,norm} - I_{bat,stress} - C_{violation}$$

### Başarı Skoru (0-100, ağırlıklı)
```
0.30 · pik batarya akımı koruması
0.25 · yüksek talepte süperkap kullanımı
0.20 · rejen sırasında süperkap şarjı
0.15 · kısıt ihlali olmaması
0.10 · talep karşılanma oranı
```

---

## Veri Kaynağı

CALCE Battery Data Group, University of Maryland Center for Advanced Life Cycle Engineering. CS2 hücreleri: 1100 mAh LiCoO₂ prizmatik. Ham veri seti GitHub'a dahil edilmemiştir; OCV-SOC eğrisi cache'lendi (`data_cache/ocv_soc.pkl`) — dataset olmadan da çalışır.

---

## Kullanım Akışı

1. **Ana sayfada** sistem özetini ve 5 senaryoyu incele
2. **Eğitim sayfasında**:
   - Senaryoları seç (birden fazla seçilirse her epizotta rastgele biri)
   - Hiperparametreleri ayarla (default: 150 epizot, α=0.15, γ=0.95)
   - "Eğitimi Başlat" → canlı reward eğrisini izle (~25 sn)
   - Model otomatik `models/ql_<timestamp>.pkl` olarak kaydedilir
3. **Simülasyon sayfasında**:
   - Senaryo seç + kayıtlı modeli seç
   - "Oynat" → araç animasyonunu, anlık SOC barlarını ve güç akışını canlı izle
   - Senaryo sonunda başarı skoru ve KPI tablosu

---

## Lisans

Akademik amaçlı, EUYZ dersi bitirme projesi. CALCE veri seti kullanım koşulları CALCE Battery Data Group politikalarına tabidir.

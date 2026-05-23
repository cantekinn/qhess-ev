# HESS Akıllı Güç Yöneticisi — Proje Özeti

> EUYZ458 (Endüstriyel Uygulamalarda Yapay Zeka) Dönem Projesi
> Bahar 2025-2026 · BTU EEM · Can Tekin

---

## 1. Proje Nedir?

Elektrikli araçlarda **batarya + süperkapasitör** birlikte kullanılan bir hibrit enerji depolama sistemi (HESS) için, iki kaynak arasındaki anlık güç paylaşımını **Q-Learning** (pekiştirmeli öğrenme) ile optimize eden bir kontrolcü geliştirildi.

**Tek cümleyle:** Araç sollama yaparken bataryayı yormamak için süperkapı ne zaman devreye alacağına RL ajanı karar veriyor.

---

## 2. Neden Bu Konu?

### Endüstriyel motivasyon

Elektrikli araç endüstrisinde batarya:
- **En pahalı bileşen** (sistem maliyetinin %30-40'ı)
- **En kısa ömürlü** (500-2000 cycle sonrası %20 kapasite kaybı)
- **En kırılgan** (yüksek akım, sıcaklık, voltaj salınımı → hızlı yaşlanma)

Süperkapasitör ise:
- Yüksek güç yoğunluğu (kW/kg olarak bataryadan 10x üstün)
- Milyonlarca cycle dayanıklılık
- Geniş sıcaklık aralığı
- Ama enerji yoğunluğu düşük (Wh/kg)

İkisini akıllıca birleştirmek = bataryayı koruyup performans almak.

### Endüstriyel kullanım alanları

- Elektrikli/hibrit araçlar
- Tramvay ve metro sistemleri (rejen kazanımı)
- Forklift, otobüs, kamyon
- Yenilenebilir enerji sistemleri (rüzgâr, güneş)
- Endüstriyel robot kolları (ani hareket talepleri)

---

## 3. Bizden Beklenenler (Ders Gereksinimleri)

EUYZ458 dönem projesi kapsamında karşılanması gereken kriterler:

| Gereksinim | Bizim Karşıladığımız |
|---|---|
| Endüstriyel uygulamaya yönelik problem | EV güç yönetimi — birebir endüstriyel |
| Yapay zeka yöntemi kullanımı | Q-Learning (pekiştirmeli öğrenme) |
| Veri kullanımı | CALCE CS2 gerçek deneysel veri |
| Çalışan bir sistem (kod çalışmalı) | Streamlit dashboard, 3 sayfa |
| Sonuç değerlendirmesi | 5 senaryoda %96+ başarı, sıfır ihlal |
| Sunum + rapor | LaTeX rapor + slayt + canlı demo |
| Kavramların doğru anlaşılması | MDP, Bellman, Q-Learning teorisi |

---

## 4. Proje Akışı (Yapılanlar Sırayla)

### Adım 1 — Problem Tanımı ve Literatür
- HESS topolojileri incelendi (pasif paralel, yarı-aktif, tam aktif)
- Aktif paralel topoloji seçildi (DC-DC dönüştürücülü)
- RL literatürü tarandı: tabular vs DQN tartışıldı, tabular seçildi
- Reward shaping yöntemi belirlendi (Ng et al. 1999)

### Adım 2 — Fiziksel Model
- **Batarya:** 2RC Thevenin Equivalent Circuit Model (Plett 2015)
- **Süperkapasitör:** Klasik RC + ESR, enerji-tabanlı SOC
- **DC-DC dönüştürücü:** Yük-bağımlı verim eğrisi
- **Araç dinamiği:** Newton II — atalet + rolling + drag + grade kuvvetleri

### Adım 3 — Veri İşleme
- CALCE CS2 LiCoO₂ batarya verisi indirildi (3 klasör, 50+ dosya)
- **OCV-SOC eğrisi** gerçek deneysel veriden çıkarıldı (237 nokta)
- Step bazlı segmentasyon algoritması yazıldı
- Düşük akımlı (C/20) deşarj segmenti otomatik tespit edildi

### Adım 4 — RL Environment Tasarımı
- State (8-D): P_load, SOC_bat, SOC_sc, V_bat, V_sc, I_prev (×2), demand
- Action (5 ayrık): batarya-süperkap güç paylaşım oranları (0/100, 25/75, ...)
- Reward (6 terim): yaşam bonusu + demand-action match + tedarik kalitesi − kayıp − stres − ihlal
- Step fonksiyonu: aksiyon → akım → fizik adımı → reward → next state

### Adım 5 — Senaryo Üretici
- 5 gerçekçi sürüş senaryosu (hız profili → güç profili)
- Şehir dur-kalk, sollama, otoyol cruise, karma şehir, dağ tırmanışı
- Cosine ease smoothing ile gerçekçi hızlanma eğrileri

### Adım 6 — Q-Learning Ajan
- State discretization: 4-D anahtar (P_load, SOC_bat, SOC_sc, demand)
- Init bias: Q[s, a=0] = 0.5 (batarya tercihi yumuşak başlangıç)
- Episode-end replay: son 100 transition tekrar update
- ε-greedy: 1.0 → 0.05 azalış

### Adım 7 — Eğitim ve Test
- 500 episode eğitim (~25 saniye CPU)
- 5 senaryoda greedy test
- KPI hesaplama: 5 bileşenli ağırlıklı başarı skoru

### Adım 8 — Dashboard
- Streamlit ile 3 sayfa
- Ana sayfa, Eğitim, Simülasyon
- Animasyonlu araç, anlık SOC barları, güç akış diyagramı

### Adım 9 — Dokümantasyon
- LaTeX teknik rapor
- Sunum scripti
- Sunum akışı + speaker notes
- Anticipated Q&A

---

## 5. Amaçlar / Hedefler

### Birincil hedefler

1. **Fiziksel olarak doğru** bir HESS simülasyonu kur (oyuncak model değil, gerçek denklemler)
2. **Q-Learning ajanı eğit** ve yakınsayan reward eğrisi göster
3. **5 farklı senaryoda** test et, %90 üzeri başarı al
4. **Görsel olarak sunulabilir** bir dashboard ile davranışı izle

### İkincil hedefler

5. RL teorisini doğru anlat (MDP, Bellman, ε-greedy, yakınsama)
6. Gerçek deneysel veri kullan (CALCE), simülasyon kalitesini artır
7. Akademik dürüstlük: sınırlılıkları açıkça belirt
8. Hocaya kod-detay sorularına hazır ol

### Başarı kriterleri (KPI)

- [x] Eğitim reward eğrisi monotonik yükseliş → yakınsama
- [x] 5 senaryoda ihlal sayısı = 0
- [x] Pik batarya akımı < I_max × %50
- [x] Süperkap peak'lerde aktif (a ≥ 3 oranı yüksek)
- [x] Regen sırasında süperkap şarj davranışı
- [x] Çalışan dashboard, canlı demo

**Tüm kriterler karşılandı.**

---

## 6. Sunumda Bahsedilecek Ana Mesajlar

### Açılış (ilk 2-3 dakika)
> "EV bataryalarını ani güç darbelerinden korumak için süperkap eklemek tek başına yeterli değil — akıllı bir kontrolcü gerek. Biz bu kontrolcüyü pekiştirmeli öğrenme ile yaptık."

### Orta bölüm (5-10. dakika)
> "Sistem üç katmandan oluşuyor: fiziksel model (batarya + süperkap + araç), RL environment (state-action-reward), ve öğrenen ajan. Her katmanı detaylıca tasarladık."

### Sonuçlar (10-15. dakika)
> "5 senaryoda ortalama %97 başarı, sıfır ihlal. Sollama anında ajan otomatik olarak süperkapa geçiyor, bunu kimse ona söylemedi — öğrendi."

### Demo (15-20. dakika)
> "Canlı dashboard üzerinde göstereyim — bakın araç sollamaya başladığında aksiyon nasıl değişiyor, batarya akımı nasıl korunuyor."

### Q&A için anahtar savunmalar
- **Reward shaping eleştirisi:** "Ng 1999, lisans süresi pratik tercih"
- **Tabular kapsam eleştirisi:** "Senaryo bazlı eğitim, görmediği state'lere zaten gitmiyor"
- **Regen override eleştirisi:** "Güvenlik kuralı, fizik kısıtı"

---

## 7. Çıktılar (Deliverables)

```
EUYZ_RL_Dashboard/
├── streamlit_app.py           ← Çalıştırılabilir dashboard
├── pages/
│   ├── 1_Egitim.py
│   └── 2_Simulasyon.py
├── src/                        ← ~12 modül, ~2500 satır kod
├── config/                     ← 8 YAML konfig dosyası
├── models/                     ← Eğitilmiş Q-tablosu
└── docs/
    ├── RAPOR.tex              ← Teknik rapor (LaTeX)
    ├── SUNUM.tex              ← Sunum scripti (Beamer)
    ├── SUNUM_AKIS.tex         ← Slayt-slayt anlatım rehberi
    └── PROJE_OZET.md          ← Bu dosya
```

**Toplam:**
- ~2500 satır Python kodu
- 3 sayfalık Streamlit dashboard
- 1 kapsamlı teknik rapor
- 24 slaytlık sunum
- Eğitilmiş model dosyası
- 5 senaryoda test sonuçları

---

## 8. Teknik Özellikler (Hızlı Referans)

| Bileşen | Değer |
|---|---|
| Batarya | CALCE CS2 LiCoO₂, 1.1 Ah, 2RC Thevenin |
| Süperkap | EDLC, 100 F, 12 mΩ ESR, 2.7 V |
| Araç | Kompakt EV, 1300 kg, 45 kW peak |
| RL Algoritması | Tabular Q-Learning, ε-greedy |
| State boyutu | 8-D (4-D tabular anahtar) |
| Action sayısı | 5 ayrık |
| Eğitim süresi | ~25 saniye (500 episode) |
| Test başarısı | Ortalama %97 (5 senaryo) |
| Dashboard | Streamlit, 3 sayfa |
| Kod boyutu | ~2500 satır Python |

---

## 9. Çalıştırma

```powershell
cd "C:\Users\Can Tekin\Desktop\EUYZ_RL_Dashboard"
py -m streamlit run streamlit_app.py
```

Tarayıcıda `http://localhost:8501` açılır. Sol menüden:
1. **Ana Sayfa** — sistem özeti
2. **Eğitim** — yeni model eğit (~25 sn)
3. **Simülasyon** — eğitilmiş ajanla canlı test

---

## 10. Gelecek Geliştirmeler (Eğer Hoca Sorarsa)

- DQN/Double DQN ile genelleme
- Termal model + Arrhenius cycle life
- Predictive horizon (MPC-RL hibridi)
- Çoklu seed istatistiksel analiz
- Baseline karşılaştırma (rule-based, LP-filter, MPC)
- Pack-level fizik (cell imbalance, thermal coupling)
- Online uyarlama (sürücü profili öğrenme)

---

## 11. Kaynaklar

- Sutton, R.S., Barto, A.G. *Reinforcement Learning: An Introduction*, MIT Press, 2018.
- Watkins, C.J.C.H., Dayan, P. "Q-Learning." *Machine Learning* 8: 279-292, 1992.
- Plett, G.L. *Battery Management Systems, Vol. 1*, Artech House, 2015.
- Ng, A.Y., Harada, D., Russell, S. "Policy invariance under reward transformations." *ICML*, 1999.
- CALCE Battery Data: https://calce.umd.edu/battery-data

---

**Hazırlayan:** Can Tekin
**Son güncelleme:** 20 Mayıs 2026

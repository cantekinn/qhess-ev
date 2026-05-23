# Belgeler ve Derleme Talimatı

Bu klasör 2 ana belge içerir:

## 1. RAPOR.tex — Detaylı Teknik Rapor (~50 sayfa)

Hocaya verilecek tam akademik rapor. İçerik:

1. Proje tanımı ve motivasyon
2. RL teorik temeli (MDP, Bellman, Q-Learning, ε-greedy, konverjans)
3. Fiziksel HESS modeli (2RC Thévenin, RC süperkap, converter, araç dinamiği)
4. CALCE CS2 veri seti
5. Sistem modeli (State, Action, Reward, Environment)
6. Q-Learning ajanı tasarımı (init bias, episode-end replay)
7. 5 senaryo (şehir, sollama, otoyol, karma, dağ)
8. KPI ve başarı skoru
9. Yazılım mimarisi
10. **Satır satır kod açıklamaları** (vehicle.py, scenarios.py, battery_ecm.py, hess_env.py, q_learning.py)
11. Sonuçlar (eğitim konverjansı +75 → +240, senaryo bazlı %92-99 başarı)
12. Sınırlılıklar ve gelecek çalışmalar
13. Kaynakça (10 referans)
14. Ek: konfigürasyon dosyaları, çalıştırma talimatı

## 2. SUNUM.tex — Sunum Rehberi (~25 sayfa)

Sunum sırasında ne söylemen gerektiğini slayt-slayt anlatır. İçerik:

- **18 slayt önerisi** + her biri için anlatım metni
- **Canlı demo akışı** (eğitim → sollama → şehir → toplu rapor)
- **Anticipated Q&A** — 25+ olası soru ve cevap, kategorilenmiş:
  - RL teorisi (T1-T6): Bellman, ε-greedy, γ seçimi, off-policy/on-policy, tabular sınırları
  - Reward tasarımı (M1-M3)
  - Fiziksel model (F1-F5): 2RC neden, parametre fit, enerji-tabanlı SOC, pack faktör, termal
  - Kod (K1-K7): discretization, dict-Q, replay, akım çözme, Streamlit animation, plotly key, pickle
  - Metodoloji (M4-M9): overfitting, dağ ihlali, konverjans kanıtı, train/test, HIL, DDPG
- **Final checklist** sunum öncesi.

---

## Derleme (LaTeX → PDF)

### Yöntem 1: Lokal pdflatex (önerilen)
```bash
cd docs
pdflatex RAPOR.tex
pdflatex RAPOR.tex   # ikinci geçiş, içindekiler için
pdflatex SUNUM.tex
pdflatex SUNUM.tex
```

Windows'ta MiKTeX veya TeX Live yüklü olmalı.
- MiKTeX indir: https://miktex.org/download
- TeX Live indir: https://tug.org/texlive/

### Yöntem 2: Online (kurulum gerektirmez)
[Overleaf](https://www.overleaf.com) hesabı aç → Yeni proje → "Upload Project" → bu klasörü ZIP olarak yükle → "Recompile".

### Yöntem 3: VS Code + LaTeX Workshop eklentisi
1. VS Code'da `LaTeX Workshop` eklentisi yükle.
2. RAPOR.tex'i aç.
3. Ctrl+Alt+B (Build).

### Türkçe karakter sorunları varsa
LaTeX dosyaları `inputenc=utf8` + `babel=turkish` kullanıyor. Tüm Türkçe karakterler (ç, ğ, ı, İ, ö, ş, ü) destekli. Eğer derleme hata verirse:

```bash
# UTF-8 BOM olmadan dosyaları aç-kaydet (örn. Notepad++)
# veya komut satırından:
chcp 65001
pdflatex -interaction=nonstopmode RAPOR.tex
```

---

## Önerilen sunum öncesi sıra

1. `RAPOR.tex` → PDF'e derle → bastır veya PDF olarak hocaya gönder.
2. `SUNUM.tex` → PDF'e derle → kendi referansın olarak yanına al.
3. Slaytları (PowerPoint/Beamer) bu içeriklere göre kendin oluştur:
   - 18 slayt önerisi `SUNUM.tex` Bölüm 1'de.
   - Her slayt için anlatım metni var.
   - Slayt görsellerini dashboard'dan ekran görüntüsü olarak al.
4. Demo provası: dashboard açık, model eğitili, akış hazır.

Başarılar!

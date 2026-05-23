# Sunum Akışı — HESS Akıllı Güç Yöneticisi

**Hedef süre:** 15-20 dakika anlatım + 5-10 dakika Q&A
**Slayt sayısı:** 24
**Hocaya not:** Detaycı sorulara hazır olmak için her slaytta "olası sorular" eklendi

---

## Slayt 1 — Kapak (30 sn)

**Başlık:** HESS Akıllı Güç Yöneticisi
**Alt başlık:** Q-Learning Tabanlı Hibrit Enerji Depolama Sistemi
**Görsel:** Bir EV silüeti + batarya + süperkapasitör ikonu

**Söyleyeceğin:**
"Merhaba, EUYZ458 dersi kapsamında geliştirdiğim projeyi sunuyorum. Konum: elektrikli araçlarda batarya ve süperkapasitörü birlikte kullanan bir hibrit enerji depolama sistemini, pekiştirmeli öğrenme ajanı ile optimize etmek. Yaklaşık 15 dakika anlatım, sonra demo ve sorular."

---

## Slayt 2 — Neden Bu Problem? (1 dk)

**Başlık:** Elektrikli Aracın Sollama Anı
**Görsel:** Bir EV otoyolda solladığı an, üstünde "45 kW ani çekim" yazısı

**Söyleyeceğin:**
"Bir senaryo düşünün: elektrikli aracınızla 80 km/h cruise'dayken sollamaya başlıyorsunuz. 5 saniye içinde 120 km/h'a çıkmanız gerek. Bu, anlık olarak 45 kW seviyesinde güç talebi demek. Saf bir batarya bunu karşılayabilir, ama üç sorunla:

Bir, akım büyüdükçe I-kare-R kaybı katlanarak artıyor, verim düşüyor. İki, bu yüksek akım sıcaklık ve yaşlanma demek — cycle life kısalıyor. Üç, batarya terminal voltajı düşüyor, motor yeterli güç alamıyor.

Bu yüzden modern hibrit sistemlerde batarya yanına süperkapasitör eklenir."

**Olası soru:** "Süperkap pahalı değil mi?" → Cevap: enerji yoğunluğu düşük olduğu için küçük modül yeterli, sadece darbe tamponu.

---

## Slayt 3 — Çözüm: HESS (1 dk)

**Başlık:** Batarya + Süperkapasitör Hibrit Topolojisi
**Görsel:** Blok diyagram: Batarya → DC-DC → DC Bara ← DC-DC ← Süperkap, sağda Motor, üstte RL Ajan

**Söyleyeceğin:**
"İşte HESS topolojisi: enerji deposu olarak batarya, güç tamponu olarak süperkapasitör. İkisi de kendi DC-DC dönüştürücüsü üzerinden ortak bir DC barayı besliyor.

Batarya enerji yoğunluğu yüksek ama güç yoğunluğu düşük. Süperkap tam tersi: güç çok ama enerji az. İkisini birleştirince — sanki yarış arabasındaki turbo gibi — normal sürüşte batarya, ani ihtiyaçta süperkap devreye giriyor.

Asıl soru burada: HER AN, hangi oranda paylaşım yapacağız? İşte RL ajanı bunu cevaplıyor."

**Olası soru:** "Pasif paralel bağlantı da olur, neden aktif kontrolcü?" → Pasif sistemde paylaşım sadece iç dirençlere göre olur, optimal değil. Aktif kontrolcüyle her senaryoya farklı tepki verebiliyoruz.

---

## Slayt 4 — Neden Reinforcement Learning? (1 dk)

**Başlık:** Klasik Yöntemler vs. RL
**Görsel:** 4 yöntem karşılaştırma tablosu (kural-tabanlı, LP-filter, MPC, RL)

**Söyleyeceğin:**
"Bu paylaşım kararı için 4 yaklaşım var.

Kural-tabanlı: 'eğer güç şu eşiği geçerse süperkap aç' gibi if-else. Kolay ama sezgisel, optimum değil.

LP-filter: yükün yüksek frekanslı kısmı süperkapa, düşük frekanslı kısmı bataryaya. HESS literatür klasiği, ama zaman sabiti sabit, adaptif değil.

MPC: geleceği öngören optimal kontrol. Çok iyi ama hesaplama yükü yüksek ve sistemin doğru modeli lazım.

RL: deneyimden öğreniyor, model bağımsız, eğitim sonrası inferans tek tablo lookup yani çok hızlı. Lisans seviyesinde anlaşılabilir, sunumda da açıklanabilir. Tercihimiz buydu."

---

## Slayt 5 — RL'in Temeli: MDP (1.5 dk)

**Başlık:** Markov Karar Süreci
**Görsel:** Ajan ↔ Çevre döngüsü (s_t, a_t, r_t, s_{t+1} okları)

**Söyleyeceğin:**
"RL bir Markov Karar Süreci olarak formüle edilir. 5 elemanı var: state uzayı, action uzayı, geçiş dinamikleri P, reward fonksiyonu R, ve discount faktörü gamma.

Çalışma şekli: ajan her zaman adımında çevreyi gözlemliyor, bir state alıyor. Bu state'e bakarak bir aksiyon seçiyor. Çevre fiziksel olarak tepki veriyor: yeni bir state ve bir ödül üretiyor. Ajan deneyimden Q tablosunu güncelliyor.

Önemli olan **Markov özelliği**: gelecek sadece şu anki state'e bağlı. Yani state'e tüm anlamlı bilgiyi koymamız gerek. Bizim sistemde 8 boyutlu vektör kullandık, az sonra detayı geleceğim."

**Olası soru:** "Geçiş olasılığı P'yi modellediniz mi?" → Hayır, bizimki deterministik fizik modeli (ODE çözücü), olasılık dağılımı yok. Bu da Q-Learning için model-free yaklaşımı uygun kılıyor.

---

## Slayt 6 — Bellman + Q-Learning Update (1.5 dk)

**Başlık:** Q-Learning Algoritması
**Görsel:** Bellman update formülü büyük yazılı, altında ε-greedy seçim kuralı

**Söyleyeceğin:**
"Q fonksiyonu, bir state-action çiftinden başlayıp optimal politikayla devam etmenin getireceği beklenen kümülatif ödül.

Bellman optimum denklemi: Q-yıldız(s,a) = anlık ödül + gamma çarpı bir sonraki state'in en iyi Q değeri. Bu denklem kendi içinde tekrarlı — sabit noktası optimum Q fonksiyonu.

Q-Learning, bu denklemi örnekleme ile çözüyor. Update formülü: Q'yu, öğrenme oranı alfa kadar TD hedefine doğru kaydır. TD hedefi = anlık reward artı gamma çarpı bir sonraki state'in maksimum Q'su.

Aksiyon seçimi epsilon-greedy ile: epsilon olasılıkla rastgele, 1 eksi epsilon olasılıkla en iyi bilinen aksiyon. Eğitim ilerledikçe epsilon azalıyor — başta keşif, sonra sömürü.

Watkins ve Dayan 1992'de kanıtladılar: yeterli koşullarda Q-Learning yakınsıyor."

**Olası soru:** "Off-policy demek ne demek?" → Bellman update'inde max alıyoruz, ajanın seçtiği aksiyon değil. Yani davranış politikası ε-greedy olabilir ama öğrendiği target politika tamamen greedy.

---

## Slayt 7 — Bizim State (1 dk)

**Başlık:** State Vektörü — 8 Boyut
**Görsel:** Bir vektör görseli, her boyut açıklamalı

```
s_t = [P_load, SOC_bat, SOC_sc, V_bat, V_sc, I_bat_prev, I_sc_prev, demand]
```

**Söyleyeceğin:**
"Bizim Markov state'imiz 8 boyutlu. İçinde:

Anlık güç talebi (Watt cinsinden), iki kaynağın doluluk oranı (SOC değerleri), iki kaynağın terminal voltajı, bir önceki adımdaki akımlar (dinamik bilgi için), ve son olarak güç talebinin ayrık seviyesi yani 'demand kodu' — sıfırdan dörde kadar.

Demand kodu, anlık gücü peak güce göre oranlayarak hesaplanıyor. Bu, ajanın 'şu an düşük mü orta mı yüksek mi talep var' diye sınıflandırmasını kolaylaştırıyor."

**Olası soru:** "8 boyutu tabular Q-table'da nasıl tutuyorsunuz, kombinatorik patlamaz mı?" → Hepsini tutmuyoruz! Discretization slaytında göstereceğim — sadece 4 boyutu Q-table anahtarına çeviriyoruz, diğerleri bilgi amaçlı.

---

## Slayt 8 — Bizim Action (45 sn)

**Başlık:** 5 Ayrık Aksiyon — Güç Paylaşım Oranları
**Görsel:** 5 satırlık tablo: a=0 (1.0, 0.0), a=1 (0.75, 0.25), ..., a=4 (0.0, 1.0)

**Söyleyeceğin:**
"Aksiyon uzayımız çok basit: 5 ayrık paylaşım oranı.

a=0: tamamen batarya, süperkap kullanılmıyor.
a=4: tamamen süperkap, batarya bekliyor.
Aradakiler: 75/25, 50/50, 25/75.

5 aksiyon yeterli granülarite veriyor (0.25 adım), ama Q-tablosunu yönetilebilir tutuyor: 400 maksimum state çarpı 5 aksiyon = 2000 hücre, hafızada hiç sorun değil."

**Olası soru:** "Sürekli aksiyon daha iyi olmaz mıydı? (DDPG, SAC)" → Olabilirdi ama lisans seviyesinde overkill, hem matematiksel hem implementasyon olarak. 0.25 oran adımı bu uygulama için yeterli çözünürlük.

---

## Slayt 9 — Environment Kurgusu (2 dk — EN KRİTİK SLAYT)

**Başlık:** HESSEnv — Aksiyon Bir Adım Nasıl İşleniyor?
**Görsel:** 8 kutucuklu akış: aksiyon gelir → bara güç oranları → hücre güçleri → akımlar → fizik ECM → reward → next state

**Söyleyeceğin:**
"Bu slayt projemin kalbi. Ajan bir aksiyon seçtiğinde ne oluyor?

Adım bir: Aksiyondan batarya-süperkap oranlarını çıkarıyoruz. Mesela a=2 için 0.5 ve 0.5.

Adım iki: Bara'ya hangi güç gidecek bunu çarpıyoruz: P_load çarpı oran. Eğer regenerasyon durumundaysak — yani araç frenliyor — buradaki oranlara güvenlik kuralı uyguluyoruz: süperkap doluysa fazlayı bataryaya gönder. Bu bilinçli bir tasarım kararı, RL'in yapması zor bir karar.

Adım üç: Bara'da istenen güç için hücrenin sağlaması gereken gücü, DC-DC verim eğrisiyle hesaplıyoruz.

Adım dört: P = V · I formülüyle hücre akımlarına çeviriyoruz.

Adım beş: Akımları fiziksel limitlerle kırpıyoruz.

Adım altı: Bu akımları batarya ve süperkap fizik modellerine veriyoruz. 2RC Thevenin denklemleri çözülüyor, SOC ve voltaj güncelleniyor.

Adım yedi: Reward hesaplanıyor.

Adım sekiz: Yeni state üretiliyor, ajana dönülüyor."

**Olası soru:** "Regen sırasında neden RL karar vermiyor?" → Güvenlik. Dolu bataryaya regen yüklersek SOC max ihlali ve kalıcı hasar var. Bu fiziksel kısıt, lisans projesi süresinde RL'in keşfetmesini bekleyemeyiz. Gelecek geliştirme olarak ajanın bunu da öğrenmesi mümkün, 5000+ episode ile.

---

## Slayt 10 — Reward Fonksiyonu (2 dk)

**Başlık:** Çok Terimli Fiziksel Reward
**Görsel:** Büyük formül + 6 terimi gösteren ikonlar

```
r_t = R_base + R_match + R_supply - R_loss - R_stress - R_violation
```

**Söyleyeceğin:**
"Reward 6 terimden oluşuyor.

R_base: artı bir, her adımda yaşam bonusu. Bu önemli — ihlal yapmadan adım atan ajan pozitif reward biriktiriyor, gradient sinyali alıyor.

R_match: demand seviyesiyle aksiyonun eşleşmesi. Düşük talepte batarya seçerse artı iki, aksi halde eksi iki. Yüksek talepte süperkap seçerse artı üç. Bu shaping terim, eğitim hızını dramatik artırıyor.

R_supply: talep karşılanma kalitesi. P_supplied'in P_load'a oranı eğer 0.9 ile 1.05 arasındaysa artı iki, dışında ceza.

R_loss: toplam ohmik kayıp normalize edilip çıkarılıyor. Ajan kayıpları minimize etmeyi öğreniyor.

R_stress: batarya akımı I_max'ın yüzde 70'ini geçerse karesel ceza. Cycle life koruma.

R_violation: SOC veya voltaj sınırı ihlal edilirse eksi beş büyük ceza.

Toplam reward bu altısının toplamı. Eğitim sırasında ortalama reward 280-340 plato'ya yakınsıyor."

**Olası soru:** "Bu shaping yapay değil mi, saf RL değil mi?" → Doğru tespit. Ng-Harada-Russell 1999'da potansiyel-tabanlı shaping'in optimum politikayı korumadığı kanıtlandı, bizimki yaklaşıktır. Saf objektif reward ile binlerce episode'da yakınsama olurdu, lisans süresine sığmazdı. Pratik bir mühendislik kararı.

---

## Slayt 11 — Batarya 2RC Thevenin Modeli (1.5 dk)

**Başlık:** Batarya Fiziksel Modeli
**Görsel:** 2RC eşdeğer devre çizimi (OCV kaynağı + R0 + iki RC kolu)

**Söyleyeceğin:**
"Bataryayı 2RC Thevenin eşdeğer devresi ile modelledim. Bu, Plett'in 2015 ders kitabındaki standart.

İçinde: OCV (açık devre voltajı, SOC'a bağlı), seri iç direnç R0, ve iki adet RC kolu. Birinci RC kısa zaman sabitli — elektrokimyasal polarizasyon, yaklaşık 100 saniye. İkinci RC uzun zaman sabitli — difüzyon, yaklaşık 9 dakika.

Terminal voltaj: OCV eksi I-çarpı-R0 eksi iki RC kolundaki gerilim düşümü. SOC ise Coulomb counting ile takip ediliyor.

RC kollarını forward-Euler ile çözmek hatalı olurdu, kararsızlık olabilir. Bunun yerine exact exponential çözüm kullandım: V_RC[k+1] = V_RC[k] çarpı exp(-dt/tau) artı I·R çarpı (1 - exp(-dt/tau)). Bu Plett formülasyonu, sayısal olarak kararlı."

**Olası soru:** "R0, R1, C1, R2, C2 değerleri nereden?" → Plett kitabından LiCoO2 hücreler için tipik başlangıç değerleri. İdeal olarak CALCE CS2 verisinden EIS veya pulse fitting ile çıkarılır; bu gelecek geliştirme olarak duruyor.

---

## Slayt 12 — Süperkapasitör ve Araç Dinamiği (1 dk)

**Başlık:** Diğer Fiziksel Modeller
**Görsel:** Sol: süperkap RC+ESR devresi. Sağ: araç kuvvet denklemi diyagramı

**Söyleyeceğin:**
"Süperkapasitörü klasik RC + ESR ile modelledim. Voltaj türevi = -I/C, terminal voltajı = V_sc - I·ESR.

Önemli bir nokta: SOC'u enerji-tabanlı tanımladım, voltaj-tabanlı değil. Çünkü süperkap enerjisi V karesi ile orantılı. Voltaj-tabanlı SOC yanıltıcı olur — V_max'ın yarısında SOC sıfır görünür ama enerji hala yüzde yetmiş beş'tir.

Araç dinamiği için Newton ikinci yasası: atalet + rolling + drag + grade kuvvetleri. Bunu hıza çarpınca tekerlek gücü, sürücü zincir verimi ile bölünce pack gücü. Regenerasyonda tam tersi yön.

Araç parametreleri kompakt EV: 1300 kg, 0.30 Cd, 45 kW peak. Renault Zoe / Fiat 500e sınıfı."

**Olası soru:** "Tek hücre simüle ediyorsunuz ama araç pack seviyesinde, nasıl uyumlu?" → Pack faktörü 5500 kullandım. 45 kW'lık araç gücünü 5500'e böldüğümde tek CS2 hücre seviyesinde 8.18 W çıkıyor — hücrenin fiziksel maksimumu. Mantıksal olarak pack 5500 hücre seriparalel kombinasyon.

---

## Slayt 13 — Veri Seti: CALCE CS2 (1 dk)

**Başlık:** Veri Kaynağı
**Görsel:** CALCE logosu + CS2 hücre fotoğrafı + 3 klasör ağacı

**Söyleyeceğin:**
"Veri için Maryland Üniversitesi CALCE merkezinin açık erişim CS2 batarya verisini kullandım. CS2 hücreleri 1.1 amp-saat LiCoO2 prizmatik hücreler.

Üç klasörden faydalandım:
Birinci, ana cycling verisi — 39 farklı xlsx dosyası, farklı deşarj akımları.
İkinci, OCV-SOC karakterizasyon — düşük akım deşarj eğrileri.
Üçüncü, dinamik sürüş profilleri — DST ve FUDS standartları.

Önemli bir nokta: OCV-SOC eğrisini literatür formülünden değil, gerçek CALCE deşarj verisinden çıkardım. Akademik dürüstlük için bu kritik."

---

## Slayt 14 — OCV-SOC Çıkarım Algoritması (1.5 dk)

**Başlık:** Gerçek Veriden OCV Eğrisi Nasıl Çıkarılıyor
**Görsel:** 6 adımlık akış diyagramı + sonuç eğrisi

**Söyleyeceğin:**
"OCV-SOC eğrisi 2RC modelinin kalbi. Şu algoritmayla çıkardım:

Adım bir: CALCE ZIP'lerini açıyoruz, Arbin formatındaki Channel sheet'leri buluyoruz.

Adım iki: Step bazlı gruplama yapıyoruz — her cycle her step'in özetini çıkarıyoruz.

Adım üç: En yavaş deşarj segmentini seçiyoruz. Kriterler: akım küçük (yani C/20 mertebesinde), derin deşarj (Q_nom'un yarısından fazla), yeterli çözünürlük.

Adım dört: SOC'u kapasite oranıyla hesaplıyoruz.

Adım beş: Monotonik düzeltme, tekrar eden SOC'leri ortalıyoruz.

Adım altı: Sağlık kontrolü — OCV(1) > OCV(0) olmalı.

Sonuç: 237 nokta, OCV(0) yaklaşık 2.70 V, OCV(1) yaklaşık 4.09 V. CS2 LiCoO2 için doğru aralık.

Eğer veri okunamazsa Chen-Mora analitik formülüne fallback ediliyoruz. Ama sağ olsun, CALCE verisi temiz."

---

## Slayt 15 — 5 Test Senaryosu (1 dk)

**Başlık:** Gerçekçi Sürüş Profilleri
**Görsel:** 5 senaryonun hız profili grafiği üst üste

**Söyleyeceğin:**
"Ajanı 5 farklı gerçekçi senaryoda eğitiyoruz ve test ediyoruz.

Şehir dur-kalk: 0-50 km/h tekrarlı, NEDC ECE-15 türevi. Sık regen → süperkapı doldurma fırsatı.

Sollama: 80 km/h cruise, sonra 5 saniyede 120 km/h, sonra geri 90 km/h. Kısa süreli peak.

Otoyol cruise: 100 km/h sabit hız. Orta sabit yük.

Karma şehir: dur-kalk artı cruise artı sollama. Adaptasyon testi.

Dağ tırmanışı: yüzde 6 eğim, 60 km/h sabit. Sürekli yüksek talep, süperkap kritik.

Her senaryo hız profilinden Newton denklemleriyle güç profiline dönüşüyor. Cosine ease smoothing kullandım, gerçekçi hızlanma eğrileri için."

---

## Slayt 16 — Eğitim Süreci (1 dk)

**Başlık:** Q-Learning Eğitim Akışı
**Görsel:** Reward eğrisi (üst) + epsilon decay eğrisi (alt)

**Söyleyeceğin:**
"Eğitim 500 episode sürdü, yaklaşık 25 saniye laptop CPU'da.

Her episode'da: rastgele bir senaryo seçiliyor, çevre sıfırlanıyor, 200 adımlık döngü başlıyor. Her adımda ajan epsilon-greedy ile aksiyon seçiyor, çevre adım atıyor, ajan Q-update yapıyor. Episode sonunda 'replay' tetikleniyor — son 100 transition tekrar update ediliyor, mini-batch benzeri pekiştirme.

Reward eğrisinde üç faz görünüyor: ilk yüz episode keşif baskın, dalgalı ama yükselişte. Yüz ile iki yüz elli arası politika oturuyor, ortalama hızla artıyor. İki yüz elli sonrası yakınsama, ortalama 280-340 platosunda.

Q-tablosunda eğitim sonunda 27 unique state aktif. Toplam mümkün state 400, yani yüzde altı buçuk coverage. Bu az gibi görünse de senaryo bazlı eğitim olduğundan ajan zaten görmediği bölgelere gitmiyor."

**Olası soru:** "Daha çok episode neden değil?" → Reward eğrisi 250'den sonra plato. Daha çok episode iyileştirme getirmiyor, sadece zaman harcıyor.

---

## Slayt 17 — Test Sonuçları (1 dk)

**Başlık:** Senaryo Bazlı Başarı Skorları
**Görsel:** Tablo: 5 senaryo, başarı %, pik akım, ihlal sayısı, SOC final

**Söyleyeceğin:**
"Eğitilmiş ajanı 5 senaryoda test ettim. Başarı skoru 5 bileşenli ağırlıklı toplam: pik batarya akım koruması yüzde 30, peak'te süperkap kullanımı yüzde 25, regen şarjı yüzde 20, ihlal yokluğu yüzde 15, talep karşılanma yüzde 10.

Sonuçlar:
Sollama: yüzde 98.
Şehir dur-kalk: yüzde 98.9.
Karma şehir: yüzde 96.2.
Otoyol: yaklaşık yüzde 98.
Dağ tırmanışı: yaklaşık yüzde 95.

Ortalama yüzde 97 üzerinde. Tüm senaryolarda **ihlal sıfır**. Pik batarya akımı I_max olan 2.2 amperin çok altında, en zorlu dağ tırmanışında bile 0.55 amper."

---

## Slayt 18 — Davranış Analizi: Sollama (1.5 dk)

**Başlık:** Ajan Sollama Anında Ne Yapıyor?
**Görsel:** Sollama senaryosu için zaman serisi: hız (üst), aksiyon (orta), iki SOC (alt)

**Söyleyeceğin:**
"Bu en güzel slaytlardan biri. Sollama senaryosunda 20-25. saniye arası peak yaşanıyor.

Aksiyon zaman serisine bakalım:
İlk 20 saniye: ajan a=1 seçmiş, yani yüzde 75 batarya yüzde 25 süperkap. Cruise davranışı, batarya dominant.

20'den 25'e: ajan a=4'e geçiyor. Yüzde 100 süperkap. Peak gücü süperkap karşılıyor, batarya korunuyor.

25'ten 35'e: ajan a=3'te kalıyor. Yüzde 25 batarya, yüzde 75 süperkap. Yavaşlama aşaması, süperkap hâlâ baskın.

35 saniyeden sonra: yeniden a=2, sonra a=1. Kararlı duruma dönüş.

Bu **tam istenen davranış**: peak'te süperkap aktif, normalde batarya, geçişlerde karma. Ajan bu kararı vermiş, biz dikte etmedik."

**Olası soru:** "Bu davranışı reward shaping ile söylemediniz mi?" → R_match terimi yüksek talepte süperkapı ödüllendiriyor, doğru. Ama hangi tam aksiyonu (a=3 mü a=4 mü) ne kadar süre seçmesi gerektiğini ajan kendi öğrendi. Geçişlerin zamanlaması da öğrenilmiş.

---

## Slayt 19 — Dashboard Tanıtım (45 sn)

**Başlık:** Streamlit Dashboard
**Görsel:** 3 sayfa screenshot

**Söyleyeceğin:**
"Sistemi sunmak için Streamlit ile bir dashboard geliştirdim. Üç sayfa var.

Ana sayfa: sistem özeti, 5 senaryo kartı.

Eğitim sayfası: hyperparametreleri ayarla, senaryoları seç, canlı reward grafiği ile eğit, model kaydet.

Simülasyon sayfası: en zengin sayfa. Animasyonlu araç, anlık hız, mod bildirimi, donut chart ile güç paylaşım yüzdesi, SOC barları, güç akış diyagramı, canlı zaman serileri.

Streamlit hilesi olarak placeholder + time.sleep deseni ile gerçek-zamanlı animasyon yaptım. Her plotly chart için unique key vermek gerek, yoksa duplicate ID hatası geliyor."

---

## Slayt 20 — Canlı Demo Geçişi (10 sn)

**Başlık:** Demo Zamanı
**Görsel:** "Dashboard'a Geçiş" yazısı, büyük play butonu

**Söyleyeceğin:**
"Şimdi canlı demoya geçeceğim. Üç senaryo göstereceğim: sollama (peak davranışı için), şehir dur-kalk (regen davranışı için), dağ tırmanışı (zorluk için)."

**[Demo sırası — slayt dışı, 4-5 dk]**
1. Ana sayfa → sistem bileşenleri göster
2. Eğitim sayfası → 50 episode hızlı eğitim (kanıt amaçlı)
3. Simülasyon → Sollama, 5x hızda oynat, animasyonu anlat
4. Simülasyon → Şehir dur-kalk, regen davranışına dikkat çek
5. Geri sunuma

---

## Slayt 21 — Sınırlılıklar (1 dk)

**Başlık:** Bilinen Sınırlılıklar (Akademik Dürüstlük)
**Görsel:** 6 madde liste

**Söyleyeceğin:**
"Akademik dürüstlük için bilinen sınırlılıkları açıkça söylüyorum.

Bir: Q-tablo coverage yüzde 6.8. Görmediği state'lere genelleme yapamaz. Tabular yöntemin doğal sınırı.

İki: Tek hücre simülasyonu. Pack cell imbalance, thermal coupling modellenmedi.

Üç: Reward shaping yaklaşık potansiyel-tabanlı değil. Teorik optimum politika garantisi yok.

Dört: Regen override hard-coded. RL bu kararı vermiyor.

Beş: Termal model yok. Sıcaklık sabit kabul.

Altı: Tek seed çalıştırma, istatistiksel varyans yok.

Bu sınırlılıklar bilinçli mühendislik kararları, gelecek çalışmada ele alınabilir. Özellikle DQN'e geçiş, termal model ekleme, baseline karşılaştırması güzel uzantılar olurdu."

---

## Slayt 22 — Sonuç (45 sn)

**Başlık:** Ne Başardık?
**Görsel:** 6 başarı maddesi tikleri

**Söyleyeceğin:**
"Özetle başardıklarımız:

Fiziksel olarak doğru HESS environment — 2RC ECM + RC+ESR + DC-DC verim.
Gerçek CALCE verisinden OCV-SOC çıkarımı.
Newtonian araç dinamiği + 5 gerçekçi senaryo.
Tabular Q-Learning ajan, başarıyla yakınsayan eğitim.
Tüm senaryolarda yüzde 96 üzeri başarı, sıfır ihlal.
Zengin Streamlit dashboard.

Anahtar mesaj: RL ajan, peak güç taleplerinde süperkapasitörü doğru kullanarak batarya akımını I_max'ın yüzde 25'i altında tutmayı öğrendi. Bu, deneyimle keşfedilmiş bir davranış."

---

## Slayt 23 — Teşekkür (15 sn)

**Başlık:** Teşekkürler — Sorular?
**Görsel:** BTU EEM logosu + iletişim bilgisi

**Söyleyeceğin:**
"Beni dinlediğiniz için teşekkürler. Sorularınızı bekliyorum."

---

## Slayt 24 (Yedek) — Anticipated Q&A Kartı

**Sadece sen göreceksin, izleyiciye gösterme. Hocanın olası sorularına hazır cevaplar:**

**S1: Q-tablo coverage neden bu kadar düşük?**
"Senaryo bazlı eğitim çünkü. Ajan görmediği state'lere zaten gitmiyor. DQN'e geçilse kapsama çok genişlerdi ama lisans projesi için tabular yeterli ve açıklanabilir."

**S2: Reward shaping kullandığınız için RL gerçekten öğrendi mi?**
"Shaping eğitim hızını artırmak için, Ng 1999. Aksiyonların tam timing'ini ajan kendi öğrendi. Saf objektif reward ile binlerce episode'la aynı sonuç gelirdi, lisans projesi süresine sığmazdı."

**S3: Neden tabular Q-Learning, DQN değil?**
"Açıklanabilirlik. Q-tablosunun her hücresi sunumda anlaşılır. DQN deadly triad sorunu, yorumlanması zor. Bu state uzayı küçük olduğu için tabular yeterli."

**S4: 2RC parametreleri nereden?**
"Plett 2015 ders kitabı standartları. Gelecek geliştirme olarak CS2 verisinden EIS / pulse fit önerilir."

**S5: Pack faktörü 5500 niye?**
"45 kW araç peak / 5500 = 8.18 W per cell, CS2 fiziksel max. Mantıksal pack."

**S6: Regen override RL değil, bu adil mi?**
"Doğru, güvenlik kuralı. Dolu bataryaya regen yüklemek SOC max ihlali yaratır. Lisans süresinde RL'in keşfetmesini bekleyemeyiz. Gelecek geliştirme."

**S7: %98 başarı nasıl hesaplandı?**
"5 bileşenli ağırlıklı toplam. Sollamada: pik koruma 1.0, sc peak kullanım 1.0, regen şarj 1.0, ihlal yok 1.0, supply 0.85. Toplam 0.30+0.25+0.20+0.15+0.085 = 0.98 yani %98."

**S8: Demand eşikleri (0.10, 0.25, 0.50, 0.80) niye?**
"Hız profilindeki tipik dağılım gözlenerek ayarlandı. Sensitivity analizi yapılırsa optimize edilebilir."

**S9: Gamma 0.95 neden?**
"1/(1-γ) = 20 adım ufuk. 200 adım episode için yeterli. 0.99 olsa 100 adım ufuk olurdu ama yakınsama yavaşlardı."

**S10: Süperkap kapasitansı 100 F nereden?**
"Maxwell BCAP serisi tipik EDLC modül değeri. Enerji ~0.1 Wh, peak yardımı için yeterli."

---

## Genel İpuçları

- **Tempo:** Her slaytta 30-90 saniye, toplam 18-20 dakika
- **Görsel ipucu:** Her slayt için bir ana görsel + metni az tut
- **Demo süresi:** 4-5 dakika ayır, hocayı sıkma
- **Q&A tutumu:** Kısa öz cevap → kabul + savun. "Bu yapılmadı çünkü ..." cümlesi her zaman iyidir
- **Heyecan kontrolü:** İlk 3 slayt önemli, akıcı geç
- **Soru gelmezse:** "Demo isteyen var mı?" diye dön, bir senaryo daha aç

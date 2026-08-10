# Soğuk Birleşme (Cold Shut) Riski – Besleyici ve Kalıp Malzemesi Etkisi

## 1. Soğuk birleşme nedir?

Dökümde iki sıvı metal cephesi birleşmeden önce yeterince soğur veya ince bir kesit katılaşmadan akışı tamamlayamazsa, **soğuk birleşme (cold shut)** oluşur. Riski artıran başlıca faktörler:

- Metalin sıvı sıcaklığının altına düşmesi
- Doldurma ön cephesinin çok yavaş ilerlemesi
- İnce kesit (küçük modülüs)
- Kalıbın metalden çok hızlı ısı çekmesi

## 2. Kalıp malzemesi neden fark yaratır?

Aynı geometri ve aynı alaşım için soğuk birleşme riski büyük ölçüde **kalıp-malzeme arayüzündeki ısı çekme hızına** bağlıdır. Bunu en iyi gösteren fiziksel nicelik **termal effusivite**dir:

```
e = sqrt(k * rho * cp)   [J m^-2 K^-1 s^-0.5]
```

| Kalıp malzemesi | k (W/m·K) | rho (kg/m³) | cp (J/kg·K) | e (effusivity) | Soğutma hızı |
|----------------|-----------|-------------|-------------|------------------|--------------|
| Kum (green sand) | 0.58 | 1600 | 1170 | ~1040 | En yavaş |
| Seramik | 1.20 | 2000 | 1000 | ~1550 | Orta |
| Metal (çelik kalıp) | 45.0 | 7850 | 460 | ~12 700 | En hızlı |

Programda `mold_chill_factor = 1.0 + 0.5 * log(e_mold / e_sand)` şeklinde uygulanır. Kum için ~1.0, seramik için ~1.2, metal kalıp için ~2.25 aralığında çıkar; böylece aynı hücrede metal kalıpta soğuk birleşme riski kuma göre belirgin şekilde yükselir, ancak küçük baz riskleri otomatik olarak `1.0`’e kliplenmez.

## 3. Besleyici (riser) etkisi

Besleyici, parçanın bağlı olduğu bölgeye **sıcak metal rezervi** sağlar. Soğuk birleşme hesabında bunu iki şekilde dikkate alıyoruz:

- **Girdi olarak `feed_risk` varsa**: `feed_risk = 0` besleyici tarafından iyi beslenen bölge demektir. `feeder_factor = 0.2 + 0.8 * feed_risk` ile soğuk birleşme riski beslenen bölgede en fazla %80 azaltılır.
- **Sadece `feeder_mask` varsa**: En yakın besleyici vokseline olan mesafe ve yerel modül (`M_mod`) kullanılarak besleme mesafesi (`~2.5 M`) üzerinden üstel sönüm uygulanır.

Sonuç: besleyiciye yakın hücreler daha düşük soğuk birleşme riski üretir, uzak veya ince bölgeler yüksek risk üretmeye devam eder.

## 4. Kodda ne değişti?

`core/sdf_analyzer.py` içindeki `compute_cold_shot_risk` fonksiyonu güncellendi:

- `fill_delay_factor` artık global "en son dolma zamanı" yerine **ön cephe hızı / kalan sıvı süresi** (`dt_step / (t_liq - fill_time)`) ile hesaplanıyor
- Parçanın **en üst serbest yüzeyinde** `fill_delay` %35 sönümleniyor; tek cephenin durduğu yer iki soğuk cephenin karşılaşması gibi değerlendirilmiyor
- `mold_chill_factor` artık `sqrt` yerine **logaritmik** bağımlılıkla hesaplanıyor; metal kalıp riski kliplenmiyor
- Termal çözücüden gelen `temperature` alanı hâlâ likidüs üzerindeyse **sıcaklık faktörü 0** yapılıyor (superheat kapısı)
- `feeder_mask` ve `feed_risk` ile besleyici etkisi korunuyor
- Dolum/katılaşma çözücülerine dokunulmadı; sadece soğuk birleşme risk skoru değişti

## 5. Sentetik doğrulama

Bir 40×40×40 voxel küpü üzerinde, iç köşede bir besleyici ve kum/seramik/metal kalıp kombinasyonları ile test edildi.

```
Malzeme    maks  ort  besleyici_hücresi  uzak_hücre
sand       0.210 0.028 0.0000            0.176
ceramic    0.251 0.033 0.0000            0.210
metal_mold 0.472 0.062 0.0000            0.395
```

- **Kum** en düşük ortalama risk (~0.03)
- **Seramik** orta risk (~0.03)
- **Metal kalıp** en yüksek risk (~0.06; uzak hücrede ~0.4)
- **Besleyici** tüm malzemelerde riski düşürüyor; besleyici hücrelerinde risk hemen hemen sıfır
- Artık "her yer kırmızı" değil; risk gerçekten yüksek olan bölgelerde belirgin

![Soğuk birleşme karşılaştırması](validation/results/cold_shot_material_feeder.png)

## 6. Sonuç

Soğuk birleşme artık hem **kalıp malzemesine** (kum / seramik / metal) hem de **besleyicinin ısıtıcı etkisine** duyarlı. Dolum ve katılaşma simülasyonu bozulmadan sadece risk skoru güncellendi.

PR: https://github.com/josecastsuite/NEWPROJECT/pull/3

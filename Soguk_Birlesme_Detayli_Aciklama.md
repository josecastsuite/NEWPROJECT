# Soğuk Birleşme (Cold-Shut) Riski – Detaylı Formül ve Yöntem Açıklaması

Bu dökümanda `core/sdf_analyzer.py` içindeki `compute_cold_shot_risk` fonksiyonunun ve onu besleyen `analyze` akışındaki formüllerin tam hali verilmiştir. Amacımız, soğuk birleşme riskini **fiziksel olarak açıklanabilir** şekilde hesaplamak; bunu yaparken ön cephe hızını, kalan sıvı süresini, serbest yüzey sönümlemesini, süper ısı kapısını ve kalıp malzemesi/besleyici etkisini birlikte kullanmaktır.

---

## 1. `compute_cold_shot_risk` – Girdi / Çıktı

Dosya: `core/sdf_analyzer.py` satırlar `757`–`1068`.

**Girdiler:**

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `part_mask` | bool ndarray | Parça hücreleri (`True`) |
| `fill_time` | float ndarray | Her hücreye metalin ulaşma zamanı `[s]` |
| `velocity_magnitude` | float ndarray \| None | Hücredeki hız büyüklüğü `[m/s]` |
| `temperature` | float ndarray | Termal çözücüden gelen sıcaklık alanı `[°C]` |
| `t_solid` | float ndarray | Her hücrenin katılaşma zamanı `t_s` `[s]` |
| `M_mod` | float ndarray | Yerel döküm modülü `M = V/A` `[mm]` |
| `alloy` | `Alloy` | Alaşım fiziksel verisi |
| `t_pour_c`, `t_mold_c` | float | Döküm sıcaklığı ve kalıp başlangıç sıcaklığı `[°C]` |
| `dx` | float | Voxel boyutu `[mm]` |
| `origin_mm` | ndarray | Izgara kökeni `[mm]` |
| `t_liq` | float ndarray \| None | Her hücre için likidus/sıvılaşma zamanı `[s]` |
| `mold` | `MoldMaterial` \| None | Kalıp malzemesi |
| `feeder_mask` | bool ndarray \| None | Besleyici/riser hücreleri |
| `feed_risk` | float ndarray \| None | Besleme riski alanı `[0,1]` |

**Çıktı:**

- `cold_shot_risk`: her voxel için `[0,1]` arası risk.
- `last_fill_point_mm`: son dolan noktanın koordinatları `[mm]`.

---

## 2. Temel faktörler (Base Risk)

Risk, dört normalize faktörün çarpımı olarak başlar:

```
R_base = temperature_factor * fill_delay_factor * low_velocity_factor * thin_section_factor
```

Her biri `[0,1]` aralığındadır. Sonra kalıp ve besleyici düzeltmeleri uygulanır.

---

### 2.1 `fill_delay_factor` – Ön cephe hızı ve kalan sıvı süresi

Artık sadece "ne kadar geç dolduğuna" bakılmıyor. Soğuk birleşme, ön cephenin bir sonraki voksele **katılaşmadan önce** yetişip yetişemediğini ölçer.

```
# dt_step: ön cephenin bir vokseli geçme süresi [s]
if velocity_magnitude varsa:
    v_mm_s = velocity_magnitude * 1000.0          # m/s -> mm/s
    dt_step = dx / max(v_mm_s, 1e-9)
else:
    |grad_fill| = sqrt( (∂ft/∂x)^2 + (∂ft/∂y)^2 + (∂ft/∂z)^2 )   [s/mm]
    dt_step = dx * |grad_fill|

# remaining: hücre dolduktan sonra likidusa kadar geçen süre [s]
remaining = max(t_liq - fill_time, 1e-9)        # t_liq yoksa t_solid kullanılır

fill_delay = clip(dt_step / remaining, 0, 1)
fill_delay[~part_mask] = 0
```

`fill_delay` yüksekse, ön cephe yavaş veya hücre çok geç donuyor demektir; bu soğuk birleşme için uygun koşuldur.

#### 2.1a Serbest yüzey sönümlemesi

Parçanın en üst katmanı (hava ile temas eden son nokta) tek bir cephenin durduğu yerdir; burada iki soğuk cephenin karşılaşması söz konusu değildir. Bu yüzden üst cidar hücrelerinde `fill_delay` `0.65` ile çarpılır (~%35 azalma).

---

### 2.2 `low_velocity_factor` – Artık ayrı hesaplanmıyor

Ön cephe hızı zaten `fill_delay_factor` içinde `dt_step` olarak girdiğinden, aynı etkiyi iki kez saymamak için `low_velocity_factor` nötr (`1.0`) tutulur. Yavaş veya hızlı cephe riski artık sadece `fill_delay = dt_step / remaining` ile ölçülür.

---

### 2.3 `temperature_factor` – Metalin ulaştığı andaki sıcaklık

`t_liq` ve `t_solid`, `solve_3d_thermal` tarafından üretilir ve **döküm başlangıcından itibaren geçen mutlak zaman**lardır; `solve_3d_thermal` içinde `fill_time` eklenmiştir. Yani `t_liq - fill_time`, hücrenin dolduktan sonra ne kadar süre sıvı kalabileceğidir.

Artık `t_liq_surf` / `t_sol_surf` doğrudan `t_liq` / `t_solid` değerleridir; `surface_scale` gibi yapay küçültme yok.

**Sabitler:**

```
t_liq_c = alloy.t_liquidus_c
t_sol_c = alloy.t_solidus_c
T_high  = t_liq_c + 30.0
T_low   = t_sol_c
```

`T_meet`, metalin `fill_time` anındaki sıcaklığıdır. Üç segment:

```
# Segment 1: Dolum henüz likidus zamanından önce
if fill_time < t_liq:
    T_meet = t_pour_c - (t_pour_c - t_liq_c) * (fill_time / max(t_liq, 1e-9))

# Segment 2: Likidus ile solidus arasında
if t_liq <= fill_time < t_solid:
    T_meet = t_liq_c - (t_liq_c - t_sol_c) * ((fill_time - t_liq) / max(t_solid - t_liq, 1e-9))

# Segment 3: Solidus zamanından sonra
if fill_time >= t_solid:
    T_meet = t_sol_c
```

`T_meet` daha sonra `[t_mold_c, t_pour_c]` aralığına kısıtlanır.

**Sıcaklık faktörü:**

```
temperature_factor = clip((T_high - T_meet) / (T_high - T_low), 0, 1)
temperature_factor[~part_mask] = 0
```

- `T_meet` yüksekse (sıcak metal) → faktör `0`.
- `T_meet` `t_sol_c`’nin altına düşerse → faktör `1`.

#### 2.3a Süper ısı (superheat) kapısı

Termal çözücü `temperature` alanı hâlâ `t_liq_c` üzerindeyse, metal henüz likidusunu geçmemiştir ve soğuk birleşme olamaz:

```
if temperature > t_liq_c:
    temperature_factor = 0
```

Bu, üst bölgelerin sadece "daha geç dolduğu" için kırmızı olmasını engeller; gerçekten soğumuş olmaları gerekir.

#### 2.3b `t_liq` yoksa (fallback)

Termal çözücü çıktısı yoksa, basit soğuma oranı kullanılır:

```
cooling_rate[part_mask] = (t_pour_c - t_sol_c) / max(t_solid[part_mask], 1e-9)
T_meet = t_pour_c - fill_time * cooling_rate
```

---

### 2.4 `thin_section_factor` – İnce kesit etkisi

Yerel modül `M_mod` küçükse (ince cidar), katılaşma hızlı olur ve soğuk birleşme riski artar.

```
m_ref = percentile(M_mod[part_mask & M_mod>0], 10)   # 10. yüzdelik referans modül
thin[part_mask] = M_mod[part_mask]
thin = m_ref / max(thin, m_ref)
thin = clip(thin, 0, 1)
thin[~part_mask] = 0
```

İnce kesitte `M_mod` küçük, payda büyük, `thin` `1`’e yakın.

---

## 3. Kalıp malzemesi faktörü – `mold_chill_factor`

Soğuk birleşme, metal–kalıp arayüzünde ne kadar hızlı ısı çekildiğine bağlıdır. Bunu en iyi ölçen büyüklük **termal effusivity**dir:

```
e = sqrt(k * rho * cp)    [J m^-2 K^-1 s^-0.5]
```

| Malzeme | k [W/m·K] | rho [kg/m³] | cp [J/kg·K] | e |
|---------|-----------|-------------|-------------|---|
| Kum (green sand) | 0.58 | 1600 | 1170 | ~1040 |
| Seramik | 1.20 | 2000 | 1000 | ~1550 |
| Metal (çelik kalıp) | 45.0 | 7850 | 460 | ~12700 |

Kod (`core/sdf_analyzer.py:1016–1036`):

```python
k_m   = float(getattr(mold, "k_w_mk", 0.0))
rho_m = float(getattr(mold, "rho_kg_m3", 0.0))
cp_m  = float(getattr(mold, "cp_j_kgk", 0.0))

if k_m > 0 and rho_m > 0 and cp_m > 0:
    e_m   = math.sqrt(k_m * rho_m * cp_m)
    e_ref = math.sqrt(0.58 * 1600.0 * 1170.0)   # green-sand referansı
    ratio = max(e_m / e_ref, 0.25)
    # Doymayan logaritmik bağımlılık
    mold_chill_factor = 1.0 + 0.5 * math.log(ratio)
else:
    alpha = float(getattr(mold, "diffusivity_mm2_s", 0.0) or 0.0)
    if alpha > 0.0:
        alpha_ref = 0.31
        mold_chill_factor = 1.0 + math.log1p(alpha / alpha_ref) * 0.5

mold_chill_factor = clip(mold_chill_factor, 0.5, 3.5)
```

**Örnek değerler:**

- Kum: `e/e_ref ≈ 1.0` → `mold_chill_factor ≈ 1.00`
- Seramik: `e/e_ref ≈ 1.49` → `mold_chill_factor ≈ 1.20`
- Metal kalıp: `e/e_ref ≈ 12.2` → `mold_chill_factor ≈ 2.25`

Son çarpan:

```
cold_shot_gain = getattr(alloy, "cold_shot_gain", 1.25)
scale = cold_shot_gain * mold_chill_factor
```

Logaritmik form sayesinde metal kalıp riski 3.5 katı aşmaz; küçük `R_base` değerleri otomatik olarak `1.0`’e kliplenmez.

---

## 4. Besleyici faktörü – `feeder_factor`

Soğuk birleşme riski, bir riser/besleyici tarafından sıcak metalle beslenen bölgede düşürülür.

### 4a. `feed_risk` varsa (tercih edilen yol)

`analyze` içinde `feed_risk` şu şekilde hesaplanır (`core/sdf_analyzer.py:3404–3408`):

```
FD_field = alloy.feed_k1 * (2.0 * M_mod) * riser_factor_field
feed_risk = dist_feed / (dist_feed + max(FD_field, 1.0))
feed_risk = clip(feed_risk, 0, 1)
```

- `dist_feed`: hücreden en yakın besleyiciye **metal içinden** geodesik mesafe `[mm]` (`feeding_distance_dijkstra`).
- `FD_field`: besleyicinin etkili besleme mesafesi. `feed_k1` genellikle `4.5`, `M_mod` yerel modül, `riser_factor_field` besleyici boyut faktörü.
- `feed_risk = 0`: iyi besleniyor.
- `feed_risk = 1`: besleyici etkisi ulaşmıyor.

Soğuk birleşmede bu riski şu şekilde tersine çeviririz:

```
feeder_factor = 0.2 + 0.8 * feed_risk
```

- `feed_risk = 0` (besleyici dibinde) → `feeder_factor = 0.2` → risk **%80 azalır**.
- `feed_risk = 1` (besleyici etki alanı dışında) → `feeder_factor = 1.0` → değişmez.

### 4b. Sadece `feeder_mask` varsa (fallback)

Eğer `feed_risk` gelmediyse, en yakın besleyici vokseline olan Euclidean mesafe kullanılır:

```
dist_to_feeder = distance_transform_edt(~feeder_mask, sampling=dx)
L_feed = max(M_mod * 2.5, dx)   # etkili besleme mesafesi ~2.5 M

feeder_factor = 1.0 - 0.8 * exp(-dist_to_feeder / L_feed)
feeder_factor[~part_mask] = 1.0
```

`L_feed` ince kesitte küçük, kalın kesitte büyük olur; yani kalın bir bölgede besleyici etkisi daha uzağa gider.

---

## 5. Nihai risk

```
cold_shot_gain = getattr(alloy, "cold_shot_gain", 1.25)
scale = cold_shot_gain * mold_chill_factor
cold_shot_risk = clip(R_base * scale * feeder_factor, 0, 1)
```

Her adımın amacı:

1. **Base risk** (`R_base`): Metalin soğuması, ön cephe hızı ve ince kesit bir araya gelince risk oluşur.
2. **Kalıp faktörü** (`mold_chill_factor`): Metal kalıp daha hızlı soğutur, ama logaritmik doyma sayesinde aşırı büyütme yapmaz.
3. **Besleyici faktörü** (`feeder_factor`): Sıcak metal rezervi yakınında risk azalır.

---

## 6. `feed_risk` kaynağı – `analyze` içinde

`analyze` fonksiyonu (`core/sdf_analyzer.py:3404`):

1. `feeding_distance_dijkstra(is_metal, feeder_mask, dx, ...)` ile `dist_feed` hesaplanır.
2. `riser_factor_field`, en yakın besleyicinin boyut faktörüdür (`factor_by_id` ve `nearest_riser` ile, satırlar `3400–3402`).
3. `FD_field = alloy.feed_k1 * 2 * M_mod * riser_factor_field`.
4. `feed_risk = dist_feed / (dist_feed + max(FD_field, 1.0))`.

Bu alan aynı zamanda `compute_pore_size`’a (porozite) ve `compute_cold_shot_risk`’e beslenir; böylece besleyici hem poroziteyi hem soğuk birleşmeyi tutarlı şekilde etkiler.

---

## 7. `feeding_distance_dijkstra` – Kısa açıklama

`core/sdf_analyzer.py` içinde `feeding_distance_dijkstra` (aramada bulunabilir) metal vokselleri üzerinde 26-komşuluk bir graf oluşturur ve her hücreden en yakın `feeder_mask` hücresine Dijkstra maliyetini hesaplar. Maliyet, aradaki voksel boyutuna `dx` bağlıdır; böylece gerçek yol mesafesi elde edilir.

Çıktı `dist_feed` her voxel için `[mm]` cinsinden en kısa metal içi mesafedir.

---

## 8. Örnek sayısal sonuç (sentetik 40³ küp)

| Malzeme | `mold_chill_factor` | `feed_risk=0` yakını | Maksimum risk | Ortalama risk | Uzak hücre |
|---------|---------------------|----------------------|---------------|---------------|------------|
| Kum | 1.00 | 0.0000 | 0.210 | 0.028 | 0.176 |
| Seramik | 1.20 | 0.0000 | 0.251 | 0.033 | 0.210 |
| Metal kalıp | 2.25 | 0.0000 | 0.472 | 0.062 | 0.395 |

Bu tablo, aynı geometri ve aynı besleyici için **kalıp malzemesinin riski nasıl farklılaştırdığını** gösterir. Artık "her yer kırmızı" değil; metal kalıpta risk kumdan yaklaşık 2.2 kat, seramikten 1.9 kat daha yüksek.

---

## 9. Dosya ve satır referansları

- `compute_cold_shot_risk`: `core/sdf_analyzer.py:757`
- `mold_chill_factor` hesabı: `core/sdf_analyzer.py:1097–1115`
- `feeder_factor` (`feed_risk` yolu): `core/sdf_analyzer.py:1128–1133`
- `feeder_factor` (`feeder_mask` yolu): `core/sdf_analyzer.py:1134–1148`
- `feed_risk` üretimi: `core/sdf_analyzer.py:3404–3408`
- Termal girdi `t_liq`, `t_s`: `core/sdf_analyzer.py:2900`
- `mold` verisi: `core/materials_data/molds.json`

---

## 10. Kısaca fiziksel gerekçe

- **Termal effusivity** `e = sqrt(k·ρ·cp)`: Arayüzdeki ani ısı çekme hızını belirler. Metal kalıp `e`’si kumdan ~12 kat büyüktür; bu da soğuk birleşmeyi çok daha olası kılar.
- **Besleyici etkisi**: Riser, katılaşmakta olan bölgeye hâlâ sıvı metal sağlar. Bu yüzden soğuk birleşme önündeki hücrelerde risk düşer; ancak besleme mesafesi dışında (`feed_risk` → 1) etki kaybolur.
- **İnce kesit + geç dolma**: Bu iki faktör, `R_base` içinde sıcak metalin yetişemeyeceği kadar hızlı soğuma ihtimalini ölçer.

Bu yapı, soğuk birleşme riskini **dolum sonuçlarını bozmadan**, sadece risk skorunu daha gerçekçi hale getirecek şekilde günceller.

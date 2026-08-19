# Soğuk Birleşme (Cold-Shut) Riski – Güncel Formül ve Yöntem Açıklaması

Bu dökümanda `core/sdf_analyzer.py` içindeki `compute_cold_shot_risk` fonksiyonunun güncel hali açıklanır. Hedef, soğuk birleşme riskini **fiziksel olarak açıklanabilir** şekilde hesaplamak; bunu yaparken Feng–Liao dolum cephesi filtresi, Kashiwai `f_s = 0.52` eşiği, Scheil katı oranı (partition coefficient), vektörel dot-product confluence, kalıp malzemesi ve besleyici etkisini birlikte kullanmaktır.

---

## 1. Girdi / Çıktı

Dosya: `core/sdf_analyzer.py`, `compute_cold_shot_risk`.

**Girdiler:**

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `part_mask` | bool ndarray | Parça hücreleri (`True`) |
| `fill_time` | float ndarray | Her hücreye metalin ulaşma zamanı `[s]` |
| `velocity_m_s` | float ndarray | 3-B hız vektör alanı `[m/s]` (3, nx, ny, nz) |
| `velocity_magnitude` | float ndarray \| None | Hız büyüklüğü `[m/s]` (yedek) |
| `M_mod` | float ndarray | Yerel döküm modülü `V/A` `[mm]` |
| `temperature` | float ndarray | Termal çözücüden son sıcaklık `[°C]` |
| `alloy` | `Alloy` | Alaşım fiziksel verisi |
| `mold` | `MoldMaterial` \| None | Kalıp malzemesi |
| `feeder_mask` / `feed_risk` | bool/float ndarray \| None | Besleyici etkisi |

**Çıktı:**

- `cold_shot_risk`: her voxel için `[0,1]` arası risk.
- `last_fill_point_mm`: son dolan noktanın koordinatları `[mm]`.

---

## 2. Chvorinov sabiti

```
C_ch = chvorinov_c_from_properties(alloy, mold)   [dk/cm²]
```

```
C = [ (rho_m * L_eff) / ((T_m - T_0) * sqrt(pi * k_s * rho_s * c_s)) ]^2
L_eff = L + cp_m * max(t_pour - t_liq, 0)
```

Bu sabit kalıp malzemesine göre değişir: kumda büyük (yavaş soğur), metal kalıpta küçük (hızlı soğur).

---

## 3. Scheil / yerel katı oranı (Kashiwai f_s = 0.52)

Kashiwai'ye göre metalurjik kilitlenme kritik katı oranı `f_s = 0.52` civarı başlar. Scheil denklemi:

```
f_s = 1 - ((T - T_sol) / (T_liq - T_sol))^(1 / (1 - k_partition))
```

`k_partition` alaşıma özgüdür; çelik ve alüminyumun katılaşma aralığını doğru ayırmak için zorunludur. `f_s = 0.52` için:

```
ratio_fs52 = 1 - 0.48^(1 - k_partition)
T_52       = T_liq - ratio_fs52 * (T_liq - T_sol)
H_total    = cp_m * (t_pour - t_sol) + L
H_to_52    = cp_m * (t_pour - T_52) + 0.52 * L
frac_to_52 = H_to_52 / H_total
```

## 4. Yerel katılaşma zamanları

```
M_cm         = max(M_mod, 0.001) / 10            [cm]
t_cool_local = C_ch * M_cm² * 60                 [s]
t_liq_local  = C_ch * M_cm² * 60 * frac_to_52    [s]
```

`M_mod` küçükse (ince cidar) her iki zaman da küçülür; kalın bölgelerde büyür. `t_cool_local`, dolum anındaki metal sıcaklığı için referans oluşturur.

## 5. Dolum anındaki metal sıcaklığı (yerel kalınlığa göre)

```
ratio  = clip(fill_time / t_cool_local, 0, 1)
T_meet = t_pour - (t_pour - t_liq) * ratio
T_meet = clip(T_meet, t_mold, t_pour)
```

Kalın kesitler süper ısısını korur (`ratio` küçük, `T_meet` yüksek); ince / geç dolan bölgeler likidüse yaklaşır. Artık **global** değil, **yerel** soğuma zamanı kullanılır.

## 6. Sıcaklık faktörü (superheat)

```
safe_super = max(superheat / 3, 10)   ; superheat = t_pour - t_liq
dT = max(T_meet - t_liq, 0)
temperature_factor = exp(-dT / safe_super)
```

Metal hâlâ likidüsün üzerindeyse faktör 1’den küçük; likidüs altındaysa 1.

## 7. Yerel katılaşma penceresi faktörü

```
solid_time_factor = exp( - (t_liq_local / t_cool_local) * (1 + (v_local / v_crit_cold)²) )
```

`v_local`, gerçek hız vektörünün büyüklüğüdür. Hızlı hareket eden metal hücrenin soğumasını geciktirdiğinden `(1 + (v/vcrit)²)` terimiyle katılaşma riski düşürülür.

## 8. Feng & Liao dolum cephesi filtresi (voxel staircasing kaldırma)

`fill_time` alanına önce 3-B Gaussian blur (`sigma=1` voksel) uygulanır, sonra gradyan büyüklüğü hesaplanır:

```
ft_blur = GaussianBlur(fill_time, sigma=1)
grad_mag = |∇ft_blur|
ridge    = 1 - exp(-grad_mag / mean(grad_mag))
```

Yüksek gradyan = sert dolum cephesi (soğuk birleşme adayı). Gaussian yumuşatma, eğimli yüzeylerdeki voxel basamaklanmasının oluşturduğu sahte tepe çizgilerini süzer.

## 9. Dolum cephesi hızı (front speed)

`fill_time` gradyanından cephe hızı çıkarılır; son statik hız alanı (dolduktan sonra sıfıra yakın) yerine daha fiziksel bir ölçü verir:

```
v_front = (dx / 1000) / max(|∇ft_blur|, 1e-9)   [m/s]
v_front = clip(v_front, 0, 10)
```

`dx` mm cinsinden voksel boyudur; `/1000` ile metreye çevrilir. Düz, tek cepheli akışta gradyan küçük, `v_front` büyük; karşı cephelerin birleştiği yerde gradyan büyük, `v_front` küçük.

## 10. Durgunluk / konveksiyon faktörü

```
v_eff       = max(v_local, v_front)
low_vel     = exp( -(v_eff / v_crit_cold)^4 )
v_crit_cold = max(0.10, 0.45 * alloy.critical_entrainment_velocity_m_s)
```

`v_eff`, gerçek akış hızı ile dolum cephesi hızından büyük olanı seçer. Böylece “doldu ama hâlâ hareketli” bölgeler cezalandırılmaz; yavaş / durgun veya karşı cephe birleşmeleri yüksek risk üretir.

## 11. Vektörel dot-product confluence kapısı

Her hücre için komşu yüz hücrelerindeki hız vektörlerinin normalize iç çarpımı incelenir:

```
dot = (v_i · v_j) / (|v_i| * |v_j|)
min_dot = min(dot) over 6 face neighbours
confluence = clip( -(min_dot + 0.5), 0, 0.5 ) * 2
confluence = confluence * ridge
```

- `dot < -0.5` : zıt yönlü iki cephe kafa kafaya çarpışıyor → maksimum risk
- `dot > 0` : aynı yönde paralel akış → soğuk birleşme yok

`ridge` faktörü sayesinde confluence sadece sert dolum cephesi olan bölgelerde tetiklenir.

## 12. Nihai akış riski

```
flow_risk = low_vel * (1 + confluence)
flow_risk = clip(flow_risk, 0, 1)
```

`low_vel` ve `confluence` birlikte tek yönlü hızlı akışta riski sıfıra, yavaş ve çok yönlü birleşmede maksimuma çeker.

## 13. İnce kesit faktörü

```
m_ref = percentile(M_mod[part], 10)
thin  = m_ref / max(M_mod, m_ref)
thin  = clip(thin, 0, 1)
```

İnce kesitlerde modül küçük, `thin` 1’e yaklaşır.

## 14. Kalıp soğurma faktörü (thermal effusivity)

```
e_m   = sqrt(k_m * rho_m * cp_m)
e_ref = sqrt(0.58 * 1600 * 1170)
ratio = max(e_m / e_ref, 0.25)
mold_chill_factor = clip(1 + 0.25 * log(ratio), 0.7, 2.0)
```

`mold_chill_factor` kumda ~1.0, seramikte ~1.2, metal kalıpta ~1.6 çıkar.

## 15. Besleyici etkisi

```
feeder_factor = 0.2 + 0.8 * feed_risk
```

Besleyici dibinde risk %80’e varan oranda azalır.

## 16. Nihai soğuk birleşme riski

```
thermal_risk = temperature_factor * solid_time_factor
if temperature > t_liq:      # hâlâ sıvı kalmışsa soğuk birleşme olmaz
    thermal_risk = 0

scale = cold_shot_gain * mold_chill_factor

cold_shot_risk = clip(
    thermal_risk * flow_risk * thin * scale * feeder_factor,
    0, 1
)
```

`cold_shot_gain` alaşıma özgü bir kazançtır (varsayılan 1.25).

## 17. Gerçek Deneme_Ring.STEP matris sonuçları

Aynı ızgarada (`target_dim=120`, dört girişli halka) farklı alaşım ve kalıp malzemeleri ile çalıştırıldı:

| Alaşım | Kalıp | Max risk | Ortalama risk |
|--------|-------|----------|---------------|
| AlSi7  | Kum (sand) | 0.559 | 0.057 |
| AlSi7  | Seramik (ceramic) | 0.583 | 0.059 |
| AlSi7  | Metal kalıp | 0.862 | 0.087 |
| A356   | Metal kalıp | 0.866 | 0.088 |
| 42CrMo4| Metal kalıp | 0.909 | 0.140 |
| GGG40  | Metal kalıp | 0.860 | 0.145 |

Gözlemler:

- Aynı alaşım için **metal kalıp > seramik > kum** riski veriyor.
- **Çelik (42CrMo4) ve dökme demir (GGG40)**, Al-Si alaşımlarına göre daha yüksek ortalama risk üretiyor (daha dar katılaşma aralığı, yüksek döküm sıcaklığı).
- Risk artık tüm yüzeyi kaplayan tek düze kırmızı değil; cephe birleşme ve ince kesitlerde lokalize.

## 18. Görselleştirme

`ui/viewer.py` `show_cold_shot_risk` parça yüzeyini her zaman `[0, 1]` sabit skalasıyla `YlOrRd` renk haritasıyla çizer; düşük riskler açık sarı/turuncu, yüksek riskler kırmızı görünür. Skala her zaman görünür.

## 19. Dosya referansları

- `compute_cold_shot_risk`: `core/sdf_analyzer.py`
- Alaşım / kalıp verileri: `core/materials_data/alloys.json`, `core/materials_data/molds.json`
- Görselleştirme: `ui/viewer.py`

PR: https://github.com/josecastsuite/NEWPROJECT/pull/3

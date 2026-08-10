# Soğuk Birleşme (Cold Shut) – Güncel Sonuç Sunumu

## 1. Değişiklik özeti

`core/sdf_analyzer.py` içindeki `compute_cold_shot_risk` fonksiyonu aşağıdaki 4 bilimsel geliştirmeyi içerecek şekilde güncellendi:

1. **Feng & Liao voxel staircasing filtresi**: `fill_time` alanına 3-B Gaussian blur (`sigma=1`) uygulanıp gradyan büyüklüğü hesaplanır; sahte tepe çizgileri süzülür.
2. **Kashiwai `f_s = 0.52` yumuşak geçişi**: Scheil denklemi ile alaşımın `partition_coefficient` değeri kullanılarak `fs = 0.52`’ye ulaşma zamanı hesaplanır.
3. **Scheil partition coefficient**: Alaşım JSON’ından `k_partition` okunarak Al ve çelik/dökme demir farklı katılaşma aralıklarına göre ayrılır.
4. **Vektörel dot-product confluence**: Komşu yüzeylerdeki hız vektörlerinin iç çarpımı (`dot < -0.5`) ile kafa kafaya çarpışan cepheler tespit edilir.

Ek olarak:
- Dolum anındaki metal sıcaklığı **yerel Chvorinov soğuma zamanına** (`t_cool_local`) göre hesaplanır; böylece kalın kesitler süper ısısını korur.
- Dolum cephesi hızı (`front_speed`) `fill_time` gradyanından çıkarılır; dolduktan sonra sıfıra yakınlaşan statik hız alanı yerine gerçek cephe hızı kullanılır.
- Kalıp malzemesi etkisi `mold_chill_factor` ile `clip(1 + 0.25 log(e_m/e_ref), 0.7, 2.0)` olarak logaritmik doyumlu hesaplanır.
- Besleyici etkisi `feeder_factor = 0.2 + 0.8 * feed_risk` ile korunur.
- Dolum ve katılaşma çözücülerine dokunulmadı.

## 2. Deneme_Ring.STEP matris sonuçları

Aynı ızgara (`target_dim=120`) ve aynı geometri üzerinde farklı alaşım/kalıp kombinasyonları:

| Alaşım | Kalıp | Max risk | Ortalama risk | Süre |
|--------|-------|----------|---------------|------|
| AlSi7 | Kum (sand) | 0.559 | 0.057 | ~236 s |
| AlSi7 | Seramik (ceramic) | 0.583 | 0.059 | ~242 s |
| AlSi7 | Metal kalıp | 0.862 | 0.087 | ~235 s |
| A356 | Metal kalıp | 0.866 | 0.088 | ~229 s |
| 42CrMo4 | Metal kalıp | 0.909 | 0.140 | ~231 s |
| GGG40 | Metal kalıp | 0.860 | 0.145 | ~225 s |

**Yorum:**
- Metal kalıp > seramik > kum riski veriyor.
- Çelik (42CrMo4) ve dökme demir (GGG40), Al-Si alaşımlarına göre daha yüksek ortalama risk üretiyor.
- Risk artık tek düze kırmızı değil; cephe birleşme ve ince kesitlerde lokalize.

## 3. Ekran görüntüleri

Aşağıdaki matris, aynı kamera açısından Al/çelik/demir + kum/seramik/metal kombinasyonlarını gösterir. Renk skalası sabit `[0, 1]`: açık sarı = düşük risk, kırmızı = yüksek risk.

![Soğuk birleşme matrisi](validation/results/cold_shot_matrix_v3.png)

## 4. Doğrulama

- `python -m py_compile core/sdf_analyzer.py core/materials.py ui/viewer.py` OK
- `python -m pytest tests/ -q` 9/9 passed
- Sentetik test: iyi hizalanmış besleyici dibinde risk ~0; karşı cephelerde maksimum risk.

## 5. PR

https://github.com/josecastsuite/NEWPROJECT/pull/3

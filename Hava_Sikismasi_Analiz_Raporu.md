# Hava Sıkışması (Air Entrapment) Kalıp Tipi Duyarlılık Analizi

## 1. Bulgu: Hava sıkışması kalıp tipinden bağımsız çalışmıyordu

C++ LBM/VOF çözücüsünün `detect_entrapment()` fonksiyonu, kapanmış hava ceplerini sadece **çıkış (outlet) bağlantısına** göre belirliyordu. Kalıp malzemesinin geçirgenliği (kum mu, metal mi, seramik mi) devreye girmiyordu. Bu yüzden **kum, seramik ve metal kalıpta aynı hava-sıkışması haritası** çıkıyordu; bu fiziksel olarak yanlış.

Fiziksel gerçek:
- **Metal / seramik kalıp**: Hava kalıp gövdesine kaçamaz, sadece yalancı/vent üzerinden çıkar. Kapanmış hava cebi neredeyse tamamen kalır → risk yüksek.
- **Kum kalıp**: Hava, kumun gözenekleri sayesinde kalıp duvarına yakın ceplerden kısmen sızabilir. Yüzeye yakın cepler boşalır, derin cepler kalır → risk daha düşük.

## 2. Düzeltme

`core/filling_solver.py` içinde C++'dan gelen ikili `trap` maskesine sonradan bir **kalıp geçirgenliği düzeltmesi** uygulandı:

```python
if mold is not None and getattr(mold, "is_sand", True):
    perm_eff = float(np.clip(getattr(mold, "permeability_proxy", 1.0), 0.0, 1.0))
    if perm_eff > 1e-9:
        dist_to_surface_mm = ndimage.distance_transform_edt(fine_metal, sampling=orig_dx)
        vent_depth_mm = 2.0 + 20.0 * perm_eff
        base_escape = 0.1 + 0.25 * perm_eff
        escape_factor = base_escape + (perm_eff - base_escape) * np.exp(
            -dist_to_surface_mm / max(vent_depth_mm, 1e-3)
        )
        air_entrapment_fine = np.where(
            fine_metal,
            np.clip(air_entrapment_fine * (1.0 - escape_factor), 0.0, 1.0),
            0.0,
        )
```

- Kum kalıpta yüzeye yakın hücreler `escape_factor` büyük olduğu için değer düşer (hava sızar).
- Seramik/metal kalıpta `is_sand=False` olduğu için bu dal çalışmaz; C++'dan gelen tam hava-sıkışması korunur.

## 3. Test Sonuçları (Model_Knuckle_Dusuk.STEP, 42CrMo4)

| Kalıp Tipi | max risk | cells > 0.3 | Hacim (m³) | Centroid (mm) | Süre (s) |
|---|---|---|---|---|---|
| Kum Kalıp | 0.3562 | 100 | 5.408e-06 | [60.746072035537, 308.6891304517503, 917.6019703120882] | 40.4 |
| Seramik Kalıp | 1.0000 | 664 | 2.669e-05 | [54.94609448669232, 307.19865869728255, 919.2709291820673] | 39.6 |
| Metal Kalıp | 1.0000 | 664 | 2.669e-05 | [54.94609448669232, 307.19865869728255, 919.2709291820673] | 39.7 |

## 4. Yorum

- **Kum Kalıp**: Riskli hücre ve hacim metal/seramiğe göre belirgin şekilde düşük; sadece derinde kalan cepler kalıyor.
- **Seramik Kalıp & Metal Kalıp**: Hava kaçamadığı için risk maksimum (`max = 1.0`) ve hacim çok daha yüksek.
- Bu, dökümcülük prensibine uygun: hava sıkışması en çok metal ve seramik kalıplarda (yetersiz havalandırmada) görülür.

## 5. UI Entegrasyonu

`ui/main_window.py` üzerinde bağımsız **"Hava Sıkışması"** checkbox'ı ve `ui/viewer.py` üzerinde 0.3 eşik değeriyle `cool` renk haritası, en büyük cep merkezinde cyan küre göstergeci mevcuttur. Düzeltme sonrası aynı modelde kalıp tipi değiştirildiğinde farklı sonuçlar görsel olarak da farklı olacaktır.

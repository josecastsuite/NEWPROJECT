# JoseCast Analyzer - Hava Sikismasi (Air Entrapment) Duzeltme Raporu

**Tarih:** 2026-08-05  
**Dal:** `devin/1784480540-p0-p1-fixes`  
**PR:** https://github.com/josecastsuite/NEWPROJECT/pull/1

## 1. Hata Ozeti

Program seramik/metal kalip secildiginde **hava sikismasi riskini gostermiyordu** (`air_entrapment` sifir, `trapped_air_volume_m3 = 0`). Aslinda LBM/VOF cozucu `trap` dizisi uretiyordu ama:

1. `solve_filling_flow` bu `trap` dizisini `FillingResult.air_entrapment` / `trapped_air_volume_m3` / `air_entrapment_centroid_mm` alanlarina kopyalamiyordu.
2. LBM cikis maskesi `_select_vent_cells` her kalip turunde **parca / besleyici ust yuzeyini** vent (hava kacisi) olarak isaretliyordu. Seramik/metal kalip gibi kapali dokumlerde bu fiziksel olarak yanlisti; hava kapan kapali cepler gercek risk olarak cikmiyordu.

## 2. Cozum

### 2.1 `core/filling_solver.py` degisiklikleri

- `_select_lbm_outlet_cells(...)` yardimci fonksiyonu eklendi:
  - **Kum kalip** (`is_sand=True`): eski gibi PART/RISER ust yuzeyleri vent.
  - **Seramik/metal/exothermic/investment** (`is_sand=False`): sadece `RISER` veya `CURUFLUK` gibi gercek acik govde ust yuzeyleri vent. Parca veya dokum ust yuzeyi hatali vent olarak kullanilmiyor.
  - Dokum hunisi / sagu (sprue/pouring basin) ustune **serbest outlet** konmuyor; aksi halde sivi metal oradan bosalip kaliptaki bosluklar doldurulmuyordu.

- LBM/VOF sonucundaki `air_entrapment` (trap) dizisi:
  - `order=0` (en yakin komsu) ile orijinal ince `orig_grid` izgarasina yeniden orneklendi.
  - Sadece metal hucresi (`fine_metal`) icinde degerlendirildi.
  - `permeability_proxy` ile gecirgenlik duzeltmesi uygulandi: kum kalip (`proxy ~1`) yuzeye yakin havayi kismen kacirir; seramik (`0.2`) ve metal (`0.02`) kalip hapsolmus havayi korur.

- Hapsolmus hava hacmi ve agirlik merkezi hesaplandi:
  - `trapped_air_volume_m3 = trapped_voxel_count * (orig_dx_m^3)`
  - `air_entrapment_centroid_mm = orig_origin + mean(trapped_voxel_indices) * orig_dx`

- `FillingResult` donusune uc alan eklendi:
  - `air_entrapment`
  - `trapped_air_volume_m3`
  - `air_entrapment_centroid_mm`

### 2.2 Bagimli moduller

- `ui/viewer.py` zaten `result.air_entrapment` bekliyordu, bu yuzden harita ek bir degisiklik olmadan calisir.
- `core/sdf_analyzer.py` akar field boyutu esitse `flow_result_for_thermal.air_entrapment`i sonuca aktariyor.

## 3. Dogrulama

### 3.1 Sentetik kapali cep testi

Yatay sagu + tabandan girisli kapali kutu (tavan saglam, tek cikis yok) geometrisi ile test edildi:

| Kalip | `air_entrapment` max | `trapped_air_volume_m3` | Yorum |
|---|---|---|---|
| Kum kalip (`silica_sand`) | 0.0 | 0.0 | Ust yuzey parting/vent kabul edildi, hava kacti. |
| Seramik kalip (`ceramic`) | 0.84 | 1.25e-05 | Kapali cep hapsolmus hava olarak tespit edildi. |

Bu, seramik/metal kalip kullanildiginda programin artik hava sikismasi gosterebildigini dogruluyor.

### 3.2 Diger testler

- `python -m py_compile` tum `.py` dosyalarinda basarili.
- `validation/run_validation.py` 5/5 STEP modelde tamamlandi, `flow_result` uretildi, animasyon verileri bozulmadi.

## 4. Kalan Bilinen Davranislar

- Kum kalip modunda **part ust yuzeyi** hala otomatik vent olarak kabul ediliyor. Bu yuzeysel hava cepleri icin kabul edilebilir, ama **kum icerisinde derin bir kapan kapi varsa** yine de hava sikismasi olabilir; gercek ventgovde `RISER`/`CURUFLUK` olarak tanimlanmalidir.
- Hava sikismasi haritasi, LBM/VOF cozucusunun gercekten calistigi durumlarda uretilir. "Hizli akis hesabi (animasyon yok)" seceneginde 3-B cozucu devre disi kalir, bu yuzden `air_entrapment` sifir olur.

## 5. Sonuc

Seramik ve metal gibi kapali kalip sistemlerinde hava sikismasi riski artik LBM/VOF ciktisindan `ui/viewer.py` haritasina ve rapor alanlarina duzgun sekilde iletiliyor. Mevcut Chvorinov ve Darcy/animasyon duzeltmeleriyle birlikte butun birim/formul zinciri tutarli hale geldi. Benim onerim budur.

# JoséCast Batch Validation Report
Models: 4
Total issues: 4

| Model | Bodies | Grid | dx (mm) | Fill (s) | max Re | Visible HS | Issues |
|-------|--------|------|---------|----------|--------|------------|--------|
| Deneme_Ring.STEP | 9 | [159, 87, 168] | 2.8662 | 0.27 | 38427 | 1 | 1 unresolved hot spot(s) |
| Knuckle.STEP | 4 | [151, 168, 151] | 1.4775 | 0.05 | 30439 | 3 | 3 unresolved hot spot(s) |
| Model_Knuckle_Dusuk.STEP | 11 | [90, 168, 93] | 2.5691 | 0.73 | 31261 | 1 | 1 unresolved hot spot(s) |
| Parca1.STEP | 5 | [169, 100, 123] | 1.3153 | 0.20 | 28523 | 1 | 1 unresolved hot spot(s) |

## Details

### Deneme_Ring.STEP
```json
{
  "model": "Deneme_Ring.STEP",
  "bodies": 9,
  "grid_shape": [
    159,
    87,
    168
  ],
  "dx_mm": 2.866198376503409,
  "elapsed_s": 78.02590465545654,
  "chvorinov_c": 2.099078939161428,
  "flow": {
    "fill_time_s": 0.26581481630443804,
    "Q_m3_s": 0.001494675140441415,
    "inlet_area_cm2": 9.964500936276101,
    "ingate_contact_velocity_m_s": 0.6967965655023449,
    "max_velocity_m_s": 0.696796715259552,
    "mean_velocity_m_s": 0.6967965960502625,
    "filter_recommendation": "Seramik filtre \u00f6nerisi: 'Kaynak \u2192 Body_7' b\u00f6lgesine \u00d865 mm, 20 PPI, 15 mm kal\u0131nl\u0131k; yakla\u015f\u0131k 15188 Pa ek bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc, y\u00fczey h\u0131z\u0131 ~0.45 m/s.",
    "reason": "C++ LBM D3Q19 dolum: giri\u015f 'SPRUE_THROAT', Q=89.68 L/dak, kaynak h\u0131z\u0131=1.500 m/s, tahmini doldurma s\u00fcresi=0.27 s, bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc=21.2 Pa.",
    "node_velocities": {
      "SPRUE_THROAT": 1.5,
      "SPRUE_BASE": 0.6967956362973254,
      "INGATE": 0.6967965655023449
    },
    "max_reynolds": 38427.3984375,
    "mean_reynolds": 6695.9921875,
    "max_turbulence_intensity_pct": 13.08265495300293
  },
  "thermal": {
    "total_hotspots": 1,
    "visible_hotspots": 1,
    "max_pore_um": 222.67880638090156
  },
  "riser_proposals": [
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; Darcy bas\u0131n\u00e7 kayb\u0131 kesme; Heuver \u00e7emberleri bozuk; y\u00f6nl\u00fc kat\u0131la\u015fma bozuk; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (-9.8, 2.7, -8.0) cm, normal=(0.71,0.00,0.71)"
    }
  ],
  "gate_exists": true,
  "issues": [
    "1 unresolved hot spot(s)"
  ]
}
```
### Knuckle.STEP
```json
{
  "model": "Knuckle.STEP",
  "bodies": 4,
  "grid_shape": [
    151,
    168,
    151
  ],
  "dx_mm": 1.4775000000000003,
  "elapsed_s": 53.86274266242981,
  "chvorinov_c": 2.099078939161428,
  "flow": {
    "fill_time_s": 0.04636517859240012,
    "Q_m3_s": 0.0004710436194935888,
    "inlet_area_cm2": 3.1402907966239253,
    "ingate_contact_velocity_m_s": 1.499999999999999,
    "max_velocity_m_s": 1.5,
    "mean_velocity_m_s": 1.5,
    "filter_recommendation": "Seramik filtre \u00f6nerisi: 'Kaynak \u2192 Body_4' b\u00f6lgesine \u00d837 mm, 20 PPI, 15 mm kal\u0131nl\u0131k; yakla\u015f\u0131k 15188 Pa ek bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc, y\u00fczey h\u0131z\u0131 ~0.45 m/s.",
    "reason": "C++ LBM D3Q19 dolum: giri\u015f 'SPRUE_THROAT', Q=28.26 L/dak, kaynak h\u0131z\u0131=1.500 m/s, tahmini doldurma s\u00fcresi=0.05 s, bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc=1308.7 Pa.",
    "node_velocities": {
      "SPRUE_THROAT": 1.5,
      "INGATE": 1.499999999999999
    },
    "max_reynolds": 30439.1953125,
    "mean_reynolds": 4961.30908203125,
    "max_turbulence_intensity_pct": 15.049436569213867
  },
  "thermal": {
    "total_hotspots": 3,
    "visible_hotspots": 3,
    "max_pore_um": 1000.468268954902
  },
  "riser_proposals": [
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; sistemde riser yok; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (-5.4, 6.8, -35.2) cm, normal=(0.00,0.00,1.00)"
    },
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; sistemde riser yok; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (-1.8, -0.1, -32.4) cm, normal=(0.00,0.00,1.00)"
    },
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; sistemde riser yok; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (-8.8, 8.3, -34.9) cm, normal=(0.00,0.00,1.00)"
    }
  ],
  "gate_exists": true,
  "issues": [
    "3 unresolved hot spot(s)"
  ]
}
```
### Model_Knuckle_Dusuk.STEP
```json
{
  "model": "Model_Knuckle_Dusuk.STEP",
  "bodies": 11,
  "grid_shape": [
    90,
    168,
    93
  ],
  "dx_mm": 2.5691015624999567,
  "elapsed_s": 108.56621146202087,
  "chvorinov_c": 2.099078939161428,
  "flow": {
    "fill_time_s": 0.7310872184224103,
    "Q_m3_s": 0.0008692629270489904,
    "inlet_area_cm2": 5.795086180326603,
    "ingate_contact_velocity_m_s": 0.5866155767260667,
    "max_velocity_m_s": 0.803970456123352,
    "mean_velocity_m_s": 0.8039703965187073,
    "filter_recommendation": "Seramik filtre \u00f6nerisi: 'Body_8 \u2192 Body_9' b\u00f6lgesine \u00d850 mm, 20 PPI, 15 mm kal\u0131nl\u0131k; yakla\u015f\u0131k 15188 Pa ek bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc, y\u00fczey h\u0131z\u0131 ~0.45 m/s.",
    "reason": "C++ LBM D3Q19 dolum: giri\u015f 'SPRUE_THROAT', Q=52.16 L/dak, kaynak h\u0131z\u0131=1.500 m/s, tahmini doldurma s\u00fcresi=0.73 s, bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc=12.3 Pa.",
    "node_velocities": {
      "SPRUE_THROAT": 1.5,
      "RUNNER": 1.5000116914598758,
      "DISTRIBUTOR": 0.8039704673460483,
      "INGATE": 0.5866155767260667
    },
    "max_reynolds": 31260.587890625,
    "mean_reynolds": 5021.7373046875,
    "max_turbulence_intensity_pct": 10.76789665222168
  },
  "thermal": {
    "total_hotspots": 1,
    "visible_hotspots": 1,
    "max_pore_um": 1200.0
  },
  "riser_proposals": [
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; Darcy bas\u0131n\u00e7 kayb\u0131 kesme; Heuver \u00e7emberleri bozuk; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (-1.8, -0.5, 97.7) cm, normal=(0.00,0.00,1.00)"
    }
  ],
  "gate_exists": true,
  "issues": [
    "1 unresolved hot spot(s)"
  ]
}
```
### Parca1.STEP
```json
{
  "model": "Parca1.STEP",
  "bodies": 5,
  "grid_shape": [
    169,
    100,
    123
  ],
  "dx_mm": 1.3153124999999997,
  "elapsed_s": 38.17332744598389,
  "chvorinov_c": 2.099078939161428,
  "flow": {
    "fill_time_s": 0.1963087029808457,
    "Q_m3_s": 0.0014774095572275057,
    "inlet_area_cm2": 9.849397048183372,
    "ingate_contact_velocity_m_s": 3.02237997689804,
    "max_velocity_m_s": 3.0223798751831055,
    "mean_velocity_m_s": 3.022379159927368,
    "filter_recommendation": "Seramik filtre \u00f6nerisi: 'Body_4 \u2192 Body_3' b\u00f6lgesine \u00d865 mm, 20 PPI, 15 mm kal\u0131nl\u0131k; yakla\u015f\u0131k 15188 Pa ek bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc, y\u00fczey h\u0131z\u0131 ~0.45 m/s.",
    "reason": "C++ LBM D3Q19 dolum: giri\u015f 'SPRUE_THROAT', Q=88.64 L/dak, kaynak h\u0131z\u0131=1.500 m/s, tahmini doldurma s\u00fcresi=0.20 s, bas\u0131n\u00e7 d\u00fc\u015f\u00fcm\u00fc=1805.3 Pa.",
    "node_velocities": {
      "SPRUE_THROAT": 1.5,
      "RUNNER": 3.0223799768980406,
      "INGATE": 3.02237997689804
    },
    "max_reynolds": 28522.91796875,
    "mean_reynolds": 4356.66796875,
    "max_turbulence_intensity_pct": 9.899985313415527
  },
  "thermal": {
    "total_hotspots": 1,
    "visible_hotspots": 1,
    "max_pore_um": 30.63401108401552
  },
  "riser_proposals": [
    {
      "shape": "cylinder",
      "diameter_mm": 0.0,
      "height_mm": 0.0,
      "volume_cm3": 0.0,
      "exothermic": false,
      "infeasible": true,
      "reason": "besleme mesafesi/yol yetersiz; Darcy bas\u0131n\u00e7 kayb\u0131 kesme; konvansiyonel silindirik besleyici \u00f6nerildi; \u00f6nerilen besleyici/\u00e7\u0131k\u0131c\u0131 par\u00e7a geometrisine s\u0131\u011fm\u0131yor; kullan\u0131c\u0131 karar\u0131 gerekiyor | ba\u011flant\u0131: (2.3, 1.9, -0.3) cm, normal=(0.00,1.00,0.00)"
    }
  ],
  "gate_exists": true,
  "issues": [
    "1 unresolved hot spot(s)"
  ]
}
```

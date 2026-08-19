# JoseCast Analyzer V8 – U4Venture/BİGG Teknik Metodoloji Belgesi

Bu belge, JoseCast Analyzer V8 yazılımının uyguladığı sayısal yöntemleri, fizik modellerini, denklemlerini ve doğrulama yaklaşımlarını ayrıntılı olarak açıklar. Belge, TÜBİTAK BİGG 1812 / U4Venture `110` numaralı çağrısı için hazırlanan teknik metodoloji dosyasıdır.

---

## 1. Genel Bakış ve Mimarî

### 1.1. Yazılım amacı

JoseCast, dökümhanede kullanılan, STEP formatındaki 3B geometrileri alarak;

- dolum süresi ve hız dağılımını,
- katılaşma ve soğuma geçmişini,
- soğuk birleşme (`cold shut`) ve `lap` riskini,
- hava sıkışması, çekinti, gözenek, hot-tear ve kalıp erozyonu risklerini,
- otomatik besleyici/çıkıcı önerilerini

hesaplayan, düşük donanımda (örn. NVIDIA GTX 1050 Ti) çalışabilen, açık-kaynak-tabanlı bir döküm simülasyon motorudur.

### 1.2. İş akışı

```
STEP/IGES gövde okuma
       ↓
Mesh onarımı, ölçek birimi ve vücut sınıflandırması
       ↓
3B voksel grid oluşturma (dx, mm)
       ↓
Döküm giriş sistemi boyutlandırma (sprue/runner/gate)
       ↓
Darcy/VOF/LBM bazlı dolum simülasyonu
       ↓
Enthalpi tabanlı 3B katılaşma/ısı iletimi
       ↓
SDF, modül, Niyama, hot-spot, besleyici önerisi
       ↓
Soğuk birleşme, lap, hava sıkışması, erozyon, termomekanik risk
       ↓
PyVista/VTK bazlı görselleştirme, HTML/PDF rapor, dolum animasyonu
```

Tüm modüller aynı `dx`, `dt`, `cs²` ve hız alanlarını (`velocity_m_s`) paylaşır; böylece soğuk birleşme, hava sıkışması, ısı ve çekinti modülleri tutarlı veriyle çalışır.

---

## 2. Geometri ve Vokselleştirme

### 2.1. CAD okuma ve mesh onarımı

- STEP/IGES parçalar `trimesh` kütüphanesi ile yüklenir.
- Meshler `trimesh.repair` ile doldurulur, `fix_inversion`, `fill_holes` ve `remove_duplicate_faces` uygulanır.
- Vücutlar otomatik sınıflandırılır: `PART`, `RISER`, `SPRUE`, `SPRUE_THROAT`, `RUNNER`, `INGATE`, `DISTRIBUTOR`, `CURUFLUK`, `FILTER`, `CORE`, `COOLING_SPRUE`, `POURING_BASIN`.
- Sadece `PART` etiketli vücutlar başlangıçta dökülen parça olarak kabul edilir; yetersiz gating tanımı varsa `gravity_vector` ve geometrik ipuçlarına göre sprue/runner/gate çıkarımı yapılır.

### 2.2. Birim dönüşümü

```python
scale = apply_unit_scale(bodies, unit)
```
- `unit` = "mm", "cm", "m" veya "in" olabilir.
- Mesh verteksleri mm'ye ölçeklenir.

### 2.3. Voksel grid oluşturma

```
build_voxel_grid(bodies, target_dim=BASE_RES, max_dim=600, auto_refine=True)
```

- Global bbox etrafında 4 voksel boşluklu (`margin=4`) 3B `grid[Nx,Ny,Nz]` üretilir.
- `grid[i,j,k]` vücut türü ID'sini (`BodyType` enum) taşır, `0` boş/uçurum.
- `body_index[i,j,k]` hangi `bodies` öğesine ait olduğunu belirtir.
- İstenen `target_dim` ile en uzun bbox kenarına göre voksel boyutu seçilir:

$$d_x = \frac{\max(\text{bbox}_{\text{size}})}{\text{target_dim}}$$

- **Nyquist ince cidar koruması**: En ince duvar kalınlığı `t_min` kestirilir ve `d_x \leq t_{min}/3` şartı denetlenir. Sağlanmazsa `target_dim` otomatik artırılır (`max_dim`e kadar).
- İçbükey boşluklar `binary_fill_holes` ile kapatılarak metal hacmi sıvı geçirmez kabul edilir.
- C++ voxelizer (`core.cpp_bridge.JOSECAST_CORE`) mevcutsa önce denenir, aksi durumda Python `voxelize_at_dim` kullanılır.

### 2.4. İki seviyeli SDF (Signed Distance Function)

```python
sdf = compute_sdf(is_metal, dx)
```

`is_metal` üzerinde `scipy.ndimage.distance_transform_edt` uygulanır:

$$\text{SDF}(\mathbf{x}) = d_x \cdot \text{EDT}[\text{is_metal}](\mathbf{x})$$

`SDF`, metal içindeki her vokselin en yakın metal-dışı sınırına olan mesafesidir (mm).
- `SDF = 0` sınırda,
- `SDF > 0` metal içinde,
- Duvar kalınlığı yaklaşık `t_{wall} \approx 2 \cdot \text{SDF}`.

`subvoxel_sdf` ile yüzey vokseli içinde alt-voksel düzeltilmiş mesafe elde edilir.

### 2.5. FAVOR tarzı yüzey kesit alanları

```python
f_Ax, f_Ay, f_Az = compute_face_fractions(is_metal, sub=4)
```

Her yüzey `sub×sub×sub` alt-voksel ile örneklenir; yüzeyden geçen alt hücrelerin oranı `f_A` olarak belirlenir. Bu sayede kavisli yüzeylerde akış alanı basamaklı voksel ayrımının gerçek kesit alanına yaklaşır.

---

## 3. Malzeme Veritabanı

### 3.1. Alaşım (`Alloy`) ve kalıp (`MoldMaterial`) veri yapısı

`core/types.py` ve `core/materials.py` içinde her alaşım için şunlar saklanır:

- Yoğunluk `\rho` (`kg/m³`)
- Özgül ısı `c_p` (`J/kg·K`)
- Termal iletkenlik `k` (`W/m·K`)
- Gizli ısı `L` (`J/kg`)
- Likidüs `T_L`, solidüs `T_S` veya ötektik `T_e` (`°C`)
- Döküm sıcaklığı `T_{pour}` (`°C`)
- Viskozite `\mu` (`Pa·s`)
- Yüzey gerilimi `\sigma` (`N/m`)
- Küçülme faktörü, ağaç aralığı, mekanik özellikler
- Niyama eşikleri ve Carlson-Beckermann eğrisi anahtarı

`MoldMaterial` için ise:

- `k_w_mk`, `rho_kg_m3`, `cp_j_kgk`
- `t0_c` (ilk sıcaklık)
- `chvorinov_c` (varsayılan kalıp sabiti)
- AFS tane boyutu, nem/binder oranları, geçirgenlik ve rijitlik faktörleri

Sistemde 32'den fazla alaşım ve 14 kalıp malzemesi ön tanımlıdır.

### 3.2. Chvorinov kalıp sabiti `C`

`chvorinov_c_from_properties(alloy, mold)` aşağıdaki kapalı-formülle hesaplar:

$$T_m = \frac{T_L + T_S}{2}$$

$$\Delta T = T_m - T_0$$

$$L_{eff} = L + c_p \cdot \max(T_{pour} - T_L, 0)$$

$$C_{SI} = \left( \frac{\rho_m L_{eff}}{\Delta T} \right)^2 \cdot \frac{\pi}{4 k_s \rho_s c_{p,s}}$$

$$C_{dk/cm^2} = \frac{C_{SI}}{60 \cdot 10^4}$$

Burada indeks `m` metal, `s` kalıp malzemesini gösterir. Sonuç `dk/cm²` (dakika/cm²) cinsindendir.

---

## 4. Döküm Giriş Sistemi Hesabı

### 4.1. Campbell dolum süresi

`core/gating_calculator.py` içinde:

```
t_base = log_interp(W_part)            # parça ağırlığına göre taban süre [sn]
f_rho  = (7000 / rho)^0.35             # yoğunluk düzeltmesi
f_thick = (t_mean / 20)^0.2            # ortalama cidar kalınlığı düzeltmesi
f_temp = (superheat / 100)^0.4         # aşırı ısıtma düzeltmesi
t_fill = t_base * f_rho * f_thick * f_temp
```

### 4.2. Etkin yükseklik

```
H_eff = H * (1 - head_reduction_fraction(W_part))
```

`head_reduction_fraction`, parça ağırlığına göre döküm hunisinden/girişten kaynaklanan basınç kaybını `0` ile `0.75` arası lineer interpolasyonla verir.

### 4.3. Bernoulli / süreklilik temelli alan hesabı

`compute_gating(W_kg, rho, H_m, t_fill_s, Cd, gating_ratio, n_ingates, V_crit)`:

$$V_c = \sqrt{2 g H_{eff}}$$

$$V_{c,eff} = \max(C_d V_c,\; 1.4\ \text{m/s})$$

$$V_{cast} = \frac{W}{\rho}$$

$$Q = \frac{V_{cast}}{t_{fill}}$$

$$A_{choke} = \frac{Q}{V_{c,eff}}$$

- Verilen oran `sprue : runner : gate` (örn. `1:2:2`) hangi eleman `choke` ise ondan başlanarak ters çözülür.
- Gate hızı `V_g = Q/A_g` Campbell kuralına göre `V_{crit}`i geçemez; geçerse `A_g` büyütülür ve tüm sistem aynı oranda ölçeklenir.
- Sprue boğaz hızı `1.4 m/s`nin altına düşmez.
- Sistem basınçlı/basıncsız sınıflandırması:

$$P_f = \frac{A_g}{A_s}$$

- `Pf < 0.9` → basınçlı (`pressurized`)
- `Pf > 1.3` → basınçsız (`unpressurized`)
- arada → yarı basınçlı (`semi-pressurized`)

### 4.4. Kesit parametreleri

```
d_sprue = sqrt(4 As / pi)              # mm cinsinden çap
d_ingate = sqrt(4 Ag/(pi n_ingates))   # her gate için çap
```

### 4.5. Akıllı sistem seçimi

`smart_Pf_selector`, ince cidarlı bölüm oranı, akış yolu uzunluğu ve ortalama kalınlığa bakarak `Pf` önerir:

```
flow_ratio = flow_path_mm / t_mean_mm
thin  = thin_vol_ratio  (>0.30 ve flow_ratio>60  → Pf=0.7/1.1)
long  = flow_ratio>25   → Pf=1.5/1.8
thick = thick_vol_ratio>0.4 → Pf=1.5
short = flow_ratio<20   → Pf=1.2
else                    → Pf=1.0/1.3
```

---

## 5. Dolum Simülasyonu (Darcy / VOF / LBM)

### 5.1. Darcy/Hele-Shaw basınç çözücüsü

`solve_filling_flow` fonksiyonu, hızlı ve bellek verimli bir 3B Darcy akışı çözer.

#### 5.1.1. Basınç denklemi

Sabit-viskoziteli, izotropik geçirgenlik varsayımıyla:

$$\nabla \cdot \left( \frac{K}{\mu} \nabla p \right) = 0$$

- `cavity_mask` üzerinde `p=1` (giriş/inlet) ve `p=0` (hava çıkış/vent) Dirichlet sınır koşulları uygulanır.
- `K` voksel boyutuna bağlıdır: `K = (dx/2)²`.
- Yüzey kesri `f_A` ile boru hattı direnci kavisli yüzeylerde düzeltilir:

$$\text{conductance} = \frac{K_{face} f_A}{\mu dx^2}$$

`K_face` komşu iki hücrenin harmonik ortalamasıdır.

#### 5.1.2. Basınç çözüm yöntemi

- 7-noktalı sonlu farklar matrisi `scipy.sparse` ile oluşturulur.
- Önce `scipy.sparse.linalg.spsolve` veya C++ hızlandırıcı (`JOSECAST_CORE.solve_pressure`) dener.
- Gerekiyorsa `scipy.sparse.linalg.cg` (eşlenik gradyan) Jacobi önşartlandırıcısı ile çözülür.

#### 5.1.3. Yüzey hızları

Yüzey merkezindeki hızlar merkez farklarıyla hesaplanır:

$$v_x = -\frac{K}{\mu} \frac{p_{i+1,j,k} - p_{i,j,k}}{dx}$$

Benzer şekilde `v_y`, `v_z`.

#### 5.1.4. Hacimsel akı `Q` ölçeklemesi

Ham Darcy alanı belirsiz bir basınç düşüşüne karşılık `Q_raw` üretir. Kullanıcı tarafından verilen `Q_user` (gating tasarımından veya `v×A`) ile ölçeklenir:

$$\text{scale} = \frac{Q_{user}}{Q_{raw}}$$

$$p_{real} = \text{scale} \cdot p$$

$$v_{real} = \text{scale} \cdot v_{raw}$$

Bu, Darcy alanına hem yön ve bölünme bilgisini hem de fiziksel debiyi aynı anda katar.

### 5.2. C++ LBM / VOF hızlandırması

`JOSECAST_CORE` varsa:
- `solve_lbm_filling_callback` çağrılır; bu, D3Q19 LBM ile hız `v`, gaz hacim oranı `F=1-\phi_{metal}` ve türbülans viskozitesi `\nu_t` verir.
- Her LBM adımında `AirEntrapmentSolver_D3Q7` çağrılarak hava taşınması ve sıkışması eş zamanlı çözülür.
- Paylaşılan `dx`, `dt` ve `cs²` hem D3Q19 metal hem D3Q7 hava çözücüleri için ortaktır.

### 5.3. Dolum zamanı (`fill_time_s`)

Her voksel için metalin ulaşma zamanı, basınç/hız alanı üzerinden front propagation ile çıkarılır:

- Hız alanı boyunca Dijkstra veya ENO front-tracking ile `T(\mathbf{x})` hesaplanır.
- Veya LBM çözücüsü `F` alanından direkt `fill_time` üretir.
- `fill_time` bütün ısı ve soğuk birleşme modüllerine beslenir.

### 5.4. Yüzey hızı (`v_front`)

```python
v_front = peclet_front_velocity_m_s(fill_time_s, dx_mm)
```

$$v_{front} = \frac{dx/1000}{|\nabla T_{fill}| + 10^{-12}} \quad [m/s]$$

`T_fill` dolum zamanı alanıdır; gradyanı büyük olan yerlerde cephe yavaşlar, küçük gradyanlı yerlerde hızlıdır.

---

## 6. Isı Transferi ve Katılaşma

### 6.1. Yöneten denklem

`solve_3d_thermal` entalpi formunda:

$$\frac{\partial H}{\partial t} = \nabla \cdot (k \nabla T) - \rho c_p (\mathbf{v} \cdot \nabla T)$$

- Metal bölgesinde `H = ρ c_p T + ρ L (1 - f_s)`
- Kalıp bölgesinde `H = ρ c_p T`
- `f_s`: katı fraksiyonu, `0 ≤ f_s ≤ 1`

### 6.2. Görünür ısı kapasitesi (latent heat regularisation)

Mushy bölgede gizli ısı, sıcaklık türevine göre dağıtılır:

$$c_{p,eff} = c_p + L \frac{df_s}{dT}$$

Bu, faz değişimini ek bir ısı kapasitesi olarak içeren kararlı, dolaylı (implicit) çözüm sağlar.

### 6.3. Scheil katı fraksiyonu

```python
_scheil_fs(T, T_L, T_S, k_partition)
```

$$f_s(T) = 1 - \left( \frac{T - T_S}{T_L - T_S} \right)^{\frac{1}{1-k}}$$

`k` dağılım katsayısıdır. Ötektik içeren alaşımlarda `T_S` yerine `T_e` kullanılarak ötektik platosu korunur.

### 6.4. Zaman integrasyonu

- Matris `A` sabit ıletkenlikten oluşturulur.
- Dirichlet dış sınır `T = T_0` tutulur; iç matris simetrik pozitif tanımlı kalır.
- İç bilinmeyenler için eşlenik gradyan (CG) çözülür.
- Hız taşınımı (advection) ayrıklaştırılmış (operator splitting) ve CFL `≤ 0.3` ile alt-adımlarla açıkça yapılır:

$$T^* = T^n - \Delta t_{sub} (\mathbf{v} \cdot \nabla T)$$

- Doldurma bitiminden sonra küçük bir `feed_velocity_m_s` (yerçekimi doğrultusunda) eklenir; böylece besleyiciler katılaşırken parçaya sıcak metal çekmeye/iletmeye devam eder.

### 6.5. Çıktılar

`thermal_solver` şunları döndürür:

- `T_final`: son sıcaklık alanı
- `fs_final`: son katı fraksiyonu
- `t_liquidus`: her vokselin likidüs sıcaklığına ulaşma zamanı
- `t_solidus`: her vokselin solidus sıcaklığına ulaşma zamanı
- `G_at_ts`: solidus anındaki sıcaklık gradyanı
- `R_at_ts`: solidus anındaki soğuma hızı
- `niyama`: `G/√R` alanı

---

## 7. SDF, Modül ve Yerel Geometri

### 7.1. Yerel duvar kalınlığı

$$t_{wall}(\mathbf{x}) \approx 2 \cdot \text{SDF}(\mathbf{x})$$

`SDF` ham voksel mesafesi (mm) verir; `subvoxel_sdf` ile yüzeyde daha ince düzeltme yapılır.

### 7.2. Eğrilik (curvature)

`compute_curvature(sdf, dx)` SDF Hessianından hesaplar:

$$\nabla \text{SDF} = (g_z, g_y, g_x)$$

$$H = \begin{bmatrix}
h_{zz} & h_{zy} & h_{zx} \\
h_{yz} & h_{yy} & h_{yx} \\
h_{xz} & h_{xy} & h_{xx}
\end{bmatrix}$$

- **Ortalama eğrilik** (Laplasyan yaklaşımı):

$$H_m = \nabla^2 \text{SDF} = h_{xx} + h_{yy} + h_{zz}$$

- **Gaussian eğrilik** (Hessian determinantı):

$$K = \det(H)$$

### 7.3. Steiner şekil düzeltilmiş modül

`compute_steiner_modulus(sdf, mean_curv, gauss_curv, clip_min=0.5)`:

$$SF = 1 - H_m \cdot \text{SDF} + K \cdot \text{SDF}^2$$

$$SF = \max(SF, 0.5)$$

$$M_{mod} = \frac{\text{SDF}}{SF}$$

`clip_min = 0.5` seçimi, daha önce `0.1` ile oluşan ve `M`nin `10 × SDF`ye sıçradığı aşırı büyüme hatasını engeller. `M_mod` ve `SDF` aynı birimdedir (mm). Hot-spot etiketlerinde ve duvar kalınlığında ham `SDF` kullanılır.

### 7.4. Geometrik modül `M_geo = V/A`

```python
geometric_m_mm = part_volume_mm3 / part_surface_area_mm2
```

Bu klasik Chvorinov modülüdür; `Baskın M` yanında `Geometrik M (V/A)` olarak raporlanır.

### 7.5. Baskın (dominant) modül

Parçanın tüm voksel modüllerinin medyanı (aykırı değerlere karşı dayanıklı):

$$M_{dominant} = \text{median}(M_{mod}[\text{part_mask}])$$

### 7.6. Yerel katılaşma zamanı

`compute_chvorinov_t(M_mod, C)`:

$$t_s = C \cdot \left( \frac{M_{mod}}{10} \right)^2 \cdot 60 \quad [s]$$

Burada `C` `dk/cm²` ve `M` mm cinsindendir.

---

## 8. Niyama Kriteri ve Çekinti Riski

### 8.1. Klasik Niyama

`compute_niyama`:

$$N = \frac{G}{\sqrt{R}} \quad [K \cdot s^{0.5} / mm]$$

- `G`: metal tarafı sıcaklık gradyanı (`K/mm`)
- `R`: soğuma hızı (`K/s`)

`G`, Stefan hızı `v_solid = M_mod / t_s` ile ilişkilendirilir:

$$G = \frac{\rho_m L_{eff} v_{solid}}{k_m \cdot 10^6}$$

`L_eff = L + c_p · max(T_pour - T_L, 0)`.

`R` için iki kaynak:
1. Sağlanan 3B `cooling_rate` alanı,
2. Chvorinov tabanlı:

$$R = \frac{\Delta T_{eff}}{t_s}, \quad \Delta T_{eff} = (T_L - T_S) + \frac{L}{c_p}$$

### 8.2. Niyama risk skorları

`compute_niyama_variants`:

$$\text{macro_risk} = \max\left(0,\; 1 - \frac{N}{N_{macro}}\right)$$

$$\text{shrinkage_risk} = \max\left(0,\; 1 - \frac{N}{N_{shrinkage}}\right)$$

`N_macro` ve `N_shrinkage` alaşıma özgü eşiklerdir.

### 8.3. Şekil düzeltmesi

Niyama, küresel/büyük hacimli bölgelerde daha düşük çıksın diye:

$$N_{corrected} = N \cdot \frac{M_{mod}}{\max(\text{SDF}, 10^{-6})}$$

Böylece tabaka gibi ince bölgeler `f≈1` kalırken, kalın küremsi bölgeler `f<1` risk artışı yaşar.

### 8.4. Yönlü besleme etkinliği

`directional_feed_efficiency`: Her voksel için en yakın besleyiciye olan Dijkstra mesafesi, besleyici yönü ve yerçekimi dikkate alınarak `[0.05, 1.0]` aralığında bir faktör verir.

- Yerçekimine karşı yukarı adımlar `10×` cezalanır.
- Aşağı adımlar `0.7` ile bonuslanır.
- Besleyiciye ulaşma zamanı vokselin katılaşma başlangıcından geç ise etkinlik düşer.

### 8.5. Gözenek boyutu – Carlson-Beckermann

`compute_pore_size`:

1. Niyama alanı, alaşım `niyama_star_scale` ile boyutlandırılır:

$$N^* = N \cdot niyama\_star\_scale$$

2. Carlson-Beckermann eğrisi (`_carlson_gp_pct`) ile gözenek hacim yüzdesi `g_p` tahmin edilir.
3. `g_p` besleme etkinliği ve Darcy basınç düşümü faktörü ile azaltılır/artırılır:

$$g_p = g_p \cdot feed\_factor \cdot darcy\_factor$$

4. Gözenek çapı:

$$d_{shrink} = g_p \cdot pore\_size\_um\_per\_porosity\_pct \cdot pore\_size\_length\_factor$$

- Üst sınır `max(2·M_mod, SDAS)` ile sınırlanır.
- Gaz/bifilm kaynaklı gözenek:

$$v_{over} = \max\left(0, \frac{v - v_{crit}}{v_{crit}}\right)$$

$$d_{gas} = d_{baseline} \cdot (1 + K_{entrain} \cdot v_{over}^{exp})$$

### 8.6. Döküm demiri grafit genleşmesi

```python
_graphite_cumulative(fs)
```

Ötektik çadırı şeklinde `fs`ye bağlı grafit genleşme fraksiyonu hesaplanır:

$$\text{expansion}(fs) = \text{graphite\_expansion\_fraction} \cdot \text{inoculation\_factor} \cdot G(fs)$$

- `G(fs)` `0`dan `1`e gider; ötektik bandın merkezi `fs_center = 0.7 - 0.12(CE - 4.3)` civarındadır.
- Net küçülme:

$$\text{net\_shrink} = \text{shrinkage\_factor} - \text{expansion} \cdot \text{mold\_rigidity\_factor}$$

---

## 9. Hot Spot ve Otomatik Besleyici Tasarımı

### 9.1. Hot-spot tespiti

`find_hotspots`:

1. Her voksel için `t_s = compute_chvorinov_t(M_mod, C)` hesaplanır.
2. Besleyiciler `feeder_time_factor` ile daha geç katılaşacak şekilde güçlendirilir.
3. Katılaşma zamanı katmanları `n_time_steps` adımda ilerletilir (kuadratik zaman adımı, son katmanlarda daha sık).
4. Her adımda kalan sıvı `26-komşu bağlantı` ile etiketlenir; besleyiciye dokunmayan bileşenler `isolated` olarak işaretlenir.
5. En geç izole olan bölgeler `iso_time` haritasında pik haline gelir.
6. `watershed` ile farklı pikler ayrılır; her bölge için en kritik voksel seçilir.

Hot-spot skorlama (geç izolasyon + yüksek modül + düşük Niyama):

$$score = \text{iso\_score} + 0.5 \cdot \text{modulus\_score} - 0.5 \cdot \text{niyama\_score}$$

### 9.2. Besleme yolu ve Darcy kontrolü

`_path_darcy_and_directional`:

- Dijkstra maliyeti `dx / M` ile besleyiciye en düşük dirençli yolu bulur.
- Heuver kuralı: Besleme yönünde modül gradyanı negatif olmamalı.
- Yönlü katılaşma: `t_sol` besleyiciye doğru artmalı.
- Kozeny-Carman geçirgenliği:

$$K = \frac{d_{dend}^2}{180} \cdot \frac{f_l^3}{(1-f_l)^2} \quad [mm^2]$$

- Darcy basınç düşümü:

$$\Delta P = \sum \mu v \frac{dx}{K}$$

- Besleyici hidrostatik basınç:

$$P_{head} = \rho g h_{feeder} \cdot head\_efficiency$$

`darcy_ok = ΔP < max(P_head, 1000 Pa)`.

### 9.3. Besleyici önerisi

`propose_risers`:

1. **Caine modül kuralı**:

$$M_{required} = k_{mod} \cdot M_{hotspot}$$

(`k_mod` alaşıma özgü, tipik `1.2`)

2. **SFSA hacim/oran kuralı**:

$$V_{riser} = f(SF_{zone}, V_{zone})$$

`SF_zone` beslenecek bölgenin şekil faktörü, `V_zone` hacmidir.

3. İki sonuçtan büyük olan çap `D` seçilir, `H = 1.5 D` (varsayılan) silindirik besleyici oluşturulur.
4. Uygunsuz büyüklükteki besleyiciler yerine:
   - **Mini exotermik besleyici** (`exo_mod_factor` ile daha küçük modül gereksinimi),
   - **Soğutucu / çıkıcı (chill)** önerilir.

---

## 10. Soğuk Birleşme (Cold Shut) ve Lap Modeli V8

### 10.1. Temel fizik varsayımı

Soğuk birleşme, iki ayrı metal cephesinin birbirine ulaştığında sıcaklık çok düşmüş veya katılaşma çok ilerlemişse kaynama yapamamasıdır. `core/sdf_analyzer.compute_cold_shot_risk` bu riski beş ana faktörün çarpımı olarak kurar:

$$\text{risk\_cs} = f_{thermal} \cdot f_{flow} \cdot f_{thin} \cdot f_{mold} \cdot f_{feeder}$$

### 10.2. Sıcaklık faktörü

- `T_meet`: cephe buluşma anındaki metal sıcaklığı.
- Süper ısıtma:

$$T_{super} = T_{pour} - T_L$$

$$safe\_super = \max(T_{super}/3,\; 10)$$

- Sıcaklık faktörü:

$$f_T = \exp\left(-\frac{\max(T_{meet}-T_L, 0)}{safe\_super}\right)$$

- Yerel katılaşma süresi oranı:

$$f_{solid} = \exp\left(-\frac{t_{liq,local}/t_{s,local}}{1 + (v/v_{crit,cold})^2}\right)$$

- `v_crit,cold = max(0.10, 0.45 · v_entrain)`.

### 10.3. Akış faktörü (durgunluk + çok yönlü buluşma)

`fill_time` alanının gradyanı:

$$|\nabla T_{fill}| = \sqrt{(\partial_x T)^2 + (\partial_y T)^2 + (\partial_z T)^2}$$

Cephe hızı:

$$v_{front} = \frac{dx/1000}{|\nabla T_{fill}|}$$

Ridge faktörü (Feng & Liao 2021):

$$f_{ridge} = 1 - \exp\left(-\frac{|\nabla T_{fill}|}{\text{mean}(|\nabla T_{fill}|)}\right)$$

Durgunluk faktörü:

$$f_{stagnation} = \exp\left(-\left(\frac{\max(v, v_{front})}{v_{crit,cold}}\right)^4\right)$$

Çok yönlü buluşma (confluence): komşu hücreler arası hız vektörleri nokta çarpımı hesaplanır; `dot < -0.5` ise karşı yönlü akış tespit edilir.

$$f_{confluence} = 2 \cdot \max(0, -dot - 0.5)$$

Son akış faktörü:

$$f_{flow} = f_{stagnation} \cdot (1 + f_{confluence}) \cdot f_{ridge}$$

### 10.4. İnce kesit faktörü

$$f_{thin} = \text{clip}\left( \frac{M_{10}}{M_{mod}},\; 0,\; 1 \right)$$

`M_10`, parçadaki modülün %10'luk dilim değeridir; ince bölgelerde `f_thin` büyük olur.

### 10.5. Kalıp soğutma ve alaşım katsayısı

Kalıp `effusivity`si (ısı biriktirme kabiliyeti):

$$e = \sqrt{k \rho c_p}$$

$$f_{mold} = 1 + 0.25 \cdot \ln\left( \frac{e}{e_{ref}} \right), \quad e_{ref}=\sqrt{0.58 \cdot 1600 \cdot 1170}$$

`0.7` ile `2.0` arası kırpılır. Soğuk birleşme kazancı (`cold_shot_gain`) ile çarpılır.

### 10.6. SFER: Spherical Front Encounter Rate

`core/confluence_sfer.py` ve `core/sphere_lut.py` ile her saddle noktasında 64 yönlü Fibonacci küre örneklemesi yapılır.

#### 10.6.1. Saddle tespiti

1. `find_saddles_gate_watershed`: gate-gövdelerinden tohumlanmış `watershed` ile farklı cepheleri birleştiren eyerleri bulur.
2. `find_saddles_sublevel`: `fill_time` alanı üzerinde 0-D sublevel-set persistence (merge tree) uygulanır.
3. `find_local_maxima`: dolum sonu kapanış noktaları.
4. `find_saddles_skeleton`: medial axis/skeleton ridge'leri.

Persistence eşiği `0.1` (Al/Mg) veya `0.3` (çelik/demir) sn aralığındadır; küçük gürültü eyerleri elenir.

#### 10.6.2. 64 yönlü küre örnekleme

Fibonacci küresi:

$$y_i = 1 - \frac{2i}{n-1}$$

$$r_i = \sqrt{1 - y_i^2}$$

$$\theta_i = \pi(3-\sqrt{5}) i$$

$$(x_i, y_i, z_i) = (r_i \cos\theta_i,\; y_i,\; r_i \sin\theta_i)$$

Her yönde `R = max(3dx, min(M_mod, 0.8·SDF))` yarıçapında `fill_time`, entalpi `H`, cephe hızı `v_front` ve hız vektörü örneklenir.

#### 10.6.3. Eyerdeki baskın cephe çifti

En düşük `fill_time`li ve birbirine en zıt (büyük açılı) iki yön seçilir:

$$\text{score} = \frac{T_{fill}[s_1] + T_{fill}[s_2]}{1 - \cos\theta_{12}}$$

Minimum skorlu çift baskın buluşmayı temsil eder.

### 10.7. Péclet düzeltilmiş etkin modül

`core/peclet.py`:

$$Pe = \frac{v_{front} \cdot M_{mod} \cdot 10^{-3}}{\alpha}$$

$$f_{Pe} = 1 + \min\left(c_{Pe} \sqrt{Pe},\; F_{max} - 1\right)$$

$$M_{eff} = M_{mod} \cdot f_{Pe} \cdot f_{feeder}$$

`F_max` tipik `2.0` ile sınırlıdır; bu, 88 sn → 1650 sn gibi eski aşırı büyümeleri önler.

### 10.8. Film drenajı ve Weber

`core/film_drainage.py`:

Dinamik basınç:

$$P_d = \frac{1}{2} \rho v_{rel}^2$$

Kapiler basınç:

$$P_c = \frac{2\sigma}{R}$$

Etkin basınç:

$$\Delta P = \max(0, P_d - P_c)$$

Film kalınlığı:

$$\frac{1}{h^2} = \frac{1}{R^2} + \frac{4 \Delta P \, t_{film}}{3 \mu R^2}$$

Eğer `ΔP ≤ 0` menisküs yırtılamaz ve `h = R` alınır.

Film drenaj faktörü:

$$H_h = \frac{1}{1 + \exp(-10 (h/h_0 - 1))}$$

`h_0 = 100 nm`.

Weber sayısı:

$$We = \frac{\rho v_{rel}^2 R}{\sigma}$$

$$risk_{We} = \frac{1}{1 + \exp(5(We - We_{crit}))}$$

### 10.9. Cold shut ve lap ayrımı

Soğuk birleşme (`cold shut`) için `θ ≥ 120°`:

$$H_{dt} = \frac{1}{1 + \exp(-10(\Delta t - \Delta t_{crit}))}$$

$$H_T = \frac{1}{1 + \exp(-(T_L - T_{int} - \Delta T_{crit}))}$$

$$H_{fs} = \frac{1}{1 + \exp(-20(f_s - f_{s,crit}))}$$

$$risk_{cs} = H_{dt} H_T H_{fs} H_h risk_{We} f_{geom} f_{feeder}$$

Burada `f_geom = (1 - cosθ)/2`.

Lap (`45° ≤ θ < 120°`) için benzer formül ama `dt_crit_lap` ve `fs_crit_lap` ile.

### 10.10. Confluence çizgisi görselleştirme

`core/confluence_lines.py`:

- `fill_time` alanına Gauss filtresi (`σ=1`) uygulanır.
- `scipy.ndimage.gaussian_laplace` ile negatif Laplacian ridge'leri bulunur.
- `skimage.morphology.skeletonize` ile 1 voksel kalınlığına inceltilir.
- 26-komşu graf oluşturulur, `scipy.sparse.csgraph.shortest_path` ile sıralanır.
- `splprep/splev` B-spline ile düzleştirilir.
- Risk eşiği `0.1`, minimum çizgi uzunluğu `max(3 mm, 1.5 dx)`, maksimum `8` çizgi.
- Her çizgi kendi lokal `risk_cs` değeriyle `inferno` renk skalasında tube olarak çizilir.

`ui/viewer.py`de parça `lightgrey` ve `opacity=0.08` ile saydam tutulur; risk bulutu yoksa "Risk düşük, isosurface yok" mesajı ve renk skalası gösterilir.

---

## 11. Hava Sıkışması (Air Entrapment)

### 11.1. Geometrik Front-Collision (GeoFC) modeli

`compute_air_entrapment_geofc` (fallback):

- `T_metal(x)`: metal cephesinin her hücreye ulaşma zamanı.
- `T_escape(x)`: havanın en yakın açık çıkışa (vent/riser top/sprue top) ulaşabileceği son zaman.
- Hava sıkışması:

$$T_{metal} > T_{escape} + tol$$

- Her kapalı cepte `t_seal` (boğaz kapanma zamanı) ve `V_pocket` hesaplanır.
- Havanın kaçabileği maksimum debi:

$$Q_{max} = C_d \, A_{eff} \sqrt{\frac{2 \Delta P}{\rho_{air}}}$$

- Risk:

$$\text{risk} = 1 - \min\left(1, \frac{Q_{max} \, t_{window}}{V_{pocket}}\right)$$

- Risk yalnızca cephenin **tavanına** (ceiling) yazılır; hava yukarı yükseldiği için tehlike tavandadır.
- Yeşil kum kalıplarında buhar hacmi `V_steam` ilave edilir.

### 11.2. D3Q7 LBM hava çözücüsü

`core/air_entrapment_d3q7.py`:

- D3Q7 lattice: 7 yön (`[0,0,0]` ve 6 eksen).
- Ağırlıklar: `w0 = 1/4`, diğerleri `1/8`.
- `cs²_lb = 1/4`.

D3Q7 dağılım denklemi:

$$g_q(\mathbf{x} + \mathbf{e}_q \Delta t, t+\Delta t) = g_q(\mathbf{x},t) - \frac{g_q - g_q^{eq}}{\tau} + \Delta t \, w_q (\dot{m}_{source} - \dot{m}_{escape})$$

- `g_q^{eq} = w_q \alpha\rho (1 + \mathbf{e}_q \cdot \mathbf{u}/cs²)`
- `τ = 0.5 + 4 D_t Δt / dx²`, `D_t = ν_t / Sc` (`Sc=0.7`).

#### 11.2.1. Hava kaynağı

Serbest yüzey işareti `F = 1 - φ_{metal}` üzerinde:

- `div(v)` merkezi farklarla.
- Ortalama eğrilik `κ = ∇ · (∇F / |∇F|)`.
- Sıkışma koşulu:

$$(\nabla F \cdot dx < -0.5) \; \wedge \; (\kappa > K_{crit}) \; \wedge \; (\nabla \cdot \mathbf{v} < -0.5)$$

ise:

$$\dot{m}_{source} = C_{entrap} |\nabla \cdot \mathbf{v}| (1-F) \rho_{atm}$$

#### 11.2.2. Kalıp duvarından kaçış (Darcy-Forchheimer)

Kapalı hücrelerde sıkışan hava, kalıp geçirgenliği `K` ve duvar kalınlığı `L_wall` üzerinden kaçar:

$$\dot{m}_{escape} = \frac{K_{app}}{\mu_{air} L_{wall} dx} \rho_g \Delta P$$

Klinkenberg düzeltmesi:

$$K_{app} = K \left(1 + \frac{b}{P_g}\right)$$

### 11.3. Kum kalıp geçirgenliği

`core/filling_solver._mold_air_escape_damping`:

- AFS tane boyutu:

$$AFS = \frac{15.5}{d_{50}(mm)}$$

- Permeabilite:

$$P = \frac{300000}{AFS^{1.5}}$$

- Kaçış azaltma faktörü:

$$damp = 1 - \exp(-P/500)$$

- Nem, binder ve sıkıştırılabilirlik düzeltmeleri:

$$damp \; *\!= 1 + 0.04 moisture + 0.03 binder + 0.005 \max(compact-45,0)$$

---

## 12. Erozyon ve Termomekanik Riskler

### 12.1. Kum kalıp erozyonu

`compute_erosion_risk`:

$$v_{thr} = v_{entrain} \cdot \max(0.3, rigidity)$$

$$r_{erosion} = \text{clip}\left( \frac{v - v_{thr}}{3v_{thr} - v_{thr}},\; 0,\; 1 \right)$$

`v_entrain` Campbell'ın kritik karıştırma hızıdır (Al için ~0.5 m/s).

### 12.2. İngate yüzey türbülansı (We/Oh)

`_compute_ingate_entrainment_risk`:

$$D = 2 \cdot \text{dist\_to\_wall}$$

$$We = \frac{\rho v^2 D}{\sigma}, \quad Oh = \frac{\mu}{\sqrt{\rho \sigma D}}$$

$$entrainment = \left(\frac{We}{We_{crit}} - 1\right) \cdot (1 - Oh/Oh_{crit}) \cdot \frac{v - v_{crit}}{v_{crit}}$$

### 12.3. Termal gerilme ve hot tear / cold crack

`compute_thermal_stress`:

$$\Delta T = \max(T_S - T, 0)$$

$$\varepsilon = \alpha \Delta T$$

$$\sigma = \min(E \varepsilon, \sigma_{yield,highT})$$

- **Hot tear**: `mushy` bölgede (`0.3 < f_s < 0.95`) `ε / tear_thr` ile risk.
- **Cold crack**: `T ≤ 100 °C` ve `f_s ≥ 0.99` bölgede `σ / σ_y,room` ile risk.

---

## 13. Görselleştirme ve Raporlama

### 13.1. 3B görselleştirme

`ui/viewer.py` PyVista kullanır:

- Parça: `lightgrey`, `opacity=0.08`.
- Soğuk birleşme: `inferno` renkli ince confluence çizgileri (`risk ∈ [0,1]`).
- Lap: `viridis` çizgiler.
- Hava sıkışması: `coolwarm` / `jet` bulut, riski tavan vokseline yayılır.
- Niyama / gözenek / erozyon: ilgili scalar bar ve izoyüzeyler.
- Boş risk durumlarında bile scalar bar ve "Risk düşük" mesajı gösterilir.

### 13.2. Rapor çıktıları

`core/reporter.py`:

- HTML rapor: malzeme, kalıp, `Baskın M`, `Geometrik M (V/A)`, `t_wall`, Chvorinov `C`, hot-spot listesi, öneriler.
- PDF özet: tek satırda baskın modül, geometrik modül, duvar kalınlığı.

---

## 14. Doğrulama ve Test

### 14.1. Birim testler

```
pytest tests/ -q  → 30 passed
```

`tests/` klasöründe:
- Vokselleştirme doğruluğu
- SDF/modül hesabı
- Chvorinov `C` el ile değerlerle karşılaştırma
- Niyama hesabı
- Soğuk birleşme riskinin fiziksel sınırlar içinde kalması
- Gating oranlarının Campbell kurallarına uyması

### 14.2. Fiziksel doğrulama örnekleri (Deneme_Ring.STEP)

| Alaşım | Kalıp | Ortalama soğuk birleşme riski | Beklenen fizik |
|---|---|---|---|
| AlSi7 | Kum | 0.214 | Düşük soğutma, orta risk |
| AlSi7 | Seramik | 0.223 | Hafif yüksek risk |
| AlSi7 | Metal kalıp | 0.330 | En hızlı soğutma, en yüksek risk |
| 42CrMo4 | Metal kalıp | ≈ 0.0 | Çelik toleranslı, simetrik halkada risk oluşmaz |

### 14.3. Hotspot modül düzeltmesi örneği

`M_mod` hesabındaki `clip_min` hatası düzeltilmeden önce:
- Hotspot `M` = 113.94 mm → ekranda `11.39 cm`
- `t_section` = 227 mm

`clip_min=0.5` ve ham `SDF` kullanımı sonrası:
- Hotspot `M` = 15.29 mm → `1.53 cm`
- `t_section` = 30.57 mm
- `Baskın M` = 0.68 cm
- `Geometrik M (V/A)` = 1.62 cm

Sonuçlar duvar kalınlığı ve klasik modül hesabıyla tutarlıdır.

---

## 15. Kullanılan Ana Teknolojiler ve Kütüphaneler

- Python 3.12
- NumPy / SciPy (seyrek matris, optimizasyon, morfoloji, interpolasyon)
- scikit-image (skeletonize, watershed, h_maxima)
- PyVista / VTK (görselleştirme)
- trimesh (CAD okuma / onarım)
- numba (JIT hızlandırma)
- C++ hızlandırıcı modülü `josecast_core` (opsiyonel, D3Q19 LBM + basınç çözücü)

---

## 16. Sonuç

JoseCast Analyzer V8, döküm simülasyonunun temel fiziklerini (Bernoulli, Darcy, Chvorinov, Niyama, Carlson-Beckermann, Kozeny-Carman, film drenajı, LBM-tabanlı hava taşınımı) tek bir voksel gridi üzerinde birleştiren, düşük maliyetli ve açık mimarili bir mühendislik yazılımıdır. Soğuk birleşme modeli, özellikle `fill_time` alanının analizi, 64 yönlü SFER örneklemesi, film drenajı ve `inferno` confluence çizgileri ile fiziksel olarak savunulabilir ve görsel olarak profesyonel bir sunuma uygundur.

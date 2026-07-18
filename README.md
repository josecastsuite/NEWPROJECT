# JoseCast Analyzer v8.0 Titan

Geometrik döküm analiz motoru. STEP dosyasındaki her body (PARÇA, BESLEYİCİ, MEME, YOLLUK, DÖKÜM AĞZI, MAÇA) ayrı ayrı voxelize edilir ve yalnızca geometri ile birlikte hafif psödo-termal katkılar kullanılarak:

- **Kullanıcı girdi parametreleri**: Döküm sıcaklığı (T_pour), liquidus/solidus, kalıp sıcaklığı, döküm süresi (t_fill), sıvı yoğunluk, viskozite. Süperheat ve sıvı akışkanlık uzunluğu hesaplanır.
- **Adaptive voxel motor**: 160 baz çözünürlükten 2040'a kadar yerel refine (hotspot çevresinde, bellek güvenliği için 384³ ile sınırlandırılmış).
- **Sub-voxel SDF**: 0.5 dolu kenar vokseli kısmi sayılabilir (2x/3x upsample + linear EDT).
- **Hessian eğrilik analizi**: SDF'nin ikinci türevinden mean/Gaussian curvature; keskin köşeler hot-spot adayı.
- **Medial axis kalınlık**: `skeletonize_3d` ile iskelet üzerinden M histogramı ve baskın duvar kalınlığı.
- **Heuver çemberleri**: Besleme yolunda SDF'nin besleyiciye doğru sürekli azalması gerekir; artarsa alarm.
- **Şekil faktörü**: `SF = V² / A³`; küreye uzak geometrilerin beslenme zorluğu ölçülür.
- **Pseudo-enthalpy Fourier ısı denklemi**: `∂H/∂t = (k/ρ)∇²T`, `H = cp·T + (1-fs)·L`; 100 explicit adım. Latent heat `cp_eff` üzerinden katılır.
- **Isı diverjansı**: `∇·(∇T) = ∇²T` hesaplanır; negatif değerler ısının toplandığını gösterir.
- **Scheil katı oranı**: `fs = 1 - ((T_l - T)/(T_l - T_s))^(1/(k-1))`; `fs < 0.6` iken besleme mümkün.
- **Katılaşma zamanı**: `t_solid = C·M² / ΔT_super`; yüksek süperheat geç donmaya neden olur.
- **Niyama ailesi**: classical, coarse, elbow, LCC — 4 varyant + ensemble; `0.775` makro, `1.5` mikro eşikleri.
- **26-komşu Dijkstra** ile metal içi en kısa besleme mesafesi ve düşük-dirençli besleme yolu maliyeti (`cost = Σ dx / M_i`).
- **Gelişmiş besleme mesafesi**: `FD = k1·t + k2·W`, yerçekimi faktörü, Heuver kontrolü ve sıvı oranı kesme kontrolü.
- **Darcy basınç kaybı**: Kozeny-Carman tarzı `f_l` bağımlı geçirgenlik, `dP` integral; `fs ≥ 0.6` → besleme kesilir.
- **Yönlü katılaşma kontrolü**: hot-spottan besleyiciye giden yolda sıcaklık düşüşü / daralma tespiti.
- **Bernoulli + dirsek kayıpları**: yolluktaki her dirsek için `h_loss = K·v²/2g`.
- **Meme akış matematiği**: ingate hızı, Reynolds, Froude ve türbülans kontrolü.
- **Direnci artırılmış modül transferi**: `M_riser = 1.2·M + direnç düzeltmesi`.
- **Histogram istatistikleri**: M'nin ortalama, std, çarpıklık ve `±dx/2` belirsizlik.
- **Alaşım / kalıp veritabanı**: AlSi7, GGG40, 42CrMo4, bronz; kum / metal / seramik kalıp.
- **Birim dedektörü**: mm / cm / m / inch otomatik ölçeklendirme.
- **BLACK AI Terminal** entegrasyonu (gelecekte LLM bağlantı noktası).
- **HTML raporu tarayıcıda görüntüleme** ve **PDF export**.

Amaç pahalı termal CFD yerine geometri + hafif PDE kullanarak 2-3 dk içinde mühendis gibi öneri sunmaktır. Otomatik düzeltme yapmaz.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Çalıştırma

```bash
python main.py
```

## Kullanım

1. **STEP Yükle**: `.step` / `.stp` dosyasını seç.
2. **Birim seç**: mm / cm / m / inch.
3. **Body tipi ata**: PARÇA / BESLEYİCİ / MEME / YOLLUK / DÖKÜM AĞZI / MAÇA.
4. **Alaşım ve kalıp seç**; parametreler otomatik dolar, istenirse elle değiştirilir.
5. **Max çözünürlük** (160-2040), **sub-voxel faktörü** (1-3), **ısı adımı** (0-2000) ayarla.
6. **Mesh Ata (Voxelize)**.
7. **Geometrik Analiz Et**.
8. **3D Görselleştirme**, **HTML Raporu Tarayıcıda Aç** veya **PDF Raporu Kaydet**.

## Proje Ruhu

Tamamen geometrik + hafif termal akıl yürütme. SDF, gradyan, eğrilik, Fourier PDE, Niyama geometrisi, Heuver çemberleri ve akış matematiği ile döküm tasarımının zayıf noktalarını tespit eder; otomatik düzeltme yapmaz, yalnızca mühendise somut öneri sunar.

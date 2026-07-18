# JoseCast Analyzer v7.2 Titan

Geometrik döküm analiz motoru. STEP dosyasındaki her body (PARÇA, BESLEYİCİ, MEME, YOLLUK, DÖKÜM AĞZI, MAÇA) ayrı ayrı voxelize edilir ve yalnızca geometri ile birlikte hafif psödo-termal katkılar kullanılarak:

- **Adaptive voxel motor**: 160 baz çözünürlükten 2040'a kadar yerel refine (hotspot etrafında, bellek güvenliği için 384³ ile sınırlandırılmış).
- **Sub-voxel SDF**: 0.5 dolu kenar vokseli kısmi sayılabilir (2x/3x upsample + linear EDT).
- **Hessian eğrilik analizi**: SDF'nin ikinci türevinden mean/Gaussian curvature; keskin köşeler hot-spot adayı olarak eklenir.
- **Medial axis kalınlık**: `skeletonize_3d` ile iskelet üzerinden M histogramı ve baskın duvar kalınlığı.
- **Şekil faktörü**: `SF = V² / A³`; küreye uzak geometrilerin beslenme zorluğu ölçülür.
- **Fourier ısı denklemi**: explicit finite-difference `∂T/∂t = α∇²T` (100 adım), metal döküm sıcaklığından başlayarak.
- **Scheil katı oranı**: `fs = 1 - ((T_l - T)/(T_l - T_s))^(1/(k-1))`; `fs < 0.6` iken besleme mümkün.
- **Niyama ailesi**: classical, coarse, elbow, LCC — 4 varyant + ensemble; `0.775` makro, `1.5` mikro eşikleri.
- **26-komşu Dijkstra** ile metal içi en kısa besleme yolu.
- **Gelişmiş besleme mesafesi**: `FD = k1·t + k2·W`, yerçekimi faktörü ve sıvı oranı kesme kontrolü.
- **Darcy basınç kaybı**: Kozeny-Carman tarzı `f_l` bağımlı geçirgenlik, `dP` integral.
- **Yönlü katılaşma kontrolü**: hot-spottan besleyiciye giden yolda sıcaklık düşüşü / daralma tespiti.
- **Bernoulli + dirsek kayıpları**: yolluktaki her dirsek için `h_loss = K·v²/2g`.
- **Direnci artırılmış modül transferi**: `M_riser = 1.2·M_casting + direnç düzeltmesi`.
- **Histogram istatistikleri**: M'nin ortalama, std, çarpıklık ve `±dx/2` belirsizlik.
- **Alaşım / kalıp veritabanı**: AlSi7, GGG40, 42CrMo4, bronz; kum / metal / seramik kalıp.
- **Birim dedektörü**: mm / cm / m / inch otomatik ölçeklendirme.
- **BLACK AI Terminal** entegrasyonu (gelecekte LLM bağlantı noktası).
- **HTML→PDF Türkçe raporu** ve 3D ekran görüntüsü embed.

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
4. **Alaşım ve kalıp seç**.
5. **Max çözünürlük** (160-2040), **sub-voxel faktörü** (1-3), **ısı adımı** (0-2000) ayarla.
6. **Mesh Ata (Voxelize)**.
7. **Geometrik Analiz Et**.
8. **3D Görselleştirme** ve **PDF Export**.

## Proje Ruhu

Tamamen geometrik + hafif termal akıl yürütme. SDF, gradyan, eğrilik, Fourier PDE ve Niyama geometrisi ile döküm tasarımının zayıf noktalarını tespit eder; otomatik düzeltme yapmaz, yalnızca mühendise somut öneri sunar.

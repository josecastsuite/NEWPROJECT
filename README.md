# JoseCast Analyzer v7.1 Titan

Geometrik döküm analiz motoru. STEP dosyasındaki her body (PARÇA, BESLEYİCİ, MEME, YOLLUK, DÖKÜM AĞZI, MAÇA) ayrı ayrı voxelize edilir ve yalnızca geometri kullanılarak:

- **Adaptive voxel motor**: 160 baz çözünürlükten 2040'a kadar yerel refine (hotspot etrafında octree-tarzı, bellek güvenliği için yerel bölge 384³ ile sınırlandırılır).
- **4 katmanlı fizik**: Chvorinov katılaşma süresi (`t_s = C·M²`), Niyama kriteri (`G / sqrt(R)` ile `0.775/1.5` eşikleri), Heuver/modulus histogram, besleme direnci integrali.
- **26-komşu Dijkstra** ile metal içi en kısa besleme yolu (`scipy.sparse.csgraph` ile hızlandırılmış).
- **DBSCAN** hot-spot kümeleme, medial-axis (skeletonize) kalınlık tespiti ve yerçekimi faktörlü besleyici verimliliği.
- **Alaşım ve kalıp veritabanı**: AlSi7, GGG40, 42CrMo4 alaşımları; kum/metal/seramik kalıplar.
- **Darcy besleme direnci** ve yönlü katılaşma kontrolü (yolda soğuk nokta/daralma tespiti).
- **Birim dedektörü**: mm / cm / m / inch otomatik ölçeklendirme.
- **BLACK AI Terminal** entegrasyonu (gelecekte LLM bağlantı noktası).
- Meme / yolluk / döküm ağız kontrolü (Campbell, Bernoulli, meme konumu, gerekli kesit alanları).
- Porozite risk haritası ve interaktif PyVista kesit düzlemleri (SDF / Risk / Niyama / Mat ID).
- HTML→PDF Türkçe raporu.

Termal çözüm (CFD, ısı denklemi, conductivity vb.) kullanılmaz. Hedef çalışma süresi: küçük modeller için ~10-30 sn, karmaşık modeller için 2-3 dk.

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
2. **Birim seç**: mm / cm / m / inch (varsayılan mm).
3. **Body tipi ata**: Sol paneldeki listeden her body için PARÇA / BESLEYİCİ / MEME / YOLLUK / DÖKÜM AĞZI / MAÇA seç.
4. **Alaşım ve kalıp seç**: Çelik / dökme demir / alüminyum / bronz ve kum / metal / seramik kalıp.
5. **Max çözünürlük**: 160 (baz) - 2040 arası ayarla. Daha yüksek = daha ağır analiz.
6. **Mesh Ata (Voxelize)**: Voxel grid oluştur.
7. **Geometrik Analiz Et**: SDF, hot spot, besleme mesafesi, riser, Niyama, Darcy, yönlü katılaşma, gating ve risk haritasını üret.
8. **3D Görselleştirme**: Hotspot, risk bulutu, yerel refine bölgeleri ve SDF/risk/Niyama/Mat ID kesit düzlemleri arasında geçiş yap.
9. **PDF Export**: Türkçe raporu ve 3D ekran görüntüsünü kaydet.

## Proje Ruhu

Tamamen geometrik akıl yürütme. Pahalı termal CFD yerine SDF, gradyan ve Niyama geometrisi kullanarak döküm tasarımının zayıf noktalarını tespit eder. Otomatik düzeltme yapmaz, yalnızca mühendise öneri sunar.

# JoseCast Analyzer v7.0 Titan

Geometrik döküm analiz motoru. STEP dosyasındaki her body (PARÇA, BESLEYİCİ, MEME, YOLLUK, DÖKÜM AĞZI, MAÇA) ayrı ayrı voxelize edilir ve yalnızca geometri kullanılarak:

- **Adaptive voxel motor**: 160 baz çözünürlükten 2040'a kadar yerel refine (hotspot etrafında octree-tarzı).
- **4 katmanlı fizik**: Chvorinov katılaşma süresi, Niyama kriteri, Heuver/modulus histogram, besleme direnci integrali.
- **26-komşu Dijkstra** ile metal içi en kısa besleme yolu.
- **DBSCAN** hot-spot kümeleme ve yerçekimi faktörlü besleyici verimliliği.
- **Malzeme seçimi**: Çelik, dökme demir, alüminyum, bronz.
- **Birim dedektörü**: mm / cm / m / inch otomatik ölçeklendirme.
- **BLACK AI Terminal** entegrasyonu (gelecekte LLM bağlantı noktası).
- Meme / yolluk / döküm ağızı kontrolü (Campbell & Bernoulli).
- Porozite risk haritası ve interaktif PyVista kesit düzlemleri.
- HTML→PDF Türkçe raporu.

Termal çözüm (CFD, ısı denklemi, conductivity vb.) kullanılmaz. Hedef çalışma süresi: küçük modeller için ~30 sn, karmaşık modeller için 2-3 dk.

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
4. **Max çözünürlük**: 160 (baz) - 2040 arası ayarla. Daha yüksek = daha ağır analiz.
5. **Mesh Ata (Voxelize)**: Voxel grid oluştur.
6. **Geometrik Analiz Et**: SDF, hot spot, besleme mesafesi, riser, Niyama, gating ve risk haritasını üret.
7. **3D Görselleştirme**: Hotspot, risk bulutu, yerel refine bölgeleri ve SDF/risk/Niyama/Mat ID kesit düzlemleri arasında geçiş yap.
8. **PDF Export**: Türkçe raporu ve 3D ekran görüntüsünü kaydet.

## Proje Ruhu

Tamamen geometrik akıl yürütme. Pahalı termal CFD yerine SDF, gradyan ve Niyama geometrisi kullanarak döküm tasarımının zayıf noktalarını tespit eder. Otomatik düzeltme yapmaz, yalnızca mühendise öneri sunar.

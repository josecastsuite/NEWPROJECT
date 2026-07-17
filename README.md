# JoseCast Analyzer v5.0

Geometrik döküm analiz motoru. STEP dosyasındaki her body (PARÇA, BESLEYİCİ, MEME, YOLLUK, DÖKÜM AĞZI, MAÇA) ayrı ayrı voxelize edilir ve yalnızca geometri kullanılarak:

- SDF (Signed Distance Field) ile sıcak nokta (hot spot) tespiti,
- 26-komşu Dijkstra ile besleme mesafesi,
- Besleyici yeterlilik kontrolü,
- Meme / yolluk / döküm ağzı kontrolü (Campbell & Bernoulli),
- Porozite risk haritası,
- Türkçe PDF raporu

üretilir. Termal çözüm (CFD, ısı denklemi vb.) kullanılmaz.

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
2. **Body tipi ata**: Sol paneldeki listeden her body için PARÇA / BESLEYİCİ / MEME / YOLLUK / DÖKÜM AĞZI / MAÇA seç.
3. **Mesh Ata (Voxelize)**: 96³ bazlı voxel grid oluştur.
4. **Geometrik Analiz Et**: Hot spot, besleme mesafesi, riser ve gating kontrollerini çalıştır.
5. **PDF Export**: Türkçe raporu ve 3D ekran görüntüsünü kaydet.

## Proje Ruhu

Tamamen geometrik akıl yürütme. Pahalı termal CFD yerine, SDF haritası sayesinde 30 saniyede döküm tasarımının zayıf noktalarını tespit eder. Otomatik düzeltme yapmaz, yalnızca mühendise öneri sunar.

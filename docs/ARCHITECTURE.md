# Yazılım Mimarisi

> Vitrin sürümü. Ana projedeki (private) mimari belgesinden kısaltılmıştır. 🔒 işaretli bileşenler projenin çekirdeğidir ve bu depoda yer almaz; görevleri burada yalnız genel hatlarıyla anlatılır.

## 1. Dizin yapısı

```text
neurovista/
├── backend/app/
│   ├── 🔒 inference_core.py    # Bilimsel protokol: ön işleme, model mimarileri, etiket kararı, ensemble, metrikler
│   ├── 🔒 pipeline.py          # Çok-modelli çalıştırma motoru: model seti, cihaz, hızlandırma yolu, konsensüs
│   ├── 🔒 settings.py          # Yollar, kilitli ayarlar, ortam bayrakları
│   ├── main.py                 # FastAPI uygulamasını kurar (yönlendiriciler, hata yanıtları, statik dosyalar)
│   ├── visualization.py        # Ortak görsel tanımlar: sınıf renkleri, kesit kontrastı, konsensüs katmanı
│   ├── api/                    # HTTP katmanı: system, analysis, viewer, export, pages, errors, access
│   ├── services/               # İş mantığı: oturum ve iş kuyruğu, yükleme doğrulama, panel/rapor, 3B mesh
│   └── static/                 # Web arayüzü: tanıtım sayfası, analiz çalışma alanı, WebGL2 sahnesi
├── 🔒 gui/                     # Masaüstü arayüzü (CustomTkinter + PyVista)
├── 🔒 training/                # Eğitim ve değerlendirme hattı (Kaggle)
├── 🔒 model_checkpoints/       # Eğitilmiş ağırlıklar (SHA-256 ile kilitli)
├── tests/                      # Bu depoda çekirdekten bağımsız testler; ana projede tam test takımı
├── docker/                     # Web imajı (CPU) ve compose dosyası
├── charts/v2/                  # Sonuç şekilleri (TR/EN)
└── docs/                       # Mimari, API, model seçimi ve değerlendirme yöntemi
```

## 2. Katmanlar

```text
🔒 inference_core   bilimsel protokol (tek kaynak): ön işleme, mimariler, etiket kararı, ensemble, metrikler
        ▲
🔒 pipeline         çalıştırma: model seti, cihaz, hızlandırma yolu, konsensüs, uyum → CaseResult
        ▲
   ┌────┴────────────────────┐
api/ + services/ (web)    gui/ (masaüstü)        ← yalnız gösterir; visualization.py ortak
```

- **Tek doğruluk kaynağı:** Aynı vaka web'de ve masaüstünde aynı modellerle, aynı ensemble ile aynı maskeyle sonuçlanır. Arayüzler hesap yapmaz; motorun ürettiği `CaseResult`'ı gösterir.
- **Eğitim–çıkarım paritesi:** Ön işleme ve etiket kararı yalnız çekirdekte tanımlıdır; eğitim hattı ve servis aynı kodu kullanır. Paritenin bozulmadığı ana projede testlerle denetlenir.
- **Model seti başlangıçta doğrulanır:** Ağırlıkların ve kilitli ensemble ayarının SHA-256 özetleri kayıtla eşleşmezse servis güncel model setini yüklemez.

## 3. Bir analizin yolculuğu

```text
NIfTI dosyaları (T1, T1ce, T2, FLAIR [+ isteğe bağlı referans maske])
  │  services/uploads   → dosya adı eşleştirme, boyut ve uzay denetimi, NaN/sonsuz reddi, ZIP güvenliği
  ▼
🔒 ön işleme            → eğitim hattıyla aynı dönüşümler
  ▼
🔒 üç model             → kayan pencereyle çıkarım (GPU: CUDA FP16 · CPU: OpenVINO FP32)
  ▼
🔒 kilitli ensemble     → ana maske; konsensüs sayımı ve modeller arası uyum skoru
  ▼
services/session        → oturum kaydı, iş kuyruğu (aynı anda tek analiz)
  ▼
api/ + static/          → kesit görüntüleri, 3B sahne, PDF/PNG rapor, NIfTI maske dışa aktarımı
```

## 4. Eğitim ve değerlendirme süreci (genel)

1. Üç model (Residual 3D U-Net, SwinUNETR, MedNeXt-S) **aynı veri akışı, aynı kayıp ve aynı bütçeyle** sıfırdan eğitildi (Kaggle, 2×T4). Her modelin en iyi checkpoint'i validation kümesinde seçilip SHA-256 ile kilitlendi.
2. Ensemble yöntemi ve çıkarım zamanı iyileştirmeleri **yalnız validation'da**, önceden yazılmış kurala göre seçildi ve kilit dosyasına mühürlendi.
3. Test kümesi **bir kez**, ön-kayıtlı planla değerlendirildi; ardından iki bağımsız kurumun verisi aynı kilitli ayarla sınandı (`docs/DEGERLENDIRME_YONTEMI.md`).
4. Modeller ONNX/OpenVINO'ya aktarıldı; CPU yolu PyTorch çıktısıyla parite denetiminden geçti.

## 5. Tasarım kuralları

- Ön işleme ve etiket kararı tek yerde tanımlıdır; giriş noktaları kendi kopyalarını yazmaz.
- Model çalıştırma, ensemble ve konsensüs yalnız motordadır; arayüzler sonucu gösterir.
- Test kümesi yalnız ön-kayıtlı, tek seferlik değerlendirmede kullanılır; hiçbir seçim test üzerinde yapılmaz.
- Kilitli checkpoint'lerin üzerine yazılmaz; yeni sürümler ayrı adlandırılır.
- Sayılar belgelere ve sayfalara elle yazılmaz; tek kaynak sonuç dosyasından üretilir ve testlerle denetlenir.
- Web katmanında iş mantığı `services/`'te, HTTP ayrıntısı `api/`'dedir; `main.py` yalnız uygulamayı kurar.
- Raporlar ve uç noktalar araştırma amaçlı olduğunu açıkça belirtir.

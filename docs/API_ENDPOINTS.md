# REST API — Yerel Web Servisi

> Vitrin sürümü. Ana projedeki (private) API belgesinden kısaltılmıştır. Bu depoda çekirdek ve model ağırlıkları bulunmadığından servis buradan çalıştırılamaz; belge, arayüz tasarımını göstermek içindir.

- **Çatı:** FastAPI (`backend/app/main.py`); sayfalar `/` (tanıtım) ve `/analysis` (analiz çalışma alanı).
- **Önbellek:** Sayfalar ve `css/`, `js/` dosyaları `Cache-Control: no-cache` ile sunulur; tarayıcı ETag ile sorar (değişmediyse 304), güncellemeden sonra eski sayfa yeni betikle karışmaz.

## Model seti

- Servis başlarken ağırlıkların ve kilitli ensemble ayarının SHA-256 özetlerini kayıtla karşılaştırır.
  - **Eşleşirse (v2):** üç model + kilitli ensemble; ana maske ensemble'dır. Ensemble ayarı yalnız kilitten okunur, web'de yeniden seçilmez.
  - **Eşleşmezse:** önceki iki modelli sete düşer; nedeni `/api/v1/health` yanıtında açıklanır.
- **Etkileşimli hız modu:** web'de test zamanı veri artırma uygulanmaz; kilitli değerlendirmedeki ayar yanıtta ayrıca gösterilir.
- **Donanım:** GPU'da CUDA FP16; CPU'da ONNX paritesi doğrulanmış modeller OpenVINO FP32 ile, diğerleri PyTorch ile.

## Uç noktalar

| Uç nokta | Metot | Parametreler | Yanıt / işlev |
|---|---|---|---|
| `/api/v1/health` | GET | — | Cihaz, model seti (+ not), modeller ve çalışma yolları, overlap, OpenVINO durumu, checkpoint SHA-256 özetleri, ortam sürümleri (Python, torch, CUDA, MONAI, numpy, OpenVINO, FastAPI), kullanım amacı. Belirteç istemez |
| `/api/v1/results` | GET | — | Kilitli sonuçlar (tek kaynak sonuç dosyası); sayfalar sayıları buradan okur |
| `/api/v1/predict` | POST | `multipart/form-data` `files`: dört NIfTI (+ isteğe bağlı `_seg`) veya bunları içeren tek ZIP | `{job_id, status}`; analiz arka planda ve sırayla (tek iş) yürür |
| `/api/v1/jobs/{job_id}` | GET | — | `status` (`queued` / `running` / `completed` / `failed`), `message`, tamamlandığında `result` |
| `/api/v1/slice` | GET | `patient_id`, `source` (`original` / `gt` / model anahtarları / `ensemble` / `activation` / `consensus`), `slice_idx`, `modality` (0–3) | Kesit PNG |
| `/api/v1/volume` | GET | `patient_id` | Tarayıcıda hacim görüntüleme için seyreltilmiş FLAIR (uint8; şekil ve aralık başlıklarda) |
| `/api/v1/scene` | GET | `patient_id`, `source` | 3B sahne katmanları (beyin, NCR, ED, ET mesh'leri) ve hacimler |
| `/api/v1/mesh` | GET | `patient_id`, `source`, `layer` (`wt` / `ncr` / `ed` / `et`) | OBJ mesh |
| `/api/v1/export-report` | GET | `patient_id`, `slice_idx`, `modality`, `format` (`png` / `pdf`) | Orijinal, referans, her model, ensemble ve konsensüs panelleri; alt bilgide vaka, model seti, dosya özetleri, çalışma yolları ve araştırma uyarısı |
| `/api/v1/export-mask` | GET | `patient_id`, `source`, `labels` (`brats` = ET 4, varsayılan / `internal` = ET 3) | Orijinal MR geometrisinde `.nii.gz` maske (3D Slicer / ITK-SNAP) |
| Statik | GET | `/css/*`, `/js/*` | Sayfa stilleri ve betikleri |

## `result` alanı (iş tamamlandığında)

- `model_set`, `model_order` (ör. `["unet", "swin", "mednext"]`), `primary_source` (`ensemble`).
- `models.<anahtar>`: `name`, `dice` (referans maske varsa WT/TC/ET; yoksa `null`), `volumes_cm3` (WT, TC, NCR, ED, ET), `inference_time_sec`, `runtime` (`cuda` / `cpu_fallback` / `openvino` / `pytorch`), `removed_outside_brain_voxels`.
- `ensemble`: `dice`, `volumes_cm3`, yöntem adı ve kilitli ayarın özeti, `inference_time_sec`.
- `agreement`:
  - `score`: modeller arası ortalama ikili Dice (WT/TC/ET); referans maske kullanmaz.
  - `consensus_voxels`: `all_models`, `majority_only`, `minority` tümör voksel sayıları.
  - Kilitli testte ve harici kohortlarda başarısız vakaları ayırt ettiği gösterildi; tek vakada kesin hüküm değildir.
- `total_time_sec`, `device`, `overlap`, `orientation`, `protocol_warnings` (ör. 1 mm dışı voksel aralığı).

## Girdi kuralları ve güvenlik

- **Dosya eşleştirme:** modaliteler dosya adındaki son ekle eşleşir.
  - BraTS 2021: `_t1`, `_t1ce`, `_t2`, `_flair`, `_seg`.
  - BraTS 2023: `-t1n`, `-t1c`, `-t2w`, `-t2f`, `-seg`.
  - Farklı hasta öneklerine sahip dosyalar birbirleriyle eşleştirilmez.
- **Boyut sınırları:** istek başına en fazla 1 GB; ZIP açılmış toplam en fazla 2 GB ve en fazla 64 NIfTI üyesi (ZIP'ten yalnız NIfTI çıkarılır); hacim başına en fazla 512³ voksel.
- **Geçici dosyalar:** yüklenen dosyalar analiz bitince (başarılı ya da başarısız) silinir; servis başlarken önceki geçici klasörler temizlenir.
- **Yalnız yüklenen dosyalar:** eksik modalite hiçbir yerden tamamlanmaz. Başka hastanın, adı anlaşılamayan ya da yinelenen dosyası varsa seçim, sorunlu dosya adlandırılarak reddedilir (422). ZIP'teki macOS üst verisi (`__MACOSX/`, `._*`) yok sayılır.
- **Vaka doğrulaması:** dört modalite (ve varsa referans) aynı boyut ve aynı uzayda (affine, 0.001 mm tolerans) olmalı; NaN/sonsuz yoğunluk reddedilir. Hata iletisi hangi dosyanın uymadığını söyler (422).
- **Hata iletileri:** beklenmeyen hatalarda istemciye iç ayrıntı (yol, yığın, CUDA iletisi) gönderilmez; ayrıntı sunucu günlüğündedir, kullanıcı iş kimliğini görür.
- **Erişim belirteci (isteğe bağlı):** `NEUROVISTA_ACCESS_TOKEN` ortam değişkeni ayarlıysa `/api/v1/health` dışındaki her istek belirteç ister. Tarayıcıda bir kez `/?token=<belirteç>` ile girilir (HttpOnly, SameSite=Strict çerez; adres çubuğundan silinir); programlar `Authorization: Bearer <belirteç>` gönderir. Karşılaştırma sabit zamanlıdır; eksik/yanlış belirteç 401.
- **Kaynak koruması:** aynı anda tek analiz çalışır, diğerleri sırada bekler; iş geçmişi son 50 kayıtla, bellekteki hasta oturumları 2 ile sınırlıdır.

## Kullanım amacı

Araştırma amaçlı karar destek prototipidir; klinik tanı veya tedavi kararı için tek başına kullanılamaz.

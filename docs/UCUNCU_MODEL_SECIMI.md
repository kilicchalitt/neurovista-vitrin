# Üçüncü Model Seçimi — Araştırma ve Karar

> Vitrin sürümü. Ana projedeki (private) karar belgesinden kısaltılmıştır; eğitim reçetesi ayrıntıları bu depoda yer almaz.
> **Karar (12 Ekim 2026):** MedNeXt-S, kernel 3.

## 1. Soru

İki modelin (Residual 3D U-Net = CNN, SwinUNETR = Transformer) yanına üçüncü bir model eklenecekti. Amaç:
- üçlü ensemble ile tekil modellerden daha doğru ve daha güvenilir sonuç almak,
- modeller arası uyuşmazlığı belirsizlik ve hata göstergesi olarak kullanmak.

## 2. Literatürden çıkan üç bulgu

1. **BraTS 2021'de mimariler arası fark çok küçük; skorlar doymuş.** "nnU-Net Revisited" (Isensee vd., MICCAI 2024) aynı titiz protokolle (5 katlı CV, 1251 vaka):

   | Model | Dice (%) |
   |---|---|
   | MedNeXt-L k5 | **91.50** |
   | MedNeXt-L k3 | 91.35 |
   | nnU-Net ResEnc M | 91.26 |
   | nnU-Net (orijinal) | 91.24 |
   | SegResNet (Auto3DSeg) | 90.79 |
   | SwinUNETR | 90.68 |
   | nnFormer | 90.22 |

   Çalışma ayrıca CNN tabanlı U-Net'lerin Transformer ve Mamba modellerinden tutarlı biçimde iyi ya da eşit olduğunu gösteriyor.
2. **Kazanan sistemler farklı ailelerden modellerin ensemble'ı:** BraTS 2023 ve 2024 yetişkin glioma birincisi nnU-Net + MedNeXt + Swin-UNETR; BraTS-Africa 2024 birincisi nnU-Net + MedNeXt.
3. **Sahra Altı Afrika (SSA) verisinde MedNeXt'in yayımlanmış güçlü sonucu var** (validation ort. Dice 0.896).

**Bizim için anlamı:** üçüncü model tek başına en yüksek Dice'a göre değil, şu ölçütlere göre seçilmeli:
- **(a)** mevcut iki modelden farklı hata yapması (ensemble faydası bundan gelir),
- **(b)** sabit Kaggle eğitim bütçesine uyması,
- **(c)** yerel arayüzde (GPU ve yalnız CPU) çalıştırılabilmesi.

## 3. Kendi donanımımızda ölçülen değerler

RTX 3050 Laptop; 96³ yama, batch 1, karma hassasiyet. CPU sütunu tek yama ölçümünden tam hacme ölçeklenmiş tahmindir.

| Aday | Parametre | Eğitim adımı | Tepe VRAM | CPU tam hacim (tahmini) | ONNX |
|---|---|---|---|---|---|
| 3D U-Net (mevcut) | 4.8 M | 52 ms | — | ~6 sn | ✅ |
| SwinUNETR (mevcut) | 15.7 M | 469 ms | — | ~160 sn (OpenVINO ile ~35 sn) | ✅ |
| SegResNet f16 | 4.7 M | 138 ms | 1.6 GB | ~34 sn | ✅ |
| SegResNetDS f32 | 19.9 M | 204 ms | 3.0 GB | ~100 sn | ✅ |
| DynUNet (nnU-Net mimarisi) | 31.0 M | 217 ms | 2.0 GB | ~93 sn | ✅ |
| **MedNeXt-S k3** | 5.2 M | 1198 ms | 3.7 GB | ~155 sn | ✅ |

## 4. Değerlendirme

| Ölçüt (ağırlık) | SegResNet f16 | SegResNetDS f32 | DynUNet (nnU-Net) | MedNeXt-S k3 |
|---|---|---|---|---|
| Mevcut iki modelden farklılık (%30) | Orta: residual CNN | Orta: residual CNN + derin denetim | **Düşük:** mevcut U-Net ile aynı aile | **Yüksek:** ConvNeXt bloğu, genişleyen MLP, depthwise evrişim |
| BraTS / SSA kanıtı (%20) | Orta | Orta-yüksek (Auto3DSeg varsayılanı) | Yüksek (altın standart) | **En yüksek:** kazanan ensemble'larda ve SSA'da |
| Kaggle maliyeti (%20) | **En düşük** | Düşük | Düşük | **Yüksek** |
| Yerel CPU/web çıkarımı (%15) | **En hızlı** | Orta | Orta | Yavaş (OpenVINO gerekir) |
| Entegrasyon riski (%15) | En düşük | Düşük | Düşük | Orta: kısa bütçede yetersiz eğitilme riski |

- **MedNeXt-S:** bilimsel olarak en güçlü ve en "farklı" aday; kazanan BraTS ensemble'larının mimari bileşimini tamamlıyor. Bedeli eğitim süresi ve web hızı.
- **SegResNetDS:** en dengeli pratik aday; farklılığı orta.
- **DynUNet:** güçlü ama mevcut U-Net'e fazla benzediği için ensemble katkısı sınırlı olabilir.
- **SegResNet f16:** en hafif ve en hızlı; kapasitesi en düşük.

## 5. Karar: MedNeXt-S (k=3)

| Kriter | Karşılık |
|---|---|
| Mevcut iki modelden **farklı** çalışan model | Üç model üç tasarım ilkesini temsil eder: klasik yerel evrişim (U-Net), pencereli öz-dikkat (SwinUNETR) ve **dikkat kullanmadan Transformer tasarım ilkelerini evrişimle uygulayan** ConvNeXt mimarisi (MedNeXt). Araştırma sorusu: Transformer'ın faydası dikkat mekanizmasından mı, tasarım ilkelerinden mi geliyor? |
| **Yüksek doğruluk** | nnU-Net Revisited BraTS21 karşılaştırmasında en yüksek skor MedNeXt'te |
| **Web'de uzun bekletmeme** | Ölçüldü (tek vaka, kırpılmış hacim): GPU FP16 tam hacim **1.55 sn**; OpenVINO CPU **48.6 sn** |
| Makalede **gerekçesi açık** seçim | BraTS 2023/2024 ve BraTS-Africa 2024 birincilerinin mimari bileşimiyle örtüşür |

- **Varyant:** S boyutu ve kernel 3, Kaggle bütçesi ve web süresi nedeniyle; kernel 5 (UpKern) ve büyük boyutlar gelecek çalışma.
- **Uygulama:** MONAI'nin resmî `create_mednext("S")` fabrikası. Resmî MIC-DKFZ koduyla blok blok karşılaştırıldı; yapı ve parametre/işlem yükü uyumlu.
- **Doğrulama:** MONAI 1.5.1'de kaydedilen ağırlıklar 1.6.0'da `strict` yükleniyor; aynı girdide çıktı farkı 0.0.

## 6. Adalet ilkesi

Benzer çalışmalardaki yüksek skorların ana kaynağı mimari değil; çok daha uzun eğitim, çapraz doğrulama ensemble'ı ve ensemble çeşitliliği. Bu projede üç model **aynı veri akışı, aynı kayıp, aynı yama boyutu ve aynı epoch bütçesiyle** eğitildi. Yalnız bir modele fayda sağlayan özel ayarlar (modele özgü öğrenme oranı, derin denetim) adalet ilkesi gereği alınmadı; resmî reçetelerden yalnız üç modele eşit uygulanabilen ayarlar benimsendi. Bu, "hangi paradigma neyi daha iyi yapıyor" sorusunun geçerliliğinin ön koşuludur. Doğruluk kazancı; çıkarım iyileştirmeleri ve üç farklı paradigmanın ensemble'ı ile hedeflendi.

## 7. Kaynaklar

- Isensee vd., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*, MICCAI 2024. arXiv:2404.09556
- Ferreira vd., *How we won BraTS 2023 Adult Glioma challenge? Just faking it!*, arXiv:2402.17317
- *Optimizing Brain Tumor Segmentation with MedNeXt: BraTS 2024 SSA and Pediatrics*, arXiv:2411.15872
- Roy vd., *MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation*, MICCAI 2023. arXiv:2303.09975

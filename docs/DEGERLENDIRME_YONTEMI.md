# Değerlendirme Yöntemi (Ön-Kayıt Özeti)

> Vitrin sürümü. Ana projedeki (private) ön-kayıt belgesinin özetidir; kilitli seçimlerin kendisi (hangi ensemble yöntemi ve ayarların kazandığı) bu depoda yer almaz. Sonuç sayıları README'dedir.

## 1. İlke: önce plan, sonra sonuç

- Değerlendirme planı **12 Ekim 2026'da**, üçüncü modelin eğitimi başlamadan ve hiçbir test sonucu görülmeden yazılıp kilitlendi.
- Sonradan yapılan her değişiklik, tarih ve gerekçesiyle bir **sapma kaydına** yazıldı. Sonuç görüldükten sonra yapılan analizler açıkça **post-hoc** diye etiketlendi ve birincil sonucu değiştirmedi.
- **Bütün seçimler validation kümesinde yapıldı; test kümesi bir kez kullanıldı.** Seçimler bir kilit dosyasına yazıldı ve SHA-256 ile mühürlendi; test koşusu bu kilidi doğrulamadan başlamaz, test kümesinde daha önce sonuç varsa koşu reddedilir.
- Test sonucu ne çıkarsa çıksın, v2 modelleri ve v2 ensemble'ı birincil sonuçtur; önceki faz (v1) yalnız ön çalışma olarak raporlanır.

## 2. Veri ve bölünüm

- **BraTS 2021:** 1000 eğitim, 125 validation, 126 test vakası; sabit tohumla bir kez üretilmiş, kimlik listesi SHA-256 ile kilitli bölünüm.
- **Birleşik protokol:** üç model, ensemble ve önceki faz modelleri **aynı** ön işleme, aynı kayan pencere çıkarımı ve FP32 hassasiyetle değerlendirildi. Önceki faz modellerinin tarihsel değerleri farklı ayarlarla üretildiği için adil karşılaştırma amacıyla hepsi yeniden ölçüldü.

## 3. Uç noktalar

- **Birincil:** hasta-bazlı ortalama Dice = mean(WT, TC, ET). Referansında ET olmayan vakalar ET ortalamasına katılmaz; bu vakalardaki yanlış-pozitif ET hacmi ayrıca raporlanır.
- **İkincil:** bölge başına Dice ve IoU, HD95, hacim hatası (cm³), model başına çıkarım süresi.

## 4. Ensemble ve çıkarım iyileştirmeleri

- Dört aday birleştirme yöntemi (etiket çoğunluğu, yumuşak oylama, hiyerarşik bölge oylaması, validation başarısıyla ağırlıklı oylama) ve çıkarım zamanı iyileştirmeleri (ayna TTA, küçük ET bölgesi temizleme) karşılaştırıldı.
- Seçim sırası ve kuralı önceden yazıldı: her adımda validation birincil uç noktası en yüksek olan seçilir; fark 0.001'den küçükse **daha basit** seçenek kalır.
- Testte bütün adaylar şeffaflık için raporlandı; birincil sonuç yalnız kilitlenen adaydır.

## 5. İstatistik

- Eşleştirilmiş Wilcoxon işaretli sıra testi: (a) ensemble ile her tekil model, (b) tekil modellerin ikili karşılaştırmaları, (c) aynı mimaride v2 ile v1.
- Çoklu karşılaştırma düzeltmesi: Holm; α = 0.05, çift yönlü.
- Ortalama farklar için bootstrap %95 güven aralığı: 10 000 yeniden örnekleme, sabit tohum.

## 6. Modeller arası uyum: referans maskesiz kalite uyarısı

- Her vaka için **uyum skoru**: üç modelin WT, TC ve ET maskeleri arasındaki ikili Dice'ların ortalaması. Referans maske kullanmaz; bu yüzden klinik kullanımda da hesaplanabilir.
- Raporlananlar: uyum skoru ile ensemble Dice'ı arasındaki Spearman korelasyonu ve **başarısız vakayı** ayırt etme AUROC'u. Başarısız vaka tanımı önceden sabittir: ensemble ortalama Dice < 0.80.
- Bu analiz klinik doğrulama değildir; referanssız kalite uyarısı için ön kanıttır.

## 7. Harici kohortlar

Modeller, eğitimde hiç görmedikleri iki kurumun verisinde **kilitli ayarlarla, hiçbir şey yeniden seçilmeden ve tek seferde** sınandı. Dahil etme kuralları ve uç noktalar, kohortun hiçbir sonucu görülmeden yazıldı.

| Kohort | Kaynak | Analiz |
|---|---|---|
| **BraTS-Africa** (Nijerya, 1.5 T) | Adewole vd., TCIA, doi:[10.7937/v8h6-8x67](https://doi.org/10.7937/v8h6-8x67), CC BY 4.0. Etiketler uzman nöroradyologlarca düzeltilip onaylandı | Glioma vakaları birincil; glioma dışı tümörler keşifsel (dağılım dışı genelleme) |
| **UTSW-Glioma** (ABD) | Reddy vd., TCIA, doi:[10.7937/DFAE-1B86](https://doi.org/10.7937/DFAE-1B86) | Birincil referans: **elle düzeltilmiş** maskeler (otomatik ön-etiket değil) |

- **Veri erişimi:** koşular, bu koleksiyonların Kaggle'daki kamuya açık kopyaları üzerinden Kaggle GPU'larında yapıldı. Kopyalar, yerelde bulunan ortak vakalarla voksel düzeyinde (biçimden bağımsız SHA-256 özeti) karşılaştırıldı: BraTS-Africa 11/11 ve UTSW 11/11 aynı. Beş checkpoint'in ve kilidin SHA-256 özetleri koşudan önce doğrulandı.
- **Referans dosyası seçimi:** birden çok maske dosyası olan vakalarda görüntüyle aynı geometride (boyut + affine) olan seçilir; belirsiz vakalar dışlanır.
- **Belgelenmemiş etiketler:** veri tanımlayıcısında olmayan etiket kodu içeren vakalar birincil analizden çıkarılıp yalnız etiket anlamından bağımsız bir **WT duyarlılık analizinde** raporlandı; eşleme tahmin edilmedi.
- **Bilinen sınırlılık (önceden yazıldı):** kurumların etiketleme taksonomisi farklı olabilir. Örneğin UTSW'de kontrast tutmayan tümör ödem olarak kodlanırken BraTS'te tümör çekirdeğine dahildir; bu fark TC metriğini doğrudan etkiler ve README'de ayrıca açıklanır.

## 8. Sapma kaydı (özet)

| Tarih | Değişiklik | Durum |
|---|---|---|
| 12.10.2026 | Çıkarım zamanı iyileştirmeleri plana eklendi | Hiçbir sonuç görülmeden |
| 12.10.2026 | Üç model aynı, daha uzun bütçeyle sıfırdan eğitilecek; v2 birincil sonuç kuralı | Hiçbir v2 sonucu yokken |
| 14.10.2026 | Usul: seçim ve test iki ayrı koşuda; test kilidi doğrular, yeniden seçim yapılmaz | Değerlendirme sonucu yokken |
| 15.10.2026 | v1 modelleri ayrı koşuda aynı birleşik protokolle test edildi; karşılaştırmalar koşudan önce tanımlandı | v2 seçimleri sabitken; hiçbir seçimi etkilemez |
| 15.10.2026 | İki harici kohort, referans maske ve dahil etme kuralları eklendi | Kohort sonuçları görülmeden |
| 15.10.2026 | UTSW'de TC'si boş vakalar ayrıca raporlandı | **Post-hoc**, birincili değiştirmez |

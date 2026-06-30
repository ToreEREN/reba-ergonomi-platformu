# REBA Ergonomik Risk Analiz Platformu

Bu proje; görüntü, video ve webcam akışından insan duruşunu algılayarak REBA
ergonomik risk skoru üreten bir karar destek prototipidir.

## Güncel mimari

1. MediaPipe Pose ile 33 vücut noktası ve görünür 3B noktalar çıkarılır.
2. YOLO11 Pose, mevcut öğrenilmiş açı modelinin geriye uyumlu girdisini üretir.
3. Extra Trees modeli 19 gizli/vücut açısını tahmin eder.
4. Görüntüden doğrudan ölçülebilen boyun, gövde, kol, dirsek, bilek, diz ve
   ayak açıları REBA hesaplamasında önceliklidir.
5. Merkezî REBA motoru standart A/B/C tablolarını; yük, kavrama, aktivite ve
   manuel duruş modifikatörleriyle birleştirir.
6. Streamlit arayüzü görsel, video ve webcam analizi ile "Neden bu skor?"
   açıklamasını sunar.

> Bu yazılım araştırma prototipidir; sertifikalı iş güvenliği ölçüm cihazının
> veya ergonomi uzmanının yerini almaz.

## Windows kurulumu

PowerShell'i proje klasöründe açın:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Kurulum bittikten sonra:

```powershell
.\run.ps1
```

Alternatif manuel kurulum:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_project.py
python -m streamlit run app.py
```

## Doğrulama

```powershell
python verify_project.py
python -m unittest discover -s tests -v
```

Beklenen kazanan model test sonucu `macro R² ≈ 0.615` değeridir. Metrikler
`ModelExperiments/leaderboard.csv` ve `extra_trees_metrics.csv` içindedir.

Rapor sonrası eklenen güvenilirlik katmanı şunları da denetler:

- eşli Wilcoxon testi ve bootstrap %95 güven aralıkları,
- yakın-kopya ve açı aralığı kontrolleri,
- Extra Trees ağaçları arasındaki tahmin belirsizliği,
- özellik şeması ile model/scaler/YOLO dosyalarının SHA-256 bütünlüğü,
- açı tahminlerinden sentetik REBA-proxy skoru, risk matrisi ve Bland–Altman analizi.

Çalıştırılmış araştırma notebook'u
`research/REBA_Model_Guvenilirlik_Analizi.ipynb` dosyasındadır. Notebook sonuçları
uzman etiketli saha doğrulaması değildir; sentetik açı verisindeki iç doğrulamadır.

## Proje yapısı

```text
app.py                     Streamlit uygulaması
Utils/                     Poz, açı, MediaPipe ve REBA modülleri
ModelExperiments/          Üretim modeli ve doğrulama sonuçları
BestModel/                 Girdi ölçekleyici ve eski model araştırma çıktıları
research/                  Yeniden üretilebilir güvenilirlik notebook'u
ml/                        Model eğitim/karşılaştırma betikleri
tests/                     Birim ve uçtan uca smoke testleri
Output/                    Örnek sonuçlar
requirements.txt           Python bağımlılıkları
verify_project.py          Hızlı proje doğrulaması
```

## Modeli yeniden eğitme

Arşivlenmiş özellik/gerçek hedef dosyaları mevcutsa:

```powershell
python ml/train_benchmarks.py
```

Test satırları eğitimden hash ile çıkarılır ve veri artırma yalnızca eğitim
bölümüne uygulanır. Yeni gerçek saha verisi toplarken `DATA_REQUIREMENTS.md`
kurallarına uyulmalıdır.

## GitHub'a gönderme

Büyük model dosyaları için Git LFS önerilir:

```powershell
git lfs install
git add .
git commit -m "REBA platformu ilk paylaşılabilir sürüm"
git push
```

`ModelExperiments_noaug/` ve `ModelExperiments_aug2/` yalnızca yerel deney
çıktılarıdır ve paylaşım dışıdır.

## Bilinen sınırlar

- Tek kameradan derinlik ve gizli eklem rotasyonları yaklaşık değerlerdir.
- En iyi sonuç için bütün vücut görünmeli ve kamera dik konumlandırılmalıdır.
- Webcam analizinde şu an tek kişi değerlendirilir.
- Yük, kavrama ve aktivite bilgileri kullanıcı/iş süreci girdisi gerektirir.

"# reba-ergonomi-platformu" 

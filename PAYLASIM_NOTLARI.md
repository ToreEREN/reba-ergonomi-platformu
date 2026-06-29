# Paylaşım paketi notları

## Pakete dahil edilenler

- Çalışan Streamlit uygulaması ve yardımcı Python modülleri
- MediaPipe 33 noktalı iskelet ve REBA hesap motoru
- Kazanan Extra Trees modeli, ölçekleyici ve YOLO ağırlığı
- Model metrikleri ve yeniden eğitim betikleri
- Testler, doğrulama komutu ve örnek çıktı
- Araştırma notebook'ları ve proje raporu

## Bilerek dışarıda bırakılanlar

- `__pycache__` ve geçici dosyalar
- Sanal Python ortamları
- `ModelExperiments_noaug` ve `ModelExperiments_aug2` model kopyaları
- Kullanılmayan Random Forest ve eski Keras/H5 model ağırlıkları
- Git geçmişi ve yerel Streamlit sonuç klasörleri

Üretimde kullanılan ana dosya `ModelExperiments/extra_trees.joblib` dosyasıdır.
Arkadaşlarınız değişiklik yapmadan önce yeni bir Git dalı açmalı ve testleri
çalıştırmalıdır.

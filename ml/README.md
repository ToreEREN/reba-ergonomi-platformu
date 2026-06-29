# Model deneyleri

Bu klasör, model seçimini notebook hücrelerinden ayıran yeniden üretilebilir
deney hattıdır. Test arşivindeki satırlar eğitim havuzundan hash ile çıkarılır;
artırma yalnızca eğitim bölmesine uygulanır. Seçim, REBA hesabında kullanılan
19 açı hedefinin bağımsız test performansına göre yapılır.

```powershell
python ml/train_benchmarks.py
```

`ModelExperiments/leaderboard.csv` karşılaştırmayı, `manifest.json` ise veri
sözleşmesini ve deney ayarlarını kaydeder.

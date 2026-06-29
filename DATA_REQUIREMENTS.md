# Üretim için veri gereksinimleri

Mevcut tahmin arşivi model karşılaştırması için kullanılabilir; ancak gerçek
saha geçerliliğini tek başına kanıtlamaz. YOLO Pose çıktısındaki üçüncü değer
derinlik değil güven skorudur. Sentetik Blender `z` koordinatı ile bu değerin
aynı özellik alanına konması yasaktır.

Yeni veri setinde her örnek için aşağıdakiler saklanmalıdır:

- ham görüntü veya video/kare kimliği;
- kişi/çalışan ve hareket/sekans kimliği;
- kamera kimliği, çözünürlük ve mümkünse kalibrasyon parametreleri;
- piksel koordinatları `(x, y)` ve bunlardan ayrı `confidence`;
- varsa metre cinsinden 3B `(X, Y, Z)` ve kaynağı;
- 19 açı hedefi, ölçüm yöntemi ve birimi;
- yük, kavrama, statik duruş, tekrarlılık ve ani hareket etiketleri;
- sentetik/gerçek alan etiketi.

Train/validation/test ayrımı kare bazında değil, kişi ve sekans gruplarıyla
yapılmalıdır. Aynı videonun komşu kareleri farklı bölmelere giremez. Artırma
yalnızca eğitim bölmesine uygulanır. Sentetik görüntüler gerçek test kümesine
karıştırılmaz.

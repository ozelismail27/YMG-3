"""
BÖLÜM 6 & 7: Basit Bir CNN Modeli Oluşturulması ve Mimarinin İncelenmesi
Bu betik, Fashion-MNIST veri setini yükler, CNN mimarisini eğitir,
sonuçları ve tahminleri görselleştirir ve Bölüm 7'deki mimari sorularını yanıtlar.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

CLASS_NAMES_TR = [
    'Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Palto',
    'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Bot'
]

def build_model(first_conv_filters=8):
    """Verilen filtre sayısına göre CNN modelini inşa eder."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        # Birinci konvolüsyon
        tf.keras.layers.Conv2D(
            first_conv_filters, (3, 3),
            activation='relu',
            name='conv2d_1'
        ),
        # Birinci pooling
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_1'),
        # İkinci konvolüsyon
        tf.keras.layers.Conv2D(
            16, (3, 3),
            activation='relu',
            name='conv2d_2'
        ),
        # İkinci pooling
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_2'),
        # Çok boyutlu özellikleri tek boyuta dönüştür
        tf.keras.layers.Flatten(name='flatten'),
        # Tam bağlantılı katman
        tf.keras.layers.Dense(64, activation='relu', name='dense_64'),
        # 10 sınıf için çıktı
        tf.keras.layers.Dense(10, activation='softmax', name='output_softmax')
    ])
    return model

def run_bolum6_and_7():
    print("=" * 60)
    print("BÖLÜM 6: FASHION-MNIST İLE CNN MODELİNİN EĞİTİLMESİ")
    print("=" * 60)

    # Rastgelelik tohumu
    tf.keras.utils.set_random_seed(42)

    # A. Veri setinin yüklenmesi
    print("\n[A] Fashion-MNIST veri seti yükleniyor...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalizasyon
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Kanal boyutu ekleme (28, 28) -> (28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"Eğitim verisi boyutu : {x_train.shape} (Etiketler: {y_train.shape})")
    print(f"Test verisi boyutu   : {x_test.shape} (Etiketler: {y_test.shape})")

    # B. CNN mimarisinin oluşturulması
    print("\n[B] Model Mimarisi Oluşturuluyor:")
    model = build_model(first_conv_filters=8)
    model.summary()

    # C. Modelin derlenmesi ve eğitilmesi
    print("\n[C] Model Derleniyor ve Eğitiliyor (5 Epoch)...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Sonucu -> Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f} (%{test_accuracy*100:.2f})")

    # D. Eğitim sonuçlarının görselleştirilmesi
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim (Training)', marker='o', color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='Doğrulama (Validation)', marker='s', color='#ff7f0e')
    plt.title('Model Doğruluğu (Accuracy)', fontsize=12, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim (Training)', marker='o', color='#1f77b4')
    plt.plot(history.history['val_loss'], label='Doğrulama (Validation)', marker='s', color='#ff7f0e')
    plt.title('Model Kaybı (Loss)', fontsize=12, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    hist_path = os.path.join("figures", "bolum6_egitim_grafigi.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Eğitim grafiği kaydedildi: {hist_path}")

    # Örnek tahminleri görselleştirme
    print("\nÖrnek test tahminleri görselleştiriliyor...")
    predictions = model.predict(x_test[:10], verbose=0)
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]
        color = 'green' if pred_label == true_label else 'red'
        confidence = np.max(predictions[i]) * 100
        plt.title(f"Tahmin: {CLASS_NAMES_TR[pred_label]}\nGerçek: {CLASS_NAMES_TR[true_label]}\n(%{confidence:.1f})",
                  color=color, fontsize=8)
        plt.axis('off')
    plt.suptitle("Fashion-MNIST Test Örnekleri Üzerinde Model Tahminleri", fontsize=12, weight='bold')
    plt.tight_layout()
    pred_path = os.path.join("figures", "bolum6_ornek_tahminler.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Örnek tahminler kaydedildi: {pred_path}")

    # =========================================================================
    # BÖLÜM 7: MODEL MİMARİSİNİN ANLAŞILMASI - SORU VE CEVAPLAR
    # =========================================================================
    print("\n" + "=" * 60)
    print("BÖLÜM 7: MODEL MİMARİSİNİN ANLAŞILMASI (SORU & CEVAPLAR)")
    print("=" * 60)
    print("""
1. Modelde kaç adet konvolüsyon katmanı vardır?
   -> Modelde 2 adet Conv2D katmanı bulunmaktadır (conv2d_1 ve conv2d_2).

2. İlk konvolüsyon katmanında neden 8 filtre kullanılmıştır?
   -> İlk katman ham görüntüyle doğrudan temas eder. Bu aşamada kenarlar, köşeler,
      ışık/gölge geçişleri ve basit çizgiler gibi düşük seviyeli (low-level) temel
      öznitelikleri yakalamak için 8 filtre yeterli ve hesaplama açısından ekonomiktir.

3. İkinci konvolüsyon katmanında neden 16 filtre kullanılmıştır?
   -> Derinleştikçe model daha karmaşık, soyut ve birleşik desenleri (yaka, kol, doku,
      düğme gibi yüksek seviyeli öznitelikleri) öğrenir. Daha fazla öznitelik kombinasyonunu
      temsil edebilmek için filtre sayısı iki katına (16'ya) çıkarılmıştır. CNN mimarilerinde
      uzamsal boyut (en x boy) küçüldükçe kanal sayısının artırılması standart bir prensiptir.

4. Konvolüsyon katmanlarından sonra görüntü boyutları nasıl değişmiştir?
   -> Giriş         : (28, 28, 1)
   -> 1. Conv2D (3x3): (26, 26, 8)  -> Valid padding nedeniyle 28 - 2 = 26
   -> 1. MaxPool(2x2): (13, 13, 8)  -> 26 / 2 = 13
   -> 2. Conv2D (3x3): (11, 11, 16) -> Valid padding nedeniyle 13 - 2 = 11
   -> 2. MaxPool(2x2): (5, 5, 16)   -> floor(11 / 2) = 5

5. Pooling katmanları ne gibi bir katkı sağlamıştır?
   -> a) Uzamsal boyutu (en x boy) küçülterek öznitelik haritasını 26x26'dan 13x13'e,
         sonra da 11x11'den 5x5'e düşürmüştür.
      b) Flatten sonrasındaki nöron sayısını ve dolayısıyla Dense katmanındaki ağırlık
         sayısını dramatik biçimde azaltarak aşırı öğrenmeyi (overfitting) önlemiştir.
      c) Görüntüdeki hafif ötelemelere karşı dayanıklılık (translation invariance) kazandırmıştır.

6. Flatten katmanının çıktısı kaç elemanlıdır?
   -> İkinci pooling çıkışı (5, 5, 16) boyutundadır.
   -> Flatten işlemi bu 3 boyutlu tensörü tek boyuta yayar:
      5 x 5 x 16 = 400 elemanlı (vektör).

7. Dense katmanındaki 64 sayısı neyi ifade etmektedir?
   -> Bu katmandaki tam bağlantılı (Fully Connected) nöron sayısını ifade eder.
      Flatten katmanından gelen 400 adet özelliği 64 boyutlu daha yoğun ve anlamlı bir
      temsile sıkıştırarak doğrusal olmayan (ReLU) birleştirme yapar.

8. Son katmanda neden 10 nöron bulunmaktadır?
   -> Fashion-MNIST veri setinde toplam 10 farklı giysi sınıfı (T-shirt, pantolon, elbise vb.)
      bulunmaktadır. Her bir nöron, girdinin o sınıfa ait olma skorunu hesaplar.

9. Softmax çıkışındaki en yüksek değer neyi ifade eder?
   -> Modelin girdiyi en yüksek olasılıkla atadığı tahmini sınıfı ve bu tahmine olan
      güven düzeyini (confidence) ifade eder (Örn: Sınıf 7 için 0.94 -> %94 ihtimalle Spor Ayakkabı).

10. Modelin tahmin başarısını hangi metrikle ölçtünüz?
   -> Başarı 'Accuracy' (Doğruluk) metriği ile ölçülmüştür. Toplam doğru tahmin edilen
      örnek sayısının tüm test örnekleri sayısına oranını ifade eder (Doğru Tahmin / Toplam Örnek).
    """)

    return history, test_loss, test_accuracy

if __name__ == "__main__":
    run_bolum6_and_7()

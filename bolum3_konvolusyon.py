"""
BÖLÜM 3: Konvolüsyon İşleminin İncelenmesi ve Filtrelerin Görselleştirilmesi
Bu betik, Conv2D katmanının ağırlıklarını, boyut dönüşümlerini inceler ve
başlangıç filtre ağırlıklarını görselleştirir.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def run_bolum3():
    print("=" * 60)
    print("BÖLÜM 3: KONVOLÜSYON İŞLEMİNİN İNCELENMESİ")
    print("=" * 60)

    # Rastgelelik kontrolü (tekrarlanabilirlik)
    tf.keras.utils.set_random_seed(42)

    # 3x3 boyutunda 16 filtreli konvolüsyon katmanı
    conv = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal'
    )

    # Örnek giriş görüntüsü (1 adet 128x128 boyutunda tek kanallı gri görüntü)
    dummy_input = tf.random.normal((1, 128, 128, 1))

    # Konvolüsyon işlemi
    output = conv(dummy_input)

    # Ağırlıkları al
    weights, biases = conv.get_weights()

    print("1. Giriş boyutu              :", dummy_input.shape)
    print("2. Çıkış boyutu              :", output.shape)
    print("3. Filtre ağırlıkları boyutu :", weights.shape)
    print("4. Bias boyutu               :", biases.shape)
    print("\n--- İlk Filtrenin Ağırlıkları (3x3) ---")
    print(weights[:, :, 0, 0])

    print("\n--- TEORİK SORULARIN CEVAPLARI ---")
    print("""
[1] 16 Filtre Ne Anlama Gelir?
    Katmanın giriş görüntüsünden 16 farklı öznitelik haritası (feature map)
    öğreneceğini ifade eder. Her bir filtre kenar, köşe, doku gibi farklı bir
    özelliği tespit etmek üzere eğitilir.

[2] Her Filtrenin Boyutu Nedir?
    Her filtre 3x3 boyutundadır ve girişin kanal sayısı 1 olduğu için
    3 x 3 x 1 = 9 adet ağırlık (weight) parametresine sahiptir.

[3] weights Değişkeninin Boyutu: (3, 3, 1, 16)
    - 3, 3 : Kernel (filtre) yükseklik ve genişliği
    - 1    : Giriş kanal sayısı (in_channels)
    - 16   : Filtre / Çıkış kanal sayısı (out_channels)
    Toplam ağırlık sayısı = 3 x 3 x 1 x 16 = 144 adet.

[4] biases Değişkeninin Boyutu: (16,)
    Her bir filtrenin / öznitelik haritasının bir adet bias (kayma) sabiti vardır.
    Toplam 16 bias değeri mevcuttur. Toplam parametre = 144 + 16 = 160.

[5] Giriş 128x128 iken Çıkış Neden 126x126 Olmuştur?
    Varsayılan dolgu (padding) modu 'valid' (padding yapılmaz) olarak belirlenmiştir.
    3x3 filtre, kenarlardan 1'er piksel pay bırakarak kaydığı için formül:
    Çıkış = ((Giriş - Filtre) / Adım) + 1 = ((128 - 3) / 1) + 1 = 126 piksel.

[6] kernel_initializer='he_normal' Ne Anlama Gelir?
    He (Kaiming) Normal başlatıcısı, ağırlıkları sıfır ortalamalı ve
    standart sapması sqrt(2 / fan_in) olan bir Gauss dağılımından rastgele seçer.
    Özellikle ReLU aktivasyon fonksiyonu ile birlikte kullanıldığında eğim kaybolması
    (vanishing gradient) ve patlamasını engelleyerek eğitimin kararlı başlamasını sağlar.
    """)

    # Bütün filtrelerin başlangıç ağırlıklarını görselleştirme (4x4 ızgara)
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("3. Bölüm: 16 Başlangıç Konvolüsyon Filtresi (3x3, He-Normal)", fontsize=14, weight='bold')

    for i in range(16):
        ax = axes[i // 4, i % 4]
        # weights: (3, 3, 1, 16) -> i. filtre: weights[:, :, 0, i]
        kernel = weights[:, :, 0, i]
        im = ax.imshow(kernel, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Filtre #{i+1}", fontsize=10)
        ax.axis('off')
        # Hücre içine sayısal değerleri yaz
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{kernel[r, c]:.2f}",
                        ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join("figures", "bolum3_filtreler.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Filtre görselleştirmesi başarıyla kaydedildi: {fig_path}")

if __name__ == "__main__":
    run_bolum3()

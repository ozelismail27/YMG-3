"""
BÖLÜM 4: Pooling İşleminin İncelenmesi
Bu betik, 2x2 Max Pooling işleminin çalışma prensibini, boyut azaltımını ve
hesaplama mantığını görselleştirerek inceler.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def run_bolum4():
    print("=" * 60)
    print("BÖLÜM 4: POOLING İŞLEMİNİN İNCELENMESİ")
    print("=" * 60)

    # Örnek 4x4 giriş matrisi
    input_data = tf.constant([
        [[1., 3., 2., 4.],
         [5., 6., 1., 2.],
         [7., 2., 8., 3.],
         [4., 1., 5., 9.]]
    ])

    # Boyutları düzenle: Batch=1, Height=4, Width=4, Channels=1
    input_data = tf.reshape(input_data, (1, 4, 4, 1))

    # 2x2 Max Pooling, adım (stride) = 2
    pool = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )

    output = pool(input_data)

    mat_in = input_data.numpy()[0, :, :, 0]
    mat_out = output.numpy()[0, :, :, 0]

    print("Giriş Matrisi (4x4):")
    print(mat_in)
    print("\nPooling Çıkış Matrisi (2x2):")
    print(mat_out)

    print("\n--- ELLE HESAPLAMA VE KARŞILAŞTIRMA ---")
    reg1 = mat_in[0:2, 0:2]  # [[1, 3], [5, 6]] -> max = 6
    reg2 = mat_in[0:2, 2:4]  # [[2, 4], [1, 2]] -> max = 4
    reg3 = mat_in[2:4, 0:2]  # [[7, 2], [4, 1]] -> max = 7
    reg4 = mat_in[2:4, 2:4]  # [[8, 3], [5, 9]] -> max = 9

    hand_calc = np.array([
        [np.max(reg1), np.max(reg2)],
        [np.max(reg3), np.max(reg4)]
    ])

    print("1. Sol-Üst 2x2 Bölge    : \n", reg1, f"-> En Büyük = {np.max(reg1)}")
    print("2. Sağ-Üst 2x2 Bölge    : \n", reg2, f"-> En Büyük = {np.max(reg2)}")
    print("3. Sol-Alt 2x2 Bölge    : \n", reg3, f"-> En Büyük = {np.max(reg3)}")
    print("4. Sağ-Alt 2x2 Bölge    : \n", reg4, f"-> En Büyük = {np.max(reg4)}")

    print("\nElle Hesaplanan Matris:\n", hand_calc)
    eslesme = np.array_equal(hand_calc, mat_out)
    print(f"Program Çıktısı ile Elle Hesaplama Uyuşuyor mu? -> {'EVET (Birebir Uyuşuyor)' if eslesme else 'HAYIR'}")

    print("\n--- TEORİK SORULARIN CEVAPLARI ---")
    print("""
[1] Giriş ve Çıkış Boyutları:
    - Giriş Boyutu: (1, 4, 4, 1) -> 1 adet, 4 yükseklik, 4 genişlik, 1 kanal.
    - Çıkış Boyutu: (1, 2, 2, 1) -> 1 adet, 2 yükseklik, 2 genişlik, 1 kanal.
    - Açıklama    : 2x2 pencere ve strides=2 kullanıldığı için yükseklik ve genişlik
                    tam olarak yarıya (4/2 = 2) inmiştir. Toplam piksel sayısı 16'dan 4'e düşmüştür (%75 azalma).

[2] Max Pooling Kullanılmasının CNN Açısından Avantajları:
    a) Boyut Azaltma (Downsampling): Veri boyutunu küçülterek sonraki katmanlardaki parametre ve işlem sayısını azaltır.
    b) Aşırı Öğrenmeyi (Overfitting) Önleme: Detay pikselleri atıp sadece en belirgin özellikleri tutarak genelleştirme gücünü artırır.
    c) Konumsal Değişmezlik (Translation Invariance): Nesne görüntü içinde hafifçe kaydığında veya döndüğünde bile en yüksek değer korunduğu için model bu değişimlere dirençli hale gelir.
    d) Alıcı Alanı (Receptive Field) Genişletme: Sonraki katmanların görüntünün daha büyük bir alanını görebilmesini sağlar.
    """)

    # Görselleştirme: Giriş ve Çıkış Matrisleri
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Giriş matrisi
    im1 = axes[0].imshow(mat_in, cmap='Blues', vmin=0, vmax=10)
    axes[0].set_title("Giriş Matrisi (4x4)", fontsize=12, weight='bold')
    for r in range(4):
        for c in range(4):
            axes[0].text(c, r, f"{mat_in[r, c]:.0f}", ha='center', va='center',
                         color='black' if mat_in[r, c] < 7 else 'white', fontsize=12, weight='bold')

    # Bölge ayrım çizgileri (2x2)
    axes[0].axvline(1.5, color='red', linestyle='--', linewidth=2)
    axes[0].axhline(1.5, color='red', linestyle='--', linewidth=2)

    # Çıkış matrisi
    im2 = axes[1].imshow(mat_out, cmap='Greens', vmin=0, vmax=10)
    axes[1].set_title("2x2 Max Pooling Sonucu (2x2, stride=2)", fontsize=12, weight='bold')
    for r in range(2):
        for c in range(2):
            axes[1].text(c, r, f"{mat_out[r, c]:.0f}", ha='center', va='center',
                         color='black' if mat_out[r, c] < 7 else 'white', fontsize=14, weight='bold')

    plt.tight_layout()
    fig_path = os.path.join("figures", "bolum4_maxpooling.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Max Pooling görselleştirmesi kaydedildi: {fig_path}")

if __name__ == "__main__":
    run_bolum4()

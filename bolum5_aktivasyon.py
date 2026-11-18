"""
BÖLÜM 5: ReLU ve Softmax Aktivasyon Fonksiyonlarının İncelenmesi
Bu betik, ReLU ve Softmax fonksiyonlarının matematiksel davranışlarını,
kod çıktılarını ve teorik temellerini görselleştirerek inceler.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def run_bolum5():
    print("=" * 60)
    print("BÖLÜM 5: AKTİVASYON FONKSİYONLARI (ReLU & Softmax)")
    print("=" * 60)

    # -----------------------------
    # A. ReLU İncelenmesi
    # -----------------------------
    print("\n--- A. ReLU İNCELENMESİ ---")
    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    relu = tf.keras.layers.ReLU()
    y_relu = relu(x)

    print("Giriş       :", x.numpy())
    print("ReLU Çıkışı :", y_relu.numpy())

    print("""
[ReLU Soruları ve Yanıtları]
1. Negatif Değerler Neden Sıfırlanmıştır?
   - ReLU matematiksel olarak f(x) = max(0, x) kuralını uygular.
   - Giriş sıfırdan küçük olduğunda çıktı 0 yapılır. Bu sayede 'seyrek aktivasyon' (sparse activation)
     sağlanır; yani ağdaki nöronların yalnızca ilgili özellikleri tespit edenleri aktifleşir,
     gereksiz sinyaller bastırılır.

2. Pozitif Değerler Neden Korunmuştur?
   - x > 0 için fonksiyonun eğimi (türevi) sabittir ve 1'e eşittir: f'(x) = 1.
   - Bu durum, Sigmoid veya Tanh gibi doyuma uğrayan fonksiyonlarda görülen 'kaybolan gradyan' (vanishing gradient)
     sorununu ortadan kaldırır; gradyan geriye yayılım sırasında zayıflamadan iletilir.

3. ReLU Kullanılmasaydı Ne Gibi Bir Farklılık Oluşabilirdi?
   - Hiçbir doğrusal olmayan (non-linear) aktivasyon kullanılmasaydı, üst üste kaç konvolüsyon
     veya yoğun katman eklenirse eklensin bütün ağ tek bir doğrusal matris çarpımına (W_total * x)
     indirgenirdi. Bu da modelin karmaşık, doğrusal olmayan desenleri (görüntü özniteliklerini) öğrenmesini imkansız kılardı.
   - Alternatif olarak Sigmoid veya Tanh kullanılsaydı, derin katmanlarda gradyanlar hızla sıfıra yaklaşır ve ağ öğrenemezdi.
    """)

    # -----------------------------
    # B. Softmax İncelenmesi
    # -----------------------------
    print("\n--- B. Softmax İNCELENMESİ ---")
    logits = tf.constant([[2.0, 1.0, 0.1]])
    softmax = tf.keras.layers.Softmax()
    probabilities = softmax(logits)

    prob_np = probabilities.numpy()
    sum_prob = tf.reduce_sum(probabilities).numpy()
    predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]

    print("Logits (Giriş Skorları) :", logits.numpy())
    print("Softmax Olasılıkları    :", prob_np)
    print("Olasılıkların Toplamı   :", sum_prob)
    print("En Yüksek Olasılıklı Sınıf (Argmax İndeksi):", predicted_class, f"(Olasılık: {prob_np[0][predicted_class]:.4f})")

    print("""
[Softmax Soruları ve Yanıtları]
1. Softmax Hangi Amaçla Kullanılır?
   - Çok sınıflı (multi-class) sınıflandırma problemlerinde, modelin son katmanından çıkan ham sayısal skorları (logits)
     0 ile 1 aralığında geçerli bir olasılık dağılımına dönüştürmek amacıyla kullanılır.

2. Çıktıdaki Değerler Neyi İfade Etmektedir?
   - Giriş örneğinin ilgili sınıfa ait olma olasılığını P(y = c | x) ifade eder.
   - Örneğin [0.659, 0.242, 0.099] çıktısı: Örneğin %65.9 ihtimalle Sınıf 0'a, %24.2 ihtimalle Sınıf 1'e,
     %9.9 ihtimalle Sınıf 2'ye ait olduğunu gösterir.

3. Olasılıkların Toplamı Neden 1'dir?
   - Softmax formülü: P_i = exp(z_i) / sum_j(exp(z_j)).
   - Her elemanın üssü alınarak tüm elemanların üslerinin toplamına bölünür (normalizasyon).
   - Dolayısıyla paydada toplam yer aldığından, bütün sınıfların olasılıklarının toplamı matematiksel olarak kesinlikle 1.0 olur.

4. En Yüksek Olasılığa Sahip Sınıf Nasıl Belirlenir?
   - En büyük olasılık değerine sahip olan indeks tespit edilir. Python/TensorFlow'da bu işlem `argmax` fonksiyonu ile yapılır:
     tahmin_edilen_sinif = np.argmax(probabilities).
    """)

    # Görselleştirme: ReLU ve Softmax Grafikleri
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ReLU grafiği
    x_range = np.linspace(-6, 6, 200)
    y_relu_range = np.maximum(0, x_range)
    axes[0].plot(x_range, y_relu_range, color='blue', linewidth=2.5, label='f(x) = max(0, x)')
    axes[0].scatter(x.numpy(), y_relu.numpy(), color='red', s=60, zorder=5, label='Örnek Noktalar')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_title("ReLU Aktivasyon Fonksiyonu", fontsize=12, weight='bold')
    axes[0].set_xlabel("Giriş (x)")
    axes[0].set_ylabel("Çıkış f(x)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Softmax çubuk grafiği
    classes = [f"Sınıf {i}\n(logit={logits.numpy()[0, i]:.1f})" for i in range(3)]
    bars = axes[1].bar(classes, prob_np[0], color=['#2ca02c', '#1f77b4', '#ff7f0e'], width=0.5)
    axes[1].set_title(f"Softmax Çıktısı (Toplam = {sum_prob:.1f})", fontsize=12, weight='bold')
    axes[1].set_ylabel("Tahmin Olasılığı")
    axes[1].set_ylim(0, 1.0)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.02, f"%{h*100:.1f}", ha='center', va='bottom', weight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join("figures", "bolum5_aktivasyonlar.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Aktivasyon fonksiyonları grafiği kaydedildi: {fig_path}")

if __name__ == "__main__":
    run_bolum5()

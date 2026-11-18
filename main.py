"""
ANA ÇALIŞTIRMA BETİĞİ (main.py)
---------------------------------
Bu betik, ödevin tüm adımlarını (Bölüm 3, 4, 5, 6, 7 ve 8) sırasıyla çalıştırır,
tüm çıktıları ekrana basar ve tüm grafik/görselleri 'figures/' klasörüne kaydeder.

Kullanım:
    python main.py
"""

import sys
import time
from bolum3_konvolusyon import run_bolum3
from bolum4_pooling import run_bolum4
from bolum5_aktivasyon import run_bolum5
from bolum6_model_egitimi import run_bolum6_and_7
from bolum8_deneyler import run_bolum8

def main():
    print("#" * 70)
    print("  DERİN ÖĞRENME VE EVRİŞİMSEL SİNİR AĞLARI (CNN) ÖDEV PROJESİ")
    print("  Veri Seti: Fashion-MNIST | Kütüphane: TensorFlow / Keras")
    print("#" * 70)

    start_total = time.time()

    # Bölüm 3
    run_bolum3()
    print("\n" + "-" * 70 + "\n")

    # Bölüm 4
    run_bolum4()
    print("\n" + "-" * 70 + "\n")

    # Bölüm 5
    run_bolum5()
    print("\n" + "-" * 70 + "\n")

    # Bölüm 6 & 7
    run_bolum6_and_7()
    print("\n" + "-" * 70 + "\n")

    # Bölüm 8
    run_bolum8()

    total_duration = time.time() - start_total
    print("\n" + "#" * 70)
    print(f"  TÜM BÖLÜMLER BAŞARIYLA TAMAMLANDI! Toplam Süre: {total_duration:.2f} saniye")
    print("  Üretilen tüm görseller 'figures/' klasörüne kaydedilmiştir.")
    print("  Ayrıntılı akademik analiz için 'RAPOR.md' ve 'README.md' dosyalarını inceleyebilirsiniz.")
    print("#" * 70)

if __name__ == "__main__":
    main()

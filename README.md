# Derin Öğrenme ve CNN Ödevi (Fashion-MNIST)

Bu projede Evrişimsel Sinir Ağlarının (CNN) temel mantığını, katmanların ne işe yaradığını ve görüntü boyutlarının nasıl değiştiğini inceledim. Uygulama kısmında TensorFlow/Keras kullanarak Fashion-MNIST veri kümesi üzerinde bir sınıflandırma modeli eğittim ve filtre sayısının başarıya etkisini görmek için küçük bir deney yaptım.

## Proje İçeriği

- **`CNN_FashionMNIST_Odevi.ipynb`**: Ödevin tüm bölümlerini, kodlarını, açıklamalarını ve grafiklerini içeren Jupyter Notebook dosyası.
- **`RAPOR.md`**: Ödevdeki tüm teorik soruların cevapları, elle yapılan pooling hesaplaması ve deney sonuçlarını içeren ödev raporu.
- **`main.py`**: Tüm adımları konsoldan sırayla çalıştıran Python betiği.
- **`figures/`**: Kodların ürettiği tüm grafik ve görselleştirmeler:
  - `bolum3_filtreler.png`: 16 adet başlangıç filtresinin ağırlıkları.
  - `bolum4_maxpooling.png`: 4x4 giriş matrisi ve 2x2 pooling sonucu.
  - `bolum5_aktivasyonlar.png`: ReLU ve Softmax grafik çıktısı.
  - `bolum6_egitim_grafigi.png`: Modelin eğitim ve doğrulama grafiği (accuracy/loss).
  - `bolum6_ornek_tahminler.png`: Test verisinden 10 örnek tahmin.
  - `bolum8_deney_karsilastirma.png`: 8 filtre ve 16 filtre modellerinin karşılaştırması.

## Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Projeyi çalıştırmak için iki yöntem kullanabilirsiniz:

1. **Jupyter Notebook ile (Tavsiye edilen):**
   `CNN_FashionMNIST_Odevi.ipynb` dosyasını VS Code veya Jupyter üzerinde açıp hücreleri sırayla çalıştırabilirsiniz.

2. **Terminal üzerinden:**
   ```bash
   python main.py
   ```

## Ödevde Yapılanlar ve Kısaca Sonuçlar

1. **Konvolüsyon Katmanı:** 3x3 boyutunda 16 filtreli bir katman oluşturuldu. Ağırlık tensörünün (3, 3, 1, 16) olduğu ve toplam 144 ağırlık + 16 bias = 160 parametre içerdiği görüldü.
2. **Max Pooling:** 4x4'lük matris elle hesaplandı ve program çıktısıyla karşılaştırıldı. 2x2 pencere ile boyutlar tam yarıya indi (piksel sayısı %75 azaldı).
3. **Aktivasyon Fonksiyonları:** ReLU'nun negatifleri sıfırlayıp pozitifleri koruduğu, Softmax'ın ise çıktıları toplamı 1 olan olasılıklara dönüştürdüğü doğrulandı.
4. **Model Eğitimi:** Fashion-MNIST üzerinde kurulan CNN modeli 5 epoch sonunda test verisinde yaklaşık %87-88 doğruluk elde etti.
5. **Deney (8 Filtre vs 16 Filtre):** İlk katmanda 16 filtre kullanıldığında model daha fazla desen öğrenebildiği için test başarısı yaklaşık %0.9 arttı, ancak eğitim süresi ve parametre sayısı biraz yükseldi.

Detaylı açıklamalar ve matematiksel hesaplamalar için `RAPOR.md` dosyasını inceleyebilirsiniz.

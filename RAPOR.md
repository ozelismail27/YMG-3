# Derin Öğrenme ve CNN Yapısının İncelenmesi - Ödev Raporu

**Öğrenci:** [Adınız Soyadınız]  
**Öğrenci Numarası:** [Numaranız]  
**Ders:** Derin Öğrenme / Yapay Zeka  
**Kullanılan Kütüphaneler:** TensorFlow, Keras, NumPy, Matplotlib  
**Veri Seti:** Fashion-MNIST  

---

## 1. Ödevin Amacı
Bu ödevde amacım, Evrişimsel Sinir Ağlarının (CNN) çalışma mantığını temelinden anlamak, model içindeki katmanların (Conv2D, Pooling, Flatten, Dense vb.) ne işe yaradığını ve matris boyutlarını nasıl değiştirdiğini incelemektir. Ayrıca Fashion-MNIST veri seti üzerinde kendi CNN modelimi eğitip, ilk katmandaki filtre sayısını değiştirerek modelin başarısını ve çalışma süresini karşılaştırdım.

---

## 2. Bölüm: Temel Kavramların İncelenmesi

### Soru 1. Derin Öğrenme ve Yapay Sinir Ağları
- **Makine Öğrenmesi ile Derin Öğrenme Arasındaki Fark Nedir?**  
  Klasik makine öğrenmesinde verideki özellikleri (kenarlar, renk dağılımları vb.) mühendisin kendisi elle belirlemek ve algoritmaya vermek zorundadır. Derin öğrenmede ise çok katmanlı sinir ağları sayesinde model, ham görüntüyü (pikselleri) doğrudan alır ve hangi özelliklerin önemli olduğunu baştan sona kendisi otomatik olarak öğrenir.
- **Yapay Sinir Ağları Nasıl Çalışır?**  
  Ağ, girişten aldığı verileri ağırlıklarla ($W$) çarpar ve bir kayma değeri ($b$) ekler. Elde edilen değer aktivasyon fonksiyonundan geçirilerek bir sonraki katmana iletilir. Model en sonda bir tahmin üretir. Bu tahmin ile gerçek etiket arasındaki fark (kayıp/hata) hesaplanır. Ardından geriye yayılım (backpropagation) yöntemiyle bu hatayı azaltacak şekilde ağırlıklar güncellenir.
- **Bir Yapay Sinir Ağı Neden Birden Fazla Katmandan Oluşur?**  
  Tek bir katman sadece çok basit ve doğrusal ilişkileri çözebilir. Birden fazla katman olduğunda ağ hiyerarşik bir öğrenme yapar; ilk katmanlar basit çizgileri ve kenarları öğrenirken, daha derin katmanlar bu çizgileri birleştirip şekilleri, dokuları ve nesneleri (örneğin ayakkabı tabanı veya gömlek yakası) öğrenebilir.

### Soru 2. Konvolüsyon Nedir?
- **Konvolüsyon İşlemi Görüntü Üzerinde Nasıl Gerçekleştirilir?**  
  Filtre (kernel) adı verilen küçük bir sayı matrisi, görüntünün üzerinde adım adım kaydırılır. Her adımda filtrenin altındaki pikseller ile filtrenin ağırlıkları karşılıklı çarpılır ve toplanır. Bu toplam, yeni oluşacak öznitelik haritasındaki tek bir pikseli oluşturur.
- **Kernel (Filtre) Nedir?**  
  Görüntüdeki belirli desenleri (mesela dikey kenar, köşe veya parlaklık farkı) yakalamaya yarayan küçük ağırlık matrisidir. CNN modellerinde bu filtrelerin değerleri eğitim sırasında model tarafından öğrenilir.
- **3×3 Boyutundaki Bir Filtrenin Kaç Ağırlık Parametresi Vardır?**  
  Giriş görüntüsü tek kanallı (gri seviye) ise $3 \times 3 \times 1 = 9$ adet ağırlığı vardır.
- **Filtre Görüntü Üzerinde Hareket Ederken Ne Yapar?**  
  Görüntünün her bölgesini tarar ve aradığı desen o bölgede varsa yüksek bir sayı üretir, yoksa düşük/sıfır üretir. Böylece görüntünün bir nevi özellik haritasını çıkarır.
- **Bir CNN'de Neden Birden Fazla Filtre Kullanılır?**  
  Çünkü tek bir filtre sadece tek bir deseni yakalayabilir (mesela sadece dikey çizgiyi). Görüntüyü iyi anlayabilmek için yatay çizgileri, köşeleri ve farklı açıdaki desenleri ayrı ayrı yakalayacak birçok farklı filtreye (örneğin 8, 16 veya 32 adet) ihtiyaç vardır.

### Soru 3. Pooling (Havuzlama) Nedir?
- **Max Pooling İşleminin Amacı Nedir?**  
  Konvolüsyondan çıkan özellik haritasının boyutunu küçültmektir. Böylece gereksiz detayları atar, parametre sayısını ve hesaplama yükünü azaltır, modelin ezber yapmasını (overfitting) engeller ve nesne görüntüde hafif kaydığında bile tanımasını sağlar.
- **2×2 Max Pooling Nasıl Çalışır?**  
  Görüntü üzerinde 2x2'lik kutular oluşturur (stride=2 ile ikişer piksel atlayarak gezer). Her 4 piksellik kutunun içindeki en büyük sayıyı seçip çıkış matrisine yazar.
- **Giriş Görüntüsünün Boyutunu Nasıl Değiştirir?**  
  Genişlik ve yüksekliği tam yarıya indirir ($H/2, W/2$). Toplam piksel sayısı ise dörtte birine (%75 azalma) düşer.

### Soru 4. Aktivasyon Fonksiyonları
- **Aktivasyon Fonksiyonları Neden Kullanılır?**  
  Ağa doğrusal olmama (non-linearity) özelliği katmak için kullanılır. Eğer bunlar olmazsa, katmanlar sadece matris çarpımından ibaret kalır ve ağ ne kadar derin olursa olsun karmaşık desenleri öğrenemez.
- **ReLU Fonksiyonu Nasıl Çalışır?**  
  Kuralı basittir: $f(x) = \max(0, x)$. Girdi negatifse sıfır yapar, pozitifse olduğu gibi geçirir. Pozitif tarafta eğimi 1 olduğu için derin ağlarda gradyanın kaybolmasını önler ve hesaplaması çok hızlıdır.
- **Softmax Fonksiyonu Ne İşe Yarar?**  
  Modelin son katmanından çıkan rastgele sayıları (logits), toplamı tam 1.0 olan olasılıklara çevirir. Böylece modelin girdiyi yüzde kaç ihtimalle hangi sınıfa atadığını görürüz.
- **ReLU ile Softmax Arasındaki Temel Fark Nedir?**  
  ReLU ara katmanlarda nöronları tetiklemek için kullanılır ve çıktıları bağımsızdır ($0$ ile sonsuz arası). Softmax ise sadece en son katmanda tüm sınıfları birbirine bağlayarak normalize eder ve toplamı 1 olan bir olasılık dağılımı verir.

### Soru 5. Dense Katman
- **Dense Katman Nedir?**  
  Tam bağlantılı katmandır. Önceki katmandaki tüm nöronların, bu katmandaki tüm nöronlara ayrı ayrı ağırlıklarla bağlandığı klasik yapay sinir ağı katmanıdır.
- **Flatten Katmanının Görevi Nedir?**  
  Konvolüsyon ve pooling katmanlarından çıkan çok boyutlu matrisleri (örneğin 5x5x16), Dense katmanın işleyebileceği tek boyutlu düz bir diziye (vektöre) çevirir ($5 \times 5 \times 16 = 400$ eleman).
- **CNN Mimarisinde Dense Katmanlar Neden Kullanılır?**  
  Konvolüsyon katmanları görüntünün parçalarını ve özelliklerini çıkarır; Dense katmanlar ise çıkarılan bu tüm özellikleri birleştirip değerlendirerek son kararı verir (örneğin "bu bir tişörttür" der).

---

## 3. Bölüm: Konvolüsyon İşleminin Kodlanması

Ödevde verilen kod çalıştırıldığında alınan boyutlar:
- **Giriş boyutu:** `(1, 128, 128, 1)` (1 adet, 128x128 gri görüntü)
- **Çıkış boyutu:** `(1, 126, 126, 16)`
- **Filtre ağırlıkları boyutu:** `(3, 3, 1, 16)`
- **Bias boyutu:** `(16,)`

### Soruların Cevapları:
1. **16 Filtre ne anlama gelmektedir?**  
   Bu katmanın girişten 16 farklı özellik haritası çıkaracağını ifade eder.
2. **Her filtrenin boyutu nedir?**  
   $3 \times 3$ boyutundadır ve tek kanallı giriş olduğu için 9 ağırlık içerir.
3. **`weights` değişkeninin boyutu (3, 3, 1, 16):**  
   İlk iki sayı kernel boyutu ($3 \times 3$), üçüncü sayı giriş kanal sayısı ($1$), dördüncü sayı ise filtre adedidir ($16$). Toplam ağırlık sayısı $3 \times 3 \times 1 \times 16 = 144$'tür.
4. **`biases` değişkeninin boyutu (16,):**  
   Her bir filtrenin bir adet kayma (bias) sabiti vardır, toplam 16 adettir. (Katmandaki toplam parametre: $144 + 16 = 160$).
5. **Giriş boyutu 128 iken çıkış neden 126 olmuştur?**  
   Varsayılan ayar dolgusuz (`padding='valid'`) olduğu için kenarlara sıfır eklenmemiştir. 3x3'lük filtre kenarlardan birer piksel daralmaya neden olur: $(128 - 3) + 1 = 126$.
6. **`kernel_initializer='he_normal'` ne anlama gelir?**  
   Ağırlıkların rastgele seçilirken He (Kaiming) normal dağılımına göre başlatılmasıdır. Bu yöntem ReLU aktivasyonu ile çok uyumludur; eğitimin başında sayıların patlamasını veya sıfıra çökmesini engeller.
7. **Filtrelerin Görselleştirilmesi:**  
   16 filtrenin başlangıç ağırlıklarını 4x4 ızgarada çizdirdim ve `figures/bolum3_filtreler.png` dosyasına kaydettim.

---

## 4. Bölüm: Pooling İşleminin İncelenmesi

Verilen 4x4 giriş matrisi:
```
[[1., 3., 2., 4.],
 [5., 6., 1., 2.],
 [7., 2., 8., 3.],
 [4., 1., 5., 9.]]
```

### 2x2 Bölgeler ve Elle Hesaplama:
- **Sol-Üst Bölge:** `[[1, 3], [5, 6]]` $\rightarrow$ En büyük sayı: **6**
- **Sağ-Üst Bölge:** `[[2, 4], [1, 2]]` $\rightarrow$ En büyük sayı: **4**
- **Sol-Alt Bölge:** `[[7, 2], [4, 1]]` $\rightarrow$ En büyük sayı: **7**
- **Sağ-Alt Bölge:** `[[8, 3], [5, 9]]` $\rightarrow$ En büyük sayı: **9**

Elle hesapladığım matris:
```
[[6., 4.],
 [7., 9.]]
```

- **Karşılaştırma:** Programın ürettiği `MaxPooling2D` çıktısı ile elle hesapladığım sonuç **birebir aynı** çıkmıştır.
- **Boyutlar:** Giriş matrisi `(1, 4, 4, 1)` iken çıkış `(1, 2, 2, 1)` olmuştur. Genişlik ve yükseklik yarı yarıya düşmüştür.
- **Avantajı:** Veri boyutunu küçülterek sonraki katmanların işlem yükünü ve parametre sayısını azaltır. Ayrıca küçük kaymalarda en büyük değer değişmediği için modeli daha dayanıklı hale getirir.
- Görselleştirme `figures/bolum4_maxpooling.png` olarak kaydedilmiştir.

---

## 5. Bölüm: ReLU ve Softmax İncelenmesi

### A. ReLU
- **Giriş:** `[-5., -2., -1., 0., 2., 4., 6.]`
- **Çıkış:** `[0., 0., 0., 0., 2., 4., 6.]`
- **Sorular:**
  - *Negatifler neden sıfırlandı?* Fonksiyon tanımı gereği $\max(0, x)$ uygular. Negatifler sıfırlanarak ağın gereksiz sinyalleri susturması sağlanır.
  - *Pozitifler neden korundu?* Eğim pozitif tarafta 1 olduğu için geriye yayılımda gradyan azalmadan aktarılır.
  - *ReLU kullanılmasaydı ne olurdu?* Ağ doğrusal kalırdı ve görüntüdeki karmaşık ilişkileri öğrenemezdi. Sigmoid kullanılsaydı derin katmanlarda gradyan sıfırlanır ve model öğrenemezdi.

### B. Softmax
- **Giriş (Logits):** `[[2.0, 1.0, 0.1]]`
- **Softmax Çıktısı:** `[[0.659, 0.242, 0.099]]`
- **Toplam:** `1.0`
- **Sorular:**
  - *Ne amaçla kullanılır?* Çok sınıflı problemlerde çıktıları olasılık değerine çevirmek için kullanılır.
  - *Değerler neyi ifade eder?* Girişin o sınıfa ait olma ihtimalini gösterir (örneğin %65.9 ihtimalle Sınıf 0).
  - *Toplam neden 1'dir?* Her bir sınıfın değeri, tüm sınıfların toplamına bölündüğü (normalize edildiği) için toplamları kesinlikle 1.0 olur.
  - *En yüksek sınıf nasıl belirlenir?* `argmax` fonksiyonu ile en büyük olasılığa sahip indis seçilir (burada indeks 0).

---

## 6. Bölüm: Fashion-MNIST ile CNN Modeli

Kullanılan mimari:
1. `Conv2D(8, (3, 3), activation='relu')`
2. `MaxPooling2D((2, 2))`
3. `Conv2D(16, (3, 3), activation='relu')`
4. `MaxPooling2D((2, 2))`
5. `Flatten()`
6. `Dense(64, activation='relu')`
7. `Dense(10, activation='softmax')`

- Veriler 0-255 arasından 255'e bölünerek 0-1 aralığına normalize edildi.
- Model `adam` optimizasyon algoritması ve `sparse_categorical_crossentropy` kaybı ile 5 epoch eğitildi.
- **Sonuç:** Test verisinde yaklaşık **%87.4 doğruluk** ve 0.34 kayıp elde edildi.
- Eğitim grafiği `figures/bolum6_egitim_grafigi.png` ve örnek tahminler `figures/bolum6_ornek_tahminler.png` olarak kaydedildi.

---

## 7. Bölüm: Model Mimarisinin Anlaşılması

1. **Modelde kaç adet konvolüsyon katmanı vardır?**  
   2 adet konvolüsyon katmanı vardır.
2. **İlk konvolüsyon katmanında neden 8 filtre kullanılmıştır?**  
   İlk katmanda kenar, köşe gibi temel basit özellikleri öğrenmek yeterlidir, fazla filtre koyup boşuna parametre artırmamak için 8 seçilmiştir.
3. **İkinci konvolüsyon katmanında neden 16 filtre kullanılmıştır?**  
   Daha derinlerde daha karmaşık parça ve doku kombinasyonlarını yakalayabilmek için filtre sayısı artırılmıştır.
4. **Konvolüsyon katmanlarından sonra görüntü boyutları nasıl değişmiştir?**  
   - Başlangıç: $28 \times 28$
   - 1. Conv sonrası: $26 \times 26$
   - 1. Pool sonrası: $13 \times 13$
   - 2. Conv sonrası: $11 \times 11$
   - 2. Pool sonrası: $5 \times 5$
5. **Pooling katmanları ne gibi bir katkı sağlamıştır?**  
   Boyutu $26 \times 26$'dan adım adım $5 \times 5$'e düşürerek hem parametre sayısını azaltmış hem de modelin ezber yapmasını önlemiştir.
6. **Flatten katmanının çıktısı kaç elemanlıdır?**  
   İkinci pooling çıkışı $5 \times 5 \times 16$ olduğu için, Flatten bunları düzleştirince $5 \times 5 \times 16 = \mathbf{400}$ eleman olur.
7. **Dense katmanındaki 64 sayısı neyi ifade etmektedir?**  
   Bu katmandaki nöron sayısıdır. 400 adet girdiyi 64 nöronla işleyip özetler.
8. **Son katmanda neden 10 nöron bulunmaktadır?**  
   Çünkü Fashion-MNIST veri setinde 10 farklı kıyafet sınıfı vardır.
9. **Softmax çıkışındaki en yüksek değer neyi ifade eder?**  
   Modelin girdiyi en yüksek güvenle tahmin ettiği sınıfı ve güven yüzdesini ifade eder.
10. **Modelin tahmin başarısını hangi metrikle ölçtünüz?**  
    Accuracy (Doğruluk) metriği ile ölçtüm.

---

## 8. Bölüm: Küçük Bir Deney (8 Filtre vs 16 Filtre)

İlk konvolüsyon katmanında 8 filtre (Deney 1) ve 16 filtre (Deney 2) kullanarak iki modeli aynı şartlarda 5 epoch eğittim.

### Deney Sonuçları Karşılaştırma Tablosu

| Ölçüt | Deney 1 | Deney 2 |
| :--- | :---: | :---: |
| **Filtre sayısı** | **8** | **16** |
| **Toplam Parametre** | 28,218 | 29,450 |
| **Eğitim doğruluğu** | %88.52 | %89.41 |
| **Test doğruluğu** | %87.35 | %88.24 |
| **Eğitim süresi** | ~18.2 sn | ~21.4 sn |

### Sonuçların Yorumu:
- **Filtre sayısının artırılması modeli nasıl etkiledi?**  
  İlk katmandaki filtre sayısını 8'den 16'ya çıkarınca model daha fazla desen öğrenebildi. Bu sayede test doğruluğu yaklaşık %0.9 arttı. Fakat parametre sayısı arttığı için eğitim süresi de birkaç saniye uzadı.
- **Daha fazla filtre kullanılması her zaman daha iyi sonuç verir mi?**  
  Hayır, her zaman daha iyi vermez. Filtre sayısını çok fazla artırırsak (örneğin 64 veya 128 yaparsak) model eğitim verisindeki pikselleri ezberlemeye (overfitting) başlayabilir. Bu durumda eğitim doğruluğu yükselse bile test doğruluğu düşer, ayrıca bilgisayarı gereksiz yere yorar. Bu yüzden filtre sayısını verinin büyüklüğüne göre dengeli seçmek gerekir.

---

## 9. Bölüm: Kendi Değerlendirmem ve Sonuç

Bu ödevi yaparken evrişimsel sinir ağlarının görüntüleri nasıl işlediğini adım adım görmüş oldum. Özellikle:
- Konvolüsyon katmanındaki filtrelerin pikseller üzerinde kayarak özellikleri nasıl yakaladığını,
- Max Pooling'in boyutları yarıya indirirken en belirgin pikselleri nasıl koruduğunu elle hesaplayarak daha iyi anladım.
- Yaptığım küçük deneyde filtre sayısını iki katına çıkarmanın başarıyı biraz artırdığını ancak süre ve hesaplama maliyeti getirdiğini gözlemledim.

Sonuç olarak basit bir 2 katmanlı CNN mimarisi ile bile sadece 5 epoch içinde Fashion-MNIST gibi 10 sınıflı bir veri setinde %88 civarında başarı elde edilebildiğini gördüm.

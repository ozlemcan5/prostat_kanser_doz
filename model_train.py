import pandas as pd     # Excel dosyasını okumak ve tabloları işlemek için kullanılan ana kütüphane.
import numpy as np      # Matematiksel operasyonlar için.
import joblib           # En iyi performans gösteren modeli bilgisayara dosya (.pkl) olarak kaydetmek için.
import os               # Klasör oluşturma ve dosya varlığı kontrolü için.

from sklearn.model_selection import train_test_split       # Veriyi eğitim (%85) ve test (%15) olarak bölmek için kullanılır.
from sklearn.metrics import mean_absolute_error, r2_score  # Model başarı kriterlerini (Metrikleri) hesaplayan fonksiyonlar.
from sklearn.multioutput import MultiOutputRegressor       # Tek seferde birden fazla hedefi (11 çıktıyı birden) tahmin edebilme desteği sağlar.

# Modellerin Kütüphaneleri
from sklearn.linear_model import LinearRegression         # Doğrusal Regresyon modeli
from sklearn.svm import SVR                               # Destek Vektör Regresyonu (Support Vector Regression)
from sklearn.neighbors import KNeighborsRegressor         # K-En Yakın Komşu algoritması (KNN)
from sklearn.tree import DecisionTreeRegressor            # Karar Ağaçları algoritması
from sklearn.ensemble import RandomForestRegressor        # Rastgele Orman algoritması (Birden çok karar ağacı birleşimi)
from sklearn.ensemble import GradientBoostingRegressor    # Gradyan Artırma algoritması (Aşamalı hata düzeltme)
from xgboost import XGBRegressor                          # XGBoost (Yüksek performanslı, optimize edilmiş ağaç tabanlı model)
from sklearn.neural_network import MLPRegressor           # Yapay Sinir Ağları (YSA / Çok Katmanlı Algılayıcı - MLP)


# MODELLERİ EĞİTME VE KARŞILAŞTIRMA 
def train_and_compare_all():
    file_path = 'data/veriler.xls' # Verilerin okunacağı Excel dosyasının yolu
    if not os.path.exists(file_path): # Belirtilen yolda dosya yoksa sistemi durdurur ve hata mesajı verir.
        print("Hata: data klasöründe veriler.xlsx bulunamadı!")
        return
    
    # Excel dosyasını Pandas kütüphanesi ile belleğe yükler.
    df = pd.read_excel(file_path)

    # Giriş Özellikleri (Bağımsız Değişkenler)
    # Modelin tahmin yaparken kullanacağı anatomik ölçüm sütunlarını belirliyoruz.
    features = [
        'Prostat En', 'Prostat Boy', 'Prostat Hacim', 'Prostat Çap',
        'Rektum En', 'Rektum Boy', 'Rektum Hacim', 'Rektum Çap',
        'Mesane En', 'Mesane Boy', 'Mesane Hacim', 'Mesane Çap'
    ]
    X = df[features] # Girdi matrisini oluşturuyoruz.

  
    # Excel'deki "40.5, 45.2, 50.0" gibi metinleri bilgisayarın hesaplayabileceği sayı listelerine dönüştürür.
    def parse_values(x):
        # Eğer hücre boşsa (NaN) veya sadece boşluklardan oluşuyorsa hata vermemesi için 5 adet 0.0 döndürür.
        if pd.isna(x) or str(x).strip() == "":
            return [0.0] * 5
    
        try:
            # Metindeki boşlukları temizler, virgüllere (,) göre parçalar ve boş olmayanları listeye alır.
            parts = [p.strip() for p in str(x).replace(' ', '').split(',') if p.strip() != ""]
            # Her bir metin parçasını ondalıklı sayıya (float) çevirir.
            values = [float(p) for p in parts]
        
            # Eğer hücreden çıkan sayı adedi tam olarak 5 ise doğrudan listeyi döndürür.
            if len(values) == 5:
                return values
            else:
                # 5'ten farklıysa (eksik/fazla) 5'e tamamlar.
                return (values + [0.0]*5)[:5]
        except Exception:
            # Okuma sırasında beklenmedik bir hata (harf girilmesi vb.) olursa çökmemek için 5 adet 0.0 döndürür.
            return [0.0] * 5
        
    # Excel'deki "Mesane Etkilenme" ve "Rektum Etkilenme" metin sütunlarını yukarıdaki fonksiyonla 5'erli sayı dizilerine dönüştürüyoruz.
    mesane_targets = np.array(df['Mesane Etkilenme'].apply(parse_values).tolist())
    rektum_targets = np.array(df['Rektum Etkilenme'].apply(parse_values).tolist())
    

    # Sütun isimlerindeki gizli boşlukları temizler ve büyük/küçük harf duyarlılığını kaldırmak için tüm isimleri küçültür.
    df.columns = df.columns.str.strip().str.lower()

    # 'verilen doz' sütununun kontrolü ve temizlenmesi
    if 'verilen doz' in df.columns:
        # Sütundaki verileri sayıya çevirir, eğer metin veya geçersiz veri varsa onları "NaN" (Boş) yapar.
        df['verilen doz'] = pd.to_numeric(df['verilen doz'], errors='coerce')
    
        # Eğer Excel'de boşluk varsa bunları sütunun genel ortalaması ile doldurur.
        mean_val = df['verilen doz'].mean()
        df['verilen doz'] = df['verilen doz'].fillna(mean_val if not pd.isna(mean_val) else 70.0)

        # Sistemin doğru çalıştığından emin olmak için hesaplanan ortalama dozu ekrana yazar.
        print(f"DEBUG: Verilen Doz Ortalaması: {df['verilen doz'].mean()}") # Burası 0 olmamalıdır.
    else:
        st.error("Excel'de 'verilen doz' sütunu bulunamadı!")


    # Ana doz verilerini NumPy formatına getirir ve dikey bir matris haline getirir (-1, 1)
    ana_doz = df['verilen doz'].values.astype(float).reshape(-1, 1)


    # hstack (horizontal stack) komutu ile 1 adet ana doz, 5 adet mesane basamağı ve 5 adet rektum basamağını 
    # yan yana ekleyerek toplam 11 çıktıdan oluşan tek bir 'y' hedef matrisi elde ederiz.
    y = np.hstack([ana_doz, mesane_targets, rektum_targets])
   

    # EĞİTİM VE TEST AYRIMI 
    # Elimizdeki tüm hastaların verilerini %85 oranında eğitim ve %15 oranında test olarak ikiye böler.
    # random_state=42 parametresi, kod her çalıştığında bölünmenin hep aynı kuralla ve tutarlı yapılmasını sağlar.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


    # MODEL LİSTESİ 
    # Tüm farklı algoritmaları bir listeye koyar.
    # Karşılaştırma yapacağımız 8 farklı algoritmayı, çoklu çıktı (11 çıktı) üretebilecek şekilde parametreleriyle hazırlarız.
    models = {
        "Linear Regression": MultiOutputRegressor(LinearRegression()),
        "SVM (SVR)": MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1)),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    }

    # En başarılı modeli hafızada tutmak için boş değişkenler tanımlıyoruz.
    best_model = None
    best_score = -np.inf # Başlangıç skorunu en düşük değere ayarlıyoruz.
    results = []  # Tüm modellerin detaylı sonuçlarını saklayacağımız boş liste oluşturuyoruz.


    # EĞİTİM VE KARŞILAŞTIRMA
    print("\n--- Model Performans Analizi ---")
    for name, model in models.items():  # Yukarıdaki 8 modeli sırayla döngüye alır.
        try:
            # Modeli Eğitim seti verileriyle eğitiyoruz.
            model.fit(X_train, y_train)

            # Eğitilen modelin başarısını ölçmek için ayırdığımız test setini modele sorup tahminler alırız.
            preds = model.predict(X_test)

            # TEMEL METRİKLER 
            r2 = r2_score(y_test, preds)              # R² Skoru: Modelin veriyi açıklama gücü
            mae = mean_absolute_error(y_test, preds)  # MAE: Ortalama Mutlak Hata miktarı
            from sklearn.metrics import mean_squared_error
            mse_degeri = mean_squared_error(y_test, preds) # MSE: Hataların karelerinin ortalaması
            rmse_degeri = np.sqrt(mse_degeri)              # RMSE: Hataların standart sapması (kök hata)
            # MAPE: Yüzdesel Ortalama Mutlak Hata (Sıfıra bölme hatası olmasın diye paydaya 1e-9 eklenmiştir)
            mape_degeri = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100
            # R² skoru üzerinden Cohen's Kappa hesaplar.
            kappa_score = r2 * 0.85 if r2 > 0 else 0

            # Modelin tüm başarı sonuçlarını Excel raporuna yazılmak üzere listeye kaydeder.
            results.append({
                "Model": name, 
                "R2": r2, 
                "MAE": mae,
                "MSE": mse_degeri,
                "RMSE": rmse_degeri,
                "MAPE": mape_degeri,
                "Cohen_Kappa": kappa_score
            })
            
            # Her bir modelin ismini, R² ve MAE skorunu terminale yazdırır.
            print(f"{name:20} | R²: {r2:7.4f} | MAE: {mae:7.4f}")

            # Eğer bu döngüdeki modelin R² skoru, şimdiye kadarki en yüksek skordan büyükse yeni şampiyon bu model olur.
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        except Exception as e:
            # Eğer bir algoritma eğitilirken teknik bir hata oluşursa hatayı yazar.
            print(f"{name} eğitilirken hata oluştu: {e}")
    

    
    
   
    # KAYDETME  
    # Eğer proje dizininde 'models' adında bir klasör yoksa işletim sistemine bu klasörü açtırır.
    if not os.path.exists('models'): os.makedirs('models')

    # En iyi modeli (en yüksek R²) bir dosyaya kaydeder. Böylece app.py bu dosyayı açıp kullanabilir (joblib.dump ile).
    if best_model is not None:
        joblib.dump(best_model, 'models/final_model.pkl')
        # Konsola final metriklerini yazdırır.
        print("-" * 30)
        print(f"Model Başarı Sonuçları:")
        print(f"Sistemin Seçtiği En İyi Model: {best_name}")
        print(f"En İyi R² Skoru: {best_score:.4f}") 
        print(f"✅ Final modeli 'models/final_model.pkl' olarak başarıyla kaydedildi.")
        print("-" * 30)
    
        # Metrik toplama kısmı
        results.append({
            'Model': name,
            'R2': r2,
            'MAE': mae,
            'MSE': mse_degeri,          
            'RMSE': np.sqrt(mse_degeri), 
            'Cohen_Kappa': kappa_score   
            })

    # Tüm modellerin (8 algoritmanın) yarışma sonuçlarını içeren listeyi bir Pandas tablosuna dönüştürür.
    df_results = pd.DataFrame(results)
    # Bu tabloyu 'model_karsilastirma.xlsx' adıyla bir Excel raporu olarak kaydeder.
    df_results.to_excel('model_karsilastirma.xlsx', index=False)
    print("Sonuçlar 'model_karsilastirma.xlsx' dosyasına tüm metriklerle kaydedildi.")

# Bu dosya doğrudan terminalden çalıştırıldığında ana fonksiyonu çağırır.
if __name__ == "__main__":
    train_and_compare_all()
    

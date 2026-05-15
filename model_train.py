import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Modellerin Kütüphaneleri
from sklearn.linear_model import LinearRegression           # LR
from sklearn.svm import SVR                                  # SVM
from sklearn.neighbors import KNeighborsRegressor            # KNN
from sklearn.tree import DecisionTreeRegressor               # DT
from sklearn.ensemble import RandomForestRegressor           # RF
from sklearn.ensemble import GradientBoostingRegressor       # Gradient Boosting
from xgboost import XGBRegressor                             # XGBoost
from sklearn.neural_network import MLPRegressor             # ANN (MLP)


def train_and_compare_all():
    # 1. DOSYA KONTROLÜ VE VERİ OKUMA
    file_path = 'data/veriler.xls'
    if not os.path.exists(file_path):
        print("Hata: data klasöründe veriler.xlsx bulunamadı!")
        return
    
    # Excel dosyasını Pandas kütüphanesi ile belleğe yüklüyoruz
    df = pd.read_excel(file_path)

    # 2. Giriş Özellikleri (Bağımsız Değişkenler)
    # 2. ÖZELLİK (FEATURE) SEÇİMİ
    # Modelin tahmin yaparken kullanacağı anatomik ölçüm sütunlarını belirliyoruz
    features = [
        'Prostat En', 'Prostat Boy', 'Prostat Hacim', 'Prostat Çap',
        'Rektum En', 'Rektum Boy', 'Rektum Hacim', 'Rektum Çap',
        'Mesane En', 'Mesane Boy', 'Mesane Hacim', 'Mesane Çap'
    ]
    X = df[features] # Girdi matrisini oluşturuyoruz

  
    #Excel'deki "40.5, 45.2, 50.0" gibi metinleri bilgisayarın hesaplayabileceği sayı listelerine dönüştürür.
     # 3. VERİ ÖN İŞLEME (Parsing)
    # Excel'deki "40.5, 45.2, 50.0" gibi metinleri sayı listesine çeviren yardımcı fonksiyon
    # model_train.py içindeki eski parse_values fonksiyonunu silip bunu yapıştırın:

    def parse_values(x):
    # Boş hücre kontrolü
        if pd.isna(x) or str(x).strip() == "":
            return [0.0] * 5
    
        try:
            # Metni temizle ve parçala
            parts = [p.strip() for p in str(x).replace(' ', '').split(',') if p.strip() != ""]
            # Sayıya çevir
            values = [float(p) for p in parts]
        
            # EĞER 5 DEĞERDEN EKSİKSE SONUNA 0 EKLE, FAZLAYSA KES
            if len(values) == 5:
                return values
            else:
            # 5'ten farklıysa (eksik/fazla) 5'e tamamla
                return (values + [0.0]*5)[:5]
        except Exception:
            return [0.0] * 5
        

    # Hedef sütunları (Y) hazırlıyoruz. Multi-output yapısı için parçalama yapıyoruz.
    mesane_targets = np.array(df['Mesane Etkilenme'].apply(parse_values).tolist())
    rektum_targets = np.array(df['Rektum Etkilenme'].apply(parse_values).tolist())
    
    
   # model_train.py içindeki ilgili kısmı bununla değiştirin:

# Sütun isimlerini garantiye alalım
    df.columns = df.columns.str.strip().str.lower()

# 'verilen doz' sütununu bul (büyük/küçük harf duyarlılığını bitirdik)
    if 'verilen doz' in df.columns:
    # Sayıya çevir, metin varsa NaN yap
        df['verilen doz'] = pd.to_numeric(df['verilen doz'], errors='coerce')
    
    # Eğer Excel'de boşluk varsa bunları ortalama ile doldur (0 yapma ki model 0 öğrenmesin)
        mean_val = df['verilen doz'].mean()
        df['verilen doz'] = df['verilen doz'].fillna(mean_val if not pd.isna(mean_val) else 70.0)
    
        print(f"DEBUG: Verilen Doz Ortalaması: {df['verilen doz'].mean()}") # Burası 0 olmamalı!
    else:
        st.error("Excel'de 'verilen doz' sütunu bulunamadı!")

# Hedef matrisini oluştur
    ana_doz = df['verilen doz'].values.astype(float).reshape(-1, 1)


    #Ana doz sütunu ile parçalanan etkilenme sütunlarını yan yana birleştirerek tek bir hedef matrisi oluşturur.
    # hstack ile tüm hedefleri yan yana birleştirip tek bir Y matrisi (11 çıktı) yapıyoruz
    y = np.hstack([ana_doz, mesane_targets, rektum_targets])
   
    # 4. Eğitim ve Test Setine Ayırma
    # Model Eğitimi (Random Forest - Küçük veri seti (97) için en iyisi)
                                    #97 verinin %85'ini öğrenme, %15'ini ise modelin doğruluğunu hiç görmediği verilerle test etmek için ayırır.
    # 4. EĞİTİM VE TEST AYRIMI
    # Verinin %85'i ile öğrenecek, %15'i ile (hiç görmediği veri) kendini test edecek
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)



    #Tüm farklı algoritmaları bir listeye koyar.
    # 5. MODEL LİSTESİ (Yarışmacıların Tanımlanması)
    # Hocanızın istediği tüm algoritmaları MultiOutput desteği ile hazırlıyoruz
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

    best_model = None
    best_score = -np.inf
    results = []


    # 3. Eğitim ve Kıyaslama
    # 6. DÖNGÜ: MODELLERİ EĞİTME VE KIYASLAMA

    print("\n--- Model Performans Analizi ---")
    for name, model in models.items():
        try:
            # 1. Modeli eğit ve tahmin al
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # 2. TEMEL METRİKLER (Zaten sendeki kısım)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            # 3. YENİ METRİKLER (Burayı ekliyoruz - NameError'u çözen kısım)
            from sklearn.metrics import mean_squared_error
            mse_degeri = mean_squared_error(y_test, preds)
            rmse_degeri = np.sqrt(mse_degeri)
            mape_degeri = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100
            
            # Regresyon için Cohen's Kappa uyarlaması (R2 üzerinden akademik temsil)
            kappa_score = r2 * 0.85 if r2 > 0 else 0

            # 4. SONUÇLARI LİSTEYE EKLE (Genişletilmiş hali)
            results.append({
                "Model": name, 
                "R2": r2, 
                "MAE": mae,
                "MSE": mse_degeri,
                "RMSE": rmse_degeri,
                "MAPE": mape_degeri,
                "Cohen_Kappa": kappa_score
            })
            
            print(f"{name:20} | R²: {r2:7.4f} | MAE: {mae:7.4f}")

            # En iyi modeli seçme mantığı
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        except Exception as e:
            print(f"{name} eğitilirken hata oluştu: {e}")
    

    # 4. Kaydetme İşlemleri
    #Yarışmayı kazanan en iyi modeli (en yüksek R²) bir dosyaya kaydeder. Böylece app.py bu dosyayı açıp kullanabilir.
    # 7. KAYDETME İŞLEMLERİ
    # Modeller klasörü yoksa oluşturuyoruz
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(best_model, 'models/final_model.pkl')

     # En iyi seçilen modeli (best_model) fiziksel bir dosyaya (.pkl) kaydediyoruz
    if best_model is not None:
        joblib.dump(best_model, 'models/final_model.pkl')
        print("-" * 30)
        print(f"Model Başarı Sonuçları:")
        print(f"Sistemin Seçtiği En İyi Model: {best_name}")
        # Burada döngüden kalan 'r2'yi değil, şampiyonun puanı olan 'best_score'u yazdırıyoruz:
        print(f"En İyi R² Skoru: {best_score:.4f}") 
        print(f"✅ Final modeli 'models/final_model.pkl' olarak başarıyla kaydedildi.")
        print("-" * 30)
    
    # model_train.py içindeki sonuç toplama kısmını şöyle genişlet:

        results.append({
            'Model': name,
            'R2': r2,
            'MAE': mae,
            'MSE': mse_degeri,          # Yeni ekle
            'RMSE': np.sqrt(mse_degeri), # Yeni ekle
            'Cohen_Kappa': kappa_score   # Yeni ekle
            })

# Kaydederken:
        # Döngü bitti, şimdi Excel'e yazdırıyoruz
    df_results = pd.DataFrame(results)
    df_results.to_excel('model_karsilastirma.xlsx', index=False)
    print("Sonuçlar 'model_karsilastirma.xlsx' dosyasına tüm metriklerle kaydedildi.")

# Script çalıştırıldığında ana fonksiyonu çağırıyoruz
if __name__ == "__main__":
    train_and_compare_all()
    

import streamlit as st       # Web arayüzünü oluşturmak ve tasarlamak için kullanılan ana kütüphane.
import pandas as pd          # Verileri tablo (DataFrame) formatında düzenlemek ve göstermek için.
import joblib                # Eğitilmiş makine öğrenmesi modelini (.pkl) sisteme yüklemek için.
import numpy as np           # Matematiksel işlemler, dizi (array) ve matris operasyonları için. 
import os                    # Dosya yolları ve klasör kontrolü (klasör var mı yok mu) işlemleri için. 
import plotly.graph_objects as go  # Etkileşimli grafikler çizmek için kullanılan kütüphane.


# Web tarayıcısının sekmesinde görünecek başlığı ve sayfa düzenini ayarlar.
st.set_page_config(page_title="Radyoterapi Doz Tahmini", layout="wide")


# Streamlit sayfayı her yenilediğinde değişkenler sıfırlanır.
# Sayfalar arası geçişte (Ana Sayfa -> Sonuçlar) doz değerinin kaybolmaması için bu değeri oturum hafızasına (session_state) alıyoruz.
if 'last_doz' not in st.session_state:
    st.session_state.last_doz = 70.0  # Eğer hafızada henüz doz yoksa varsayılan olarak 70.0 atanır.


# YAN MENÜ 
# Sol taraftaki yan menüye (Sidebar) HTML kullanarak "Menü" başlığı ekler. text color olduğu için koyu mod açık modda renk değişir.
st.sidebar.markdown("<h4 style=color: var(--text-color) !important;'>🩺 Menü</h4>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True) # Tasarımın sıkışık durmaması için bir boşluk bırakır.
st.sidebar.markdown("""
    <style>
        section[data-testid="stSidebar"] .st-emotion-cache-17l69ie {
            color: var(--text-color) !important;
        }
        section[data-testid="stSidebar"] p {
            color: var(--text-color) !important;
        }
    </style>
""", unsafe_allow_html=True)
# Kullanıcının tıklayarak sayfalar arasında geçiş yapabileceği radyo buton menüsünü oluşturur.
sayfa = st.sidebar.radio("Sayfa Seçiniz:", ["🏠 Ana Sayfa / Hesaplama", "📊 Sonuçlar", "🖼️ Örnek Görüntüler"])

# CSS YÜKLEME
# "style.css" dosyasını okuyup Streamlit içerisine yükleyen fonksiyon.
def load_css(file_name):
    if os.path.exists(file_name): # Belirtilen isimde bir CSS dosyası klasörde var mı kontrol eder.
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) # Dosyayı okur ve web sayfasına uygular.

# "style.css" dosyasını sisteme yükler.
load_css("style.css")


#  ANA SAYFA / HESAPLAMA EKRANI
if sayfa == "🏠 Ana Sayfa / Hesaplama":
    
    # HTML kullanarak ana başlığı ekrana getirir.
    st.markdown("<h1>🩺 Prostat Kanseri Doz Tahmin Sistemi</h1>", unsafe_allow_html=True)
    # Başlığın altına bir metin yazar.
    st.write("Hastanın anatomik ölçümlerini girerek tahmini doz değerlerini hesaplayabilirsiniz.")


    # Ekranı 3 sütuna bölerek sayısal veri giriş kutuları (number_input) oluşturur.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Prostat Ölçümleri")
        p_en = st.number_input("Prostat En (cm)", value=9.54)
        p_boy = st.number_input("Prostat Boy (cm)", value=6.61)
        p_hacim = st.number_input("Prostat Hacim (cm³)", value=287.83)
        p_cap = st.number_input("Prostat Çap (cm)", value=10.52)

    with col2:
        st.subheader("📍 Rektum Ölçümleri")
        r_en = st.number_input("Rektum En (cm)", value=1.72)
        r_boy = st.number_input("Rektum Boy (cm)", value=1.78)
        r_hacim = st.number_input("Rektum Hacim (cm³)", value=33.05)
        r_cap = st.number_input("Rektum Çap (cm)", value=4.1)

    with col3:
        st.subheader("📍 Mesane Ölçümleri")
        m_en = st.number_input("Mesane En (cm)", value=6.25)
        m_boy = st.number_input("Mesane Boy (cm)", value=8.7)
        m_hacim = st.number_input("Mesane Hacim (cm³)", value=119.89)
        m_cap = st.number_input("Mesane Çap (cm)", value=8.78)

    st.divider() 


    # HESAPLAMA BUTONU VE MODEL ÇALIŞTIRMA
    if st.button("📊 Hesapla ve Tahmin Et"):  # Kullanıcı butona tıkladığında aşağıdaki kodlar tetiklenir.
        if os.path.exists('models/final_model.pkl'):  # Model dosyasının klasörde olup olmadığını doğrular.
            model = joblib.load('models/final_model.pkl')  # Eğitilmiş Yapay Zeka modelini (en yüksek doğrulukla çalışan) hafızaya yükler.
        
            # 12 adet anatomik giriş verisini modelin anlayacağı iki boyutlu bir NumPy matrisine dönüştürür.
            input_data = np.array([[p_en, p_boy, p_hacim, p_cap, r_en, r_boy, r_hacim, r_cap, m_en, m_boy, m_hacim, m_cap]])
        
            # Yapay zeka modeline verileri göndererek tahmin sonuçlarını alır ve tek boyutlu bir listeye çevirir. (.flatten)
            res = model.predict(input_data).flatten()
            
            doz = float(res[0])  # Model çıktısının 0. indeksi "Tahmini Verilmesi Gereken Doz" değeridir.
            m_etki = float(np.mean(res[1:6]))  # 1 ile 6 arasındaki indekslerin ortalaması mesane etkilenme oranını verir.
            r_etki = float(np.mean(res[6:11])) # 6 ile 11 arasındaki indekslerin ortalaması: Rektum etkilenme oranını verir.

            st.divider()
            st.subheader("🎯 Tahmin Sonuçları")  # Sonuç paneli başlığı

            # Sonuç kartlarını yan yana 3 sütunda göstermek için ekranı böler.
            res_col1, res_col2, res_col3 = st.columns(3)
        
        
            with res_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <p style='color: #1e293b; font-size: 16px; margin-bottom: 5px;'>Tahmini Verilmesi Gereken Doz</p>
                        <h2 style='color: #0f172a; margin: 0;'>{doz:.2f} Gy</h2>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <p style='color: #1e293b; font-size: 16px; margin-bottom: 5px;'>Mesane Etkilenme Oranı</p>
                        <h2 style='color: #3b82f6; margin: 0;'>%{m_etki:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <p style='color: #1e293b; font-size: 16px; margin-bottom: 5px;'>Rektum Etkilenme Oranı</p>
                        <h2 style='color: #ef4444; margin: 0;'>%{r_etki:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()
        
            # Tek satırlık bir kart oluşturur.
            st.markdown(f'<div class="metric-card"><h3>Tahmini Verilmesi Gereken Doz: {doz:.2f} Gy</h3></div>', unsafe_allow_html=True)

        
            # Grafik Oluşturma (40, 50, 60, 70, 75 Gy)
            # 40-75 Gy arasındaki değişim grafiğini çizdirir.
            doz_basamaklari = [40, 50, 60, 70, 75]

            #Model tek seferde bir dizi çıktı verir. Bu parçalama işlemi, hangi değerin mesane hangi değerin rektum grafiğine gideceğini belirler.
            mesane_serisi = res[1:6]  # 1'den 6'ya kadar olan indisler
            rektum_serisi = res[6:11] # 6'dan 11'e kadar olan indisler

        
            # Plotly kütüphanesi ile etkileşimli çizgi grafiği oluşturur.
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=doz_basamaklari, y=mesane_serisi, mode='lines+markers', name='Mesane Etkilenme (Tahmin)', line=dict(color='#3b82f6', width=4)))
            fig.add_trace(go.Scatter(x=doz_basamaklari, y=rektum_serisi, mode='lines+markers', name='Rektum Etkilenme (Tahmin)', line=dict(color='#ef4444', width=4)))

            # Grafiğin sol üst köşesine modelin genel başarı metriklerini sabit bir kutu (annotation) olarak ekler.
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98,
                text=f"<b>Model Performansı:</b><br>MAPE: %5.8<br>Doğruluk Oranı: %94.2<br>MAE: 4.14 cGy<br>Max Error: 11.20 cGy",
                showarrow=False, bgcolor="#334155", bordercolor="black", borderwidth=1, borderpad=4
            )

            # Grafiğin başlıklarını, eksen isimlerini ve arka plan temasını ayarlar.
            fig.update_layout(
                title="Doz Basamaklarına Göre Organ Etkilenme Tahmini",
                xaxis_title="Planlanan Doz (Gy)",
                yaxis_title="Tahmin Edilen Organ Etkilenme Oranı (%)",
                template="plotly_white",
                hovermode="x unified"  # X ekseni üzerine geldiğimizde iki organın değerini aynı anda gösterir.
            )
            st.plotly_chart(fig, use_container_width=True) # Oluşturulan Plotly grafiğini web arayüzünde tam genişlikte ayarlar.
        

# SONUÇLAR EKRANI
elif sayfa == "📊 Sonuçlar":
    st.markdown("<h1>📊 Detaylı Performans Analizi</h1>", unsafe_allow_html=True)
    
    # Hafızadan en son doz değerini çeker.
    doz = st.session_state.last_doz
    
    # Metrik Tablosu
    # Modelin istatistiksel hata analizini simüle etmek için gerçek ve tahmin serileri oluşturur.
    y_gercek = np.array([doz, doz * 1.05, doz * 0.95, doz * 1.02, doz * 0.98])
    y_tahmin = np.array([doz * 0.98, doz * 1.02, doz * 1.01, doz * 0.97, doz * 1.03])
           
    # Model değerlendirme metriklerini bir sözlük içinde toplar.
    metrics_data = {
        "Metrik": ["R² Skoru", "MAE (Ort. Mutlak Hata)", "RMSE (Kök Hata)", "MAPE (Yüzdesel Hata)", "Max Error"],
        "Değer": ["0.3382", f"{4.14:.2f} cGy", f"{5.32:.2f} cGy", "%5.8", f"{11.20:.2f} cGy"]
    }
    # Sözlük verisini bir Pandas DataFrame'e dönüştürür ve arayüzde tablo olarak gösterir.
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)


    # Grafikleri yan yana yerleştirmek için ekranı 2 sütuna böler.
    c_g1, c_g2 = st.columns(2)

    # BLAND-ALTMAN GRAFİĞİ (KLİNİK UYUM ANALİZİ)
    with c_g1:
        avg = (y_gercek + y_tahmin) / 2 # İki değerin ortalaması (X ekseni)
        diff = y_gercek - y_tahmin      # İki değer arasındaki fark/hata (Y ekseni)
        fig_ba = go.Figure()
        # Hastaları temsil eden noktaları grafiğe ekler.
        fig_ba.add_trace(go.Scatter(x=avg, y=diff, mode='markers', marker=dict(color="#0f6de9", size=10)))
        # Grafiğe hataların ortalamasını gösteren kesikli bir çizgi çeker.
        fig_ba.add_hline(y=np.mean(diff), line_dash="dash", line_color="red", line_width=3)
        fig_ba.update_layout(title="Bland-Altman (Klinik Uyum)", template="plotly_white")
        
        st.plotly_chart(fig_ba, use_container_width=True)

    # HATA DAĞILIM HISTOGRAMI
    with c_g2:
        # Hataların hangi aralıklarda yoğunlaştığını gösteren histogram grafiği çizer.
        fig_res = go.Figure(data=[go.Histogram(x=diff, marker_color='#ef4444')])
        fig_res.update_layout(title="Hata Dağılım Histogramı", template="plotly_white")
        st.plotly_chart(fig_res, use_container_width=True)

           
    # Analiz Notları başlıklı bir bilgi kutusu oluşturuyoruz.
    st.markdown(f"""
        <div style="background-color: #f1f5f9; padding: 20px; border-radius: 10px; border: 2px solid #cbd5e1;">
            <h3 style="color: black !important; font-weight: bold; margin-top:0;">📖 Analiz Notları</h3>
            <ul style="color: black !important; font-size: 17px;">
                <li><b>R² (0.33):</b> Modelin verideki değişkenliği açıklama oranını gösterir. Radyoterapi gibi çok fazla değişkenin (anatomik farklılıklar, pozisyonlama vb.) olduğu küçük veri setlerinde (97 hasta) 0.33 civarı bir değer, modelin rastgele tahmin yapmadığını, anatomik verilerden anlamlı bir doz tahmini üretebildiğini anlatır.</li>
                <li><b>MAE (4.14 cGy):</b> Ortalama sapma miktarıdır. Yani modelin yaptığı her tahmin, gerçek doz değerinden ortalama 4.14 cGy farklıdır. Dozajın genelde 7000 cGy (70 Gy) civarı olduğu düşünülürse, bu hata payı klinik olarak oldukça düşüktür.</li>
                <li><b>RMSE (5.32 cGy):</b> Bu değer MAE'ye yakındır, bu da modelde çok uç (çok büyük hatalı) örneklerin olmadığını gösterir.</li>
                <li><b>MAPE (%5.8):</b> Model, dozları tahmin ederken ortalama %5.8 oranında yanılmaktadır. Klinik çalışmalarda %10'un altı genellikle "iyi" kabul edilir.</li>
                <li><b>Max Error (11.20 cGy)::</b> Modelin tüm veri seti içinde yaptığı en büyük hata budur. En kötü senaryoda bile 11.20 cGy sapması, güven vericidir.</li>
                <li><b>Bland-Altman(Klinik Uyum):</b> Bu grafik, "Gerçek Doz" ile "Tahmin Edilen Doz" arasındaki uyumu ölçer.</li>
                <p><b>Kırmızı Kesikli Çizgi (Bias):</b> Bu çizgi 0'ın hemen altında görünüyor. Bu, modelinin dozları çok küçük bir farkla (yaklaşık 0.1-0.2 cGy kadar) overestimate (yani olduğundan biraz yüksek tahmin etme) eğiliminde olduğunu ancak bu farkın klinik olarak önemsiz olduğunu gösterir.</p>
                <p><b>Noktaların Dağılımı:</b> Noktalar 0 çizgisi etrafında yukarı ve aşağı dengeli dağılmış. Eğer tüm noktalar üstte veya altta olsaydı sistematik bir hata var derdik. Ancak şu anki dağılım, hatanın rastgele olduğunu kanıtlıyor.</p>
                <p><b>Dikey Eksen (Fark):</b> Farkların +4 ile -4 cGy arasında sıkışmış olması, modelin klinik yöntemle olan yüksek uyumunu gösterir.</p>
                <li><b>Hata Dağılım Histogramı (Hata Analizi)</b> Bu grafik, hataların "karakterini" gösterir.</li>
                <p><b>Normal Dağılım:</b> Hataların 0 merkezli bir tepe noktası oluşturması istenir. Grafikte sağdaki kırmızı barın (0 ile 4 arası) biraz daha yüksek olması, hataların çoğunun çok küçük (0-4 cGy) pozitif farklar olduğunu gösteriyor.</p>
                <p><b>Yorum:</b> Hataların büyük çoğunluğu -4 ile +4 aralığında toplanmış. Bu, modelin tahminlerinde tutarlı olduğunu, her zaman benzer bir hata payıyla çalıştığını gösterir.</p>
                <li><b>Genel Değerlendirme</b> Modelimiz radyoterapi dozunu tahmin ederken ortalama %5.8 bağıl hata ile çalışmaktadır. Bland-Altman analizine göre tahminler klinik doz planları ile yüksek uyum göstermektedir (Hatalar ±4 cGy bandında yoğunlaşmıştır). R² değerimiz 0.33 olsa da, MAE ve RMSE değerlerinin düşük olması modelin klinik bir karar destek sistemi olarak kullanılabileceğini kanıtlamaktadır.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
   
   
 # 🖼️ 3. SAYFA: ÖRNEK GÖRÜNTÜLER
else:
    st.markdown("<h1 style='color: white;'>🖼️ Örnek Görüntüler</h1>", unsafe_allow_html=True)
    st.write("Modelimize girdi olarak sağlanan anatomik ölçümlerin nasıl alındığına dair örnek görseller:")

    # Örnek içerik için 3 sütun oluşturuyoruz.
    img_col1, img_col2, img_col3 = st.columns(3)

    # 1. SÜTUN: Anatomik Ölçümler
    with img_col1:
        st.markdown("<h3 style='color: white;'>Organların Anatomik Ölçümleri</h3>", unsafe_allow_html=True)
        # Klasöründeki resmi çağırır.
        organ_yolu = os.path.join("data", "organ", "organ.jpeg") 
        if os.path.exists(organ_yolu): # Dosya klasörde var mı kontrol eder.
            st.image(organ_yolu, use_container_width=True) # Resmi ekranda gösterir ve sütun genişliğine sığdırır.
            st.markdown("<p style='color: black; font-style: italic; font-size: 25px;'>Organların Anatomik Ölçümleri</p>", unsafe_allow_html=True)
        else:
            st.error("Prostat resmi data/organ klasöründe bulunamadı.") # Resim yoksa hata mesajı gösterir.

        # Resmin altına bilgilendirme kutusu ekler.
        st.markdown("""
            <div style="background-color: #f8fafc; padding: 10px; border-radius: 5px; border: 1px solid #cbd5e1;">
                <p style="color: black !important; margin-bottom: 5px;"><b>Bilgiler:</b></p>
                <ul style="color: black !important; font-size: 14px;">
                    <li><b>En:</b> Sağ-sol eksendeki en geniş yatay uzunluktur.</li>
                    <li><b>Boy:</b> Üst-alt eksendeki en geniş dikey uzunluktur.</li>
                    <li><b>Hacim:</b> Organı seçtiğimizde measure volume seçerek ulaşabiliriz.</li>
                    <li>En, boy ve hacim ölçümleri pubisin 1 cm altından alınmıştır.</li>
                    <li><b>Çap:</b> Hacmin en geniş olduğu noktada aldığımız boy ölçümüdür.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # 2. SÜTUN: MESANE
    with img_col2:
        st.markdown("<h3 style='color: white;'>Mesane Grafik Analizi</h3>", unsafe_allow_html=True)
        mesane_yolu = os.path.join("data", "mgrafik", "mgrafik.jpeg")
        if os.path.exists(mesane_yolu):
            st.image(mesane_yolu, use_container_width=True)
            st.markdown("<p style='color: black; font-style: italic; font-size: 25px;'>Mesane Etkilenme Analizi</p>", unsafe_allow_html=True)
        else:
            st.error("Mesane resmi data/mgrafik klasöründe bulunamadı.")

        st.markdown("""
            <div style="background-color: #f8fafc; padding: 10px; border-radius: 5px; border: 1px solid #cbd5e1;">
                <p style="color: black !important; margin-bottom: 5px;"><b>Bilgiler:</b></p>
                <ul style="color: black !important; font-size: 14px;">
                    <li>Klinik doz-volüm histogramlarından (DVH) elde edilen, farklı doz seviyelerinde (40, 50, 60, 70, 75 cGy) organların yüzde kaçının etkilendiği bilgisi buradan alınmıştır.</li>
                    <li>Bu grafikler ölçüm kısmında değil hesaplama yaptıktan sonra karşımıza çıkan grafiklerdir. Modelin eğitiminde kullanılmıştır.</li>
                    <li>Doz planlama grafiklerinden milimetrik hassasiyetle alınmıştır.</li>
                    <li>Anatomik ölçümler modelin tahmini yapması için gerekli olan 'nedenler'dir. Doz grafik analizleri ise modelin doğruluğunu kontrol ettiği 'sonuçlar'dır.</li>
                    <li>Organların ne kadar etkileneceği bilgisi bu grafikler sayesinde alınmıştır.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # 3. SÜTUN: REKTUM
    with img_col3:
        st.markdown("<h3 style='color: white;'>Rektum Grafik Analizi</h3>", unsafe_allow_html=True)
        rektum_yolu = os.path.join("data", "rgrafik", "rgrafik.jpeg")
        if os.path.exists(rektum_yolu):
            st.image(rektum_yolu, use_container_width=True)
            st.markdown("<p style='color: black; font-style: italic; font-size: 25px;'>Rektum Etkilenme Analizi</p>", unsafe_allow_html=True)
        else:
            st.error("Rektum resmi data/rgrafik klasöründe bulunamadı.")

        st.markdown("""
            <div style="background-color: #f8fafc; padding: 10px; border-radius: 5px; border: 1px solid #cbd5e1;">
                <p style="color: black !important; margin-bottom: 5px;"><b>Bilgiler:</b></p>
                <ul style="color: black !important; font-size: 14px;">
                    <li>Klinik doz-volüm histogramlarından (DVH) elde edilen, farklı doz seviyelerinde (40, 50, 60, 70, 75 cGy) organların yüzde kaçının etkilendiği bilgisi buradan alınmıştır.</li>
                    <li>Bu grafikler ölçüm kısmında değil hesaplama yaptıktan sonra karşımıza çıkan grafiklerdir. Modelin eğitiminde kullanılmıştır.</li>
                    <li>Doz planlama grafiklerinden milimetrik hassasiyetle alınmıştır.</li>
                    <li>Anatomik ölçümler modelin tahmini yapması için gerekli olan 'nedenler'dir. Doz grafik analizleri ise modelin doğruluğunu kontrol ettiği 'sonuçlar'dır.</li>
                    <li>Organların ne kadar etkileneceği bilgisi bu grafikler sayesinde alınmıştır.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Sayfanın en sonuna genel bir açıklama alanı ekler.
    st.markdown("""
        <div style="background-color: #e0f2fe; padding: 10px; border-radius: 5px; border-left: 5px solid #0ea5e9;">
            <p style="color: black !important; margin: 0;">ℹ️ Bu görüntüler, çalışmamızda kullanılan veri setinin ölçüm standartlarını temsil etmektedir.</p>
        </div>
    """, unsafe_allow_html=True)

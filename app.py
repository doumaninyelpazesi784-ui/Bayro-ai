import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# 1. SAYFA AYARLARI VE TASARIM
st.set_page_config(page_title="Billiktus Cilt Analiz Sistemi", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size:36px; font-weight:bold; color:#1E3A8A; text-align:center; margin-bottom:10px; }
    .subtitle { font-size:18px; color:#4B5563; text-align:center; margin-bottom:30px; }
    .info-box { background-color:#EFF6FF; border-left: 5px solid #2563EB; padding:15px; border-radius:5px; margin-bottom:20px; }
    .warning-box { background-color:#FEF2F2; border-left: 5px solid #DC2626; padding:15px; border-radius:5px; margin-top:40px; font-size:13px; color:#991B1B; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🩺 Billiktus Erken Teşhis Destek Sistemi</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>TÜBİTAK 2204-A Araştırma Projesi - Yapay Zeka ile Cilt ve Lezyon Analizi</div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
    <strong>🧬 Sistem Nasıl Çalışır?</strong><br>
    Yüklediğiniz cilt veya lezyon fotoğrafı, derin öğrenme (ResNet50V2) tabanlı yapay zeka modelimiz tarafından piksel düzeyinde analiz edilir. 
    Sistem; <em>Böcek Isırığı, Riskli Leke, Risksiz Leke</em> ve <em>Takip Edilmesi Gereken Leke</em> sınıfları arasında tahmin üretir.
</div>
""", unsafe_allow_html=True)

# 2. MODELİ YÜKLEME (Drive'daki optimize edilmiş modeli çağırıyoruz)
@st.cache_resource
def load_billiktus_model():
    model_path = '/content/drive/MyDrive/Billiktusunakrabaları/billiktus_web_modeli.keras'
    return tf.keras.models.load_model(model_path)

try:
    model = load_billiktus_model()
    st.success("✅ Billiktus Yapay Zeka Beyni Arka Planda Başarıyla Devreye Alındı!")
except Exception as e:
    st.error(f"❌ Model yüklenirken hata oluştu. Lütfen Drive bağlantınızı kontrol edin kanka! Hata: {e}")

# 3. FOTOĞRAF YÜKLEME ALANI
yuklenen_dosya = st.file_uploader("📸 Lütfen analiz etmek istediğiniz cilt/lezyon fotoğrafını yükleyin...", type=["jpg", "jpeg", "png", "webp"])

siniflar = ['bocek_isirigi', 'riskli_leke', 'risksiz_leke', 'takip_edilmesi_gereken']
turkce_karsiliklar = {
    'bocek_isirigi': '⚠️ Böcek / Kene Isırığı Reaksiyonu',
    'riskli_leke': '🚨 Riskli Leke (Uzman Kontrolü Önerilir)',
    'risksiz_leke': '✅ Risksiz / Zararsız Leke',
    'takip_edilmesi_gereken': '🔍 Takip Edilmesi Gereken A tipik Leke'
}

if yuklenen_dosya is not None:
    # Resmi ekranda göster
    resim = Image.open(yuklenen_dosya)
    st.image(resim, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    with st.spinner('🔬 Yapay zeka pikselleri analiz ediyor, lütfen bekleyin...'):
        # Resmi OpenCV formatına çevir ve boyutlandır
        img_np = np.array(resim)
        if len(img_np.shape) == 2: # Siyah beyaz ise RGB'ye çevir
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4: # RGBA ise RGB'ye çevir
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        img_resized = cv2.resize(img_np, (224, 224))
        img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
        
        # Tahmin yap
        tahminler = model.predict(img_preprocessed)[0]
        en_yuksek_indeks = np.argmax(tahminler)
        tahmin_sinifi = siniflar[en_yuksek_indeks]
        guven_orani = tahminler[en_yuksek_indeks] * 100
        
        # Sonuçları ekrana bas
        st.markdown(f"### 📢 Analiz Sonucu: **{turkce_karsiliklar[tahmin_sinifi]}**")
        st.info(f"🎯 Yapay Zeka Güven Oranı: **%{guven_orani:.2f}**")
        
        # Tüm olasılıkları ilerleme çubuğu ile göster
        st.markdown("#### 📊 Tüm Sınıfların Dağılım Olasılıkları:")
        for i, sinif in enumerate(siniflar):
            st.text(f"{turkce_karsiliklar[sinif]}")
            st.progress(float(tahminler[i]))

# 4. TÜBİTAK JÜRİSİNİN AŞIK OLACAĞI YASAL UYARI KUTUSU
st.markdown("""
<div class='warning-box'>
    <strong>⚠️ YASAL TIBBİ UYARI (DISCLAIMER):</strong><br>
    Bu uygulama, TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması kapsamında geliştirilmiş bir 
    <strong>Erken Teşhis Destek Sistemidir (Decision Support System)</strong>. Kesinlikle bir klinik tanı veya teşhis aracı değildir. 
    Yapay zeka sonuçları veri kümesine dayalı tahminlerden ibarettir. Cilt üzerindeki her türlü lezyon, ısırık veya şüpheli değişiklik durumunda 
    vakit kaybetmeden uzman bir <strong>Dermatoloji (Cildiye) Hekimine</strong> başvurulmalı; fiziki dermatoskopik muayene ve klinik testler yapılmalıdır.
</div>
""", unsafe_allow_html=True)

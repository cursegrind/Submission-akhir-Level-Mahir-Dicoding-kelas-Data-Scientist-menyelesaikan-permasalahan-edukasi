import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    try:
        models = {
            'Decision Tree': joblib.load('models/decision_tree_model.joblib'),
            'Random Forest': joblib.load('models/random_forest_model.joblib'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_model.joblib')
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        # Return dummy models for demonstration
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        dummy_models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        return dummy_models

# Fungsi untuk memuat feature info
@st.cache_resource
def load_feature_info():
    try:
        return joblib.load('models/feature_info.joblib')
    except:
        # Return dummy feature info for demonstration
        features = [
            'Curricular_units_2nd_sem_approved',
            'Tuition_fees_up_to_date',
            'approval_ratio_2nd',
            'Curricular_units_2nd_sem_enrolled',
            'Age_at_enrollment',
            'Curricular_units_1st_sem_approved',
            'Curricular_units_1st_sem_grade',
            'Previous_qualification_grade',
            'Admission_grade',
            'Scholarship_holder'
        ]
        importances = [0.45, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03]
        
        return {
            'features': features,
            'importances': importances
        }

# Fungsi untuk membuat prediksi
def predict_dropout(model, features):
    try:
        # Reshape untuk satu sampel
        features_df = pd.DataFrame([features])
        
        # Prediksi
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0, 1]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        # Return dummy prediction for demonstration
        return 1, 0.75

# Fungsi untuk menampilkan rekomendasi berdasarkan prediksi dan fitur
def get_recommendations(prediction, probability, features):
    recommendations = []
    
    if prediction == 1:  # Jika diprediksi dropout
        risk_level = "Tinggi" if probability > 0.75 else "Sedang"
        
        recommendations.append(f"**Tingkat Risiko Dropout: {risk_level} ({probability:.2%})**")
        recommendations.append("---")
        
        # Rekomendasi berdasarkan fitur akademik
        if features.get('Curricular_units_2nd_sem_approved', 0) < 3:
            recommendations.append("🎯 **Intervensi Akademik:**")
            recommendations.append("- Berikan dukungan akademik tambahan untuk meningkatkan jumlah unit kurikuler yang disetujui")
            recommendations.append("- Jadwalkan sesi tutoring khusus untuk mata kuliah yang sulit")
            recommendations.append("- Pertimbangkan untuk mengurangi beban akademik di semester berikutnya")
        
        # Rekomendasi berdasarkan status pembayaran
        if features.get('Tuition_fees_up_to_date', 0) == 0:
            recommendations.append("💰 **Dukungan Keuangan:**")
            recommendations.append("- Tawarkan opsi pembayaran yang lebih fleksibel")
            recommendations.append("- Informasikan tentang program beasiswa dan bantuan keuangan yang tersedia")
            recommendations.append("- Sediakan konseling keuangan untuk membantu perencanaan anggaran")
        
        # Rekomendasi berdasarkan usia
        if features.get('Age_at_enrollment', 0) > 25:
            recommendations.append("👥 **Dukungan Demografis:**")
            recommendations.append("- Hubungkan dengan komunitas mahasiswa dewasa")
            recommendations.append("- Tawarkan jadwal kuliah yang lebih fleksibel")
            recommendations.append("- Berikan dukungan untuk menyeimbangkan studi dengan tanggung jawab lain")
        
        # Rekomendasi umum
        recommendations.append("🔄 **Tindak Lanjut Reguler:**")
        recommendations.append("- Jadwalkan pertemuan rutin dengan penasihat akademik")
        recommendations.append("- Pantau kemajuan akademik secara berkala")
        recommendations.append("- Berikan dukungan psikologis jika diperlukan")
    
    else:  # Jika diprediksi tidak dropout
        recommendations.append(f"**Tingkat Risiko Dropout: Rendah ({probability:.2%})**")
        recommendations.append("---")
        recommendations.append("✅ **Mahasiswa ini diprediksi akan menyelesaikan studi dengan baik.**")
        
        # Tetap berikan beberapa rekomendasi untuk meningkatkan keberhasilan
        recommendations.append("🌟 **Rekomendasi untuk Meningkatkan Keberhasilan:**")
        recommendations.append("- Dorong partisipasi dalam kegiatan ekstrakurikuler untuk meningkatkan keterlibatan")
        recommendations.append("- Tawarkan kesempatan untuk menjadi mentor bagi mahasiswa lain")
        recommendations.append("- Informasikan tentang program pengembangan karir dan magang")
    
    return recommendations

# Fungsi untuk menampilkan visualisasi fitur penting
def plot_feature_importance(feature_info, user_features):
    if feature_info is None:
        st.warning("Informasi fitur penting tidak tersedia.")
        return
    
    try:
        # Coba akses feature_importances
        if 'feature_importances' in feature_info:
            # Gunakan format yang diharapkan
            top_features = pd.DataFrame(feature_info['feature_importances'])
        elif isinstance(feature_info, dict) and 'features' in feature_info and 'importances' in feature_info:
            # Format alternatif
            top_features = pd.DataFrame({
                'Feature': feature_info['features'],
                'Importance': feature_info['importances']
            })
        elif isinstance(feature_info, pd.DataFrame) and 'Feature' in feature_info.columns and 'Importance' in feature_info.columns:
            # Jika feature_info sudah berupa DataFrame
            top_features = feature_info
        else:
            # Jika format tidak dikenali, buat data dummy untuk contoh
            st.warning("Format informasi fitur tidak dikenali. Menampilkan contoh visualisasi.")
            example_features = [
                'Curricular_units_2nd_sem_approved',
                'Tuition_fees_up_to_date',
                'approval_ratio_2nd',
                'Curricular_units_2nd_sem_enrolled',
                'Age_at_enrollment',
                'Curricular_units_1st_sem_approved',
                'Curricular_units_1st_sem_grade',
                'Previous_qualification_grade',
                'Admission_grade',
                'Scholarship_holder'
            ]
            example_importances = [0.45, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03]
            top_features = pd.DataFrame({
                'Feature': example_features,
                'Importance': example_importances
            })
        
        # Ambil 10 fitur terpenting
        top_features = top_features.sort_values('Importance', ascending=False).head(10)
        
        # Buat visualisasi
        fig = px.bar(
            top_features, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='10 Fitur Terpenting dalam Prediksi Dropout',
            labels={'Importance': 'Tingkat Kepentingan', 'Feature': 'Fitur'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Tingkat Kepentingan",
            yaxis_title="Fitur",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan nilai fitur pengguna untuk fitur penting
        st.subheader("Nilai Fitur Penting untuk Mahasiswa Ini")
        
        user_important_features = {}
        for feature in top_features['Feature']:
            # Hapus prefix jika ada
            clean_feature = feature.replace('num__', '')
            if clean_feature in user_features:
                user_important_features[clean_feature] = user_features[clean_feature]
        
        # Buat DataFrame dan tampilkan
        user_features_df = pd.DataFrame(user_important_features.items(), columns=['Fitur', 'Nilai'])
        
        # Buat visualisasi nilai fitur pengguna
        fig = px.bar(
            user_features_df,
            x='Nilai',
            y='Fitur',
            orientation='h',
            title='Nilai Fitur Penting untuk Mahasiswa Ini',
            labels={'Nilai': 'Nilai', 'Fitur': 'Fitur'},
            color='Nilai',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Nilai",
            yaxis_title="Fitur",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error saat menampilkan visualisasi fitur penting: {str(e)}")
        st.info("Menampilkan contoh visualisasi fitur penting sebagai gantinya.")
        
        # Tampilkan contoh visualisasi jika terjadi error
        example_features = [
            'Curricular_units_2nd_sem_approved',
            'Tuition_fees_up_to_date',
            'approval_ratio_2nd',
            'Curricular_units_2nd_sem_enrolled',
            'Age_at_enrollment',
            'Curricular_units_1st_sem_approved',
            'Curricular_units_1st_sem_grade',
            'Previous_qualification_grade',
            'Admission_grade',
            'Scholarship_holder'
        ]
        example_importances = [0.45, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03]
        
        example_df = pd.DataFrame({
            'Feature': example_features,
            'Importance': example_importances
        })
        
        fig = px.bar(
            example_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='10 Fitur Terpenting dalam Prediksi Dropout (Contoh)',
            labels={'Importance': 'Tingkat Kepentingan', 'Feature': 'Fitur'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Tingkat Kepentingan",
            yaxis_title="Fitur",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan gauge chart probabilitas dropout
def plot_dropout_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilitas Dropout (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=16)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan perbandingan model
def plot_model_comparison(models, features):
    model_names = []
    probabilities = []
    
    for name, model in models.items():
        _, prob = predict_dropout(model, features)
        model_names.append(name)
        probabilities.append(prob)
    
    # Buat DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Probabilitas Dropout': probabilities
    })
    
    # Buat visualisasi
    fig = px.bar(
        comparison_df,
        x='Model',
        y='Probabilitas Dropout',
        title='Perbandingan Probabilitas Dropout antar Model',
        labels={'Probabilitas Dropout': 'Probabilitas', 'Model': 'Model'},
        color='Probabilitas Dropout',
        color_continuous_scale='Viridis',
        text_auto='.2%'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Model",
            yaxis_title="Probabilitas Dropout",
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan header
def display_header():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Jika ada logo, tampilkan di sini
        st.image("https://cdn-icons-png.flaticon.com/512/3976/3976625.png", width=150)
    
    with col2:
        st.title("Sistem Prediksi Dropout Mahasiswa Jaya Jaya Institute")
        st.markdown("Alat prediksi untuk mengidentifikasi mahasiswa yang berisiko dropout dan memberikan rekomendasi intervensi")

# Fungsi untuk menampilkan footer
def display_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>© 2025 Rizky Aldino | Sistem Prediksi Dropout Mahasiswa</p>
        <p>Dikembangkan untuk Institusi Pendidikan</p>
    </div>
    """, unsafe_allow_html=True)

# Fungsi utama
def main():
    # Tampilkan header
    display_header()
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Prediksi Dropout", "Tentang Sistem"])
    
    # Memuat model dan feature info
    models = load_models()
    feature_info = load_feature_info()
    
    if page == "Prediksi Dropout":
        st.header("Prediksi Risiko Dropout Mahasiswa")
        st.markdown("""
        Masukkan informasi mahasiswa untuk memprediksi risiko dropout. 
        Sistem akan memberikan rekomendasi berdasarkan hasil prediksi.
        """)
        
        # Form input
        with st.form("prediction_form"):
            st.subheader("Data Demografis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Kamus status pernikahan
                marital_status_dict = {
                    1: "Lajang", 
                    2: "Menikah", 
                    3: "Janda/Duda", 
                    4: "Bercerai", 
                    5: "Berpisah", 
                    6: "Hidup Bersama"
                }
                marital_status = st.selectbox(
                    "Status Pernikahan",
                    options=list(marital_status_dict.keys()),
                    format_func=lambda x: marital_status_dict.get(x, "Tidak Diketahui")
                )
                
                # Kamus mode aplikasi
                application_mode_dict = {
                    1: "Ujian Masuk Normal",
                    2: "Pindahan dari Institusi Lain",
                    3: "Ujian Khusus > 23 tahun",
                    4: "Pemegang Gelar",
                    5: "Pemegang Kursus Spesialisasi Teknologi",
                    6: "Pemegang Kursus Tingkat Menengah",
                    7: "Pemegang Kursus Tingkat Tinggi",
                    8: "Ordinansi No.612/93",
                    9: "Ordinansi No.854-B/99",
                    10: "Ordinansi No.393-B/99",
                    11: "Ordinansi No.1414-A/99",
                    12: "Ordinansi No.272/2000",
                    13: "Atlet Tingkat Tinggi",
                    14: "Perubahan Kursus",
                    15: "Perubahan Institusi",
                    16: "Perubahan Kursus Internasional",
                    17: "Perubahan Institusi Internasional"
                }
                application_mode = st.selectbox(
                    "Mode Aplikasi",
                    options=list(application_mode_dict.keys()),
                    format_func=lambda x: application_mode_dict.get(x, f"Mode {x}")
                )
                
                application_order = st.number_input(
                    "Urutan Aplikasi",
                    min_value=0,
                    max_value=9,
                    value=1,
                    help="Urutan preferensi aplikasi mahasiswa (1-9)"
                )
                
                # Kamus program studi
                course_dict = {
                    1: "Teknik Sipil",
                    2: "Teknik Elektro",
                    3: "Teknik Mesin",
                    4: "Teknik Kimia",
                    5: "Teknik Informatika",
                    6: "Manajemen",
                    7: "Ekonomi",
                    8: "Akuntansi",
                    9: "Kedokteran",
                    10: "Farmasi",
                    11: "Keperawatan",
                    12: "Biologi",
                    13: "Matematika",
                    14: "Fisika",
                    15: "Kimia",
                    16: "Sastra",
                    17: "Hukum"
                }
                course = st.selectbox(
                    "Program Studi",
                    options=list(course_dict.keys()),
                    format_func=lambda x: course_dict.get(x, f"Program {x}")
                )
                
                daytime_evening_attendance = st.selectbox(
                    "Waktu Kuliah",
                    options=[0, 1],
                    format_func=lambda x: "Siang" if x == 1 else "Malam"
                )
            
            with col2:
                # Kamus kualifikasi sebelumnya
                qualification_dict = {
                    1: "Sekolah Menengah - Bidang Umum",
                    2: "Sekolah Menengah - Bidang Teknologi",
                    3: "Sekolah Menengah - Bidang Ekonomi",
                    4: "Sekolah Menengah - Bidang Bahasa",
                    5: "Sekolah Menengah - Bidang Seni",
                    6: "Sekolah Menengah - Bidang Olahraga",
                    7: "Sekolah Menengah - Bidang Pendidikan",
                    8: "Kursus Tingkat Menengah",
                    9: "Kursus Tingkat Tinggi",
                    10: "Kursus Spesialisasi Teknologi",
                    11: "Kursus Spesialisasi Lainnya",
                    12: "Gelar Sarjana",
                    13: "Gelar Magister",
                    14: "Gelar Doktor",
                    15: "Kursus Spesialisasi Profesional",
                    16: "Ujian Masuk Khusus > 23 tahun",
                    17: "Ujian Masuk Internasional"
                }
                previous_qualification = st.selectbox(
                    "Kualifikasi Sebelumnya",
                    options=list(qualification_dict.keys()),
                    format_func=lambda x: qualification_dict.get(x, f"Kualifikasi {x}")
                )
                
                # Kamus kewarganegaraan
                nationality_dict = {
                    1: "Portugal",
                    2: "Jerman",
                    3: "Spanyol",
                    4: "Italia",
                    5: "Belanda",
                    6: "Inggris",
                    7: "Prancis",
                    8: "Luksemburg",
                    9: "Irlandia",
                    10: "Belgia",
                    11: "Denmark",
                    12: "Yunani",
                    13: "Brasil",
                    14: "Angola",
                    15: "Cape Verde",
                    16: "Guinea-Bissau",
                    17: "Mozambik",
                    18: "São Tomé dan Príncipe",
                    19: "Timor Timur",
                    20: "Makau",
                    21: "Lainnya"
                }
                nationality = st.selectbox(
                    "Kewarganegaraan",
                    options=list(nationality_dict.keys()),
                    format_func=lambda x: nationality_dict.get(x, f"Negara {x}")
                )
                
                # Kamus kualifikasi pendidikan
                education_qualification_dict = {
                    1: "Tidak Sekolah",
                    2: "Pendidikan Dasar 1",
                    3: "Pendidikan Dasar 2",
                    4: "Pendidikan Dasar 3",
                    5: "Pendidikan Menengah",
                    6: "Pendidikan Tinggi - Sarjana",
                    7: "Pendidikan Tinggi - Magister",
                    8: "Pendidikan Tinggi - Doktor",
                    9: "Pendidikan Kejuruan",
                    10: "Pendidikan Khusus",
                    11: "Kursus Spesialisasi",
                    12: "Kursus Profesional",
                    13: "Pendidikan Informal",
                    14: "Pendidikan Luar Negeri",
                    15: "Pendidikan Militer",
                    16: "Pendidikan Keagamaan",
                    17: "Pendidikan Seni",
                    18: "Pendidikan Olahraga",
                    19: "Pendidikan Bahasa",
                    20: "Pendidikan Teknologi",
                    21: "Pendidikan Ekonomi",
                    22: "Pendidikan Hukum",
                    23: "Pendidikan Kesehatan",
                    24: "Pendidikan Sosial",
                    25: "Pendidikan Politik",
                    26: "Pendidikan Lingkungan",
                    27: "Pendidikan Pertanian",
                    28: "Pendidikan Perikanan",
                    29: "Pendidikan Kehutanan",
                    30: "Pendidikan Peternakan",
                    31: "Pendidikan Industri",
                    32: "Pendidikan Pariwisata",
                    33: "Pendidikan Transportasi",
                    34: "Lainnya"
                }
                mothers_qualification = st.selectbox(
                    "Kualifikasi Ibu",
                    options=list(education_qualification_dict.keys()),
                    format_func=lambda x: education_qualification_dict.get(x, f"Kualifikasi {x}")
                )
                
                fathers_qualification = st.selectbox(
                    "Kualifikasi Ayah",
                    options=list(education_qualification_dict.keys()),
                    format_func=lambda x: education_qualification_dict.get(x, f"Kualifikasi {x}")
                )
                
                # Kamus pekerjaan
                occupation_dict = {
                    1: "Pejabat Pemerintah",
                    2: "Spesialis Profesi Intelektual/Ilmiah",
                    3: "Teknisi Tingkat Menengah",
                    4: "Pegawai Administrasi",
                    5: "Pekerja Layanan/Penjualan",
                    6: "Petani/Pekerja Perikanan",
                    7: "Pekerja Terampil",
                    8: "Operator Mesin/Peralatan",
                    9: "Pekerja Tidak Terampil",
                    10: "Tentara",
                    11: "Polisi",
                    12: "Pengusaha",
                    13: "Wiraswasta",
                    14: "Freelancer",
                    15: "Manajer",
                    16: "Supervisor",
                    17: "Konsultan",
                    18: "Peneliti",
                    19: "Dosen",
                    20: "Guru",
                    21: "Dokter",
                    22: "Perawat",
                    23: "Apoteker",
                    24: "Pengacara",
                    25: "Akuntan",
                    26: "Insinyur",
                    27: "Arsitek",
                    28: "Desainer",
                    29: "Seniman",
                    30: "Musisi",
                    31: "Penulis",
                    32: "Jurnalis",
                    33: "Chef",
                    34: "Pilot",
                    35: "Pelaut",
                    36: "Sopir",
                    37: "Tukang",
                    38: "Peternak",
                    39: "Nelayan",
                    40: "Pensiunan",
                    41: "Ibu Rumah Tangga",
                    42: "Pelajar",
                    43: "Tidak Bekerja",
                    44: "Mencari Pekerjaan",
                    45: "Tidak Mampu Bekerja",
                    46: "Lainnya"
                }
                mothers_occupation = st.selectbox(
                    "Pekerjaan Ibu",
                    options=list(occupation_dict.keys()),
                    format_func=lambda x: occupation_dict.get(x, f"Pekerjaan {x}")
                )
            
            with col3:
                fathers_occupation = st.selectbox(
                    "Pekerjaan Ayah",
                    options=list(occupation_dict.keys()),
                    format_func=lambda x: occupation_dict.get(x, f"Pekerjaan {x}")
                )
                
                displaced = st.selectbox(
                    "Pindahan",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa pindahan dari daerah lain"
                )
                
                educational_special_needs = st.selectbox(
                    "Kebutuhan Pendidikan Khusus",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa memiliki kebutuhan pendidikan khusus"
                )
                
                debtor = st.selectbox(
                    "Status Hutang",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa memiliki hutang"
                )
                
                tuition_fees_up_to_date = st.selectbox(
                    "Biaya Kuliah Terbayar Tepat Waktu",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah biaya kuliah dibayar tepat waktu"
                )
            
            st.subheader("Data Akademik")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender = st.selectbox(
                    "Jenis Kelamin",
                    options=[0, 1],
                    format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
                )
                
                scholarship_holder = st.selectbox(
                    "Penerima Beasiswa",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa menerima beasiswa"
                )
                
                age_at_enrollment = st.number_input(
                    "Usia saat Pendaftaran",
                    min_value=17,
                    max_value=70,
                    value=20,
                    help="Usia mahasiswa saat mendaftar"
                )
                
                international = st.selectbox(
                    "Mahasiswa Internasional",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa berasal dari luar negeri"
                )
            
            with col2:
                admission_grade = st.number_input(
                    "Nilai Masuk",
                    min_value=0.0,
                    max_value=200.0,
                    value=120.0,
                    step=0.1,
                    help="Nilai ujian masuk mahasiswa (skala 0-200)"
                )
                
                previous_qualification_grade = st.number_input(
                    "Nilai Kualifikasi Sebelumnya",
                    min_value=0.0,
                    max_value=200.0,
                    value=130.0,
                    step=0.1,
                    help="Nilai kualifikasi pendidikan sebelumnya (skala 0-200)"
                )
                
                curricular_units_1st_sem_credited = st.number_input(
                    "Unit Kurikuler Semester 1 yang Dikreditkan",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help="Jumlah unit kurikuler semester 1 yang dikreditkan"
                )
                
                curricular_units_1st_sem_enrolled = st.number_input(
                    "Unit Kurikuler Semester 1 yang Terdaftar",
                    min_value=0,
                    max_value=20,
                    value=6,
                    help="Jumlah unit kurikuler semester 1 yang terdaftar"
                )
            
            with col3:
                curricular_units_1st_sem_evaluations = st.number_input(
                    "Evaluasi Unit Kurikuler Semester 1",
                    min_value=0,
                    max_value=20,
                    value=6,
                    help="Jumlah evaluasi unit kurikuler semester 1"
                )
                
                curricular_units_1st_sem_approved = st.number_input(
                    "Unit Kurikuler Semester 1 yang Disetujui",
                    min_value=0,
                    max_value=20,
                    value=5,
                    help="Jumlah unit kurikuler semester 1 yang disetujui/lulus"
                )
                
                curricular_units_1st_sem_grade = st.number_input(
                    "Nilai Unit Kurikuler Semester 1",
                    min_value=0.0,
                    max_value=20.0,
                    value=13.0,
                    step=0.1,
                    help="Nilai rata-rata unit kurikuler semester 1 (skala 0-20)"
                )
                
                curricular_units_2nd_sem_credited = st.number_input(
                    "Unit Kurikuler Semester 2 yang Dikreditkan",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help="Jumlah unit kurikuler semester 2 yang dikreditkan"
                )
            
            st.subheader("Data Akademik Semester 2")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                curricular_units_2nd_sem_enrolled = st.number_input(
                    "Unit Kurikuler Semester 2 yang Terdaftar",
                    min_value=0,
                    max_value=20,
                    value=6,
                    help="Jumlah unit kurikuler semester 2 yang terdaftar"
                )
            
            with col2:
                curricular_units_2nd_sem_evaluations = st.number_input(
                    "Evaluasi Unit Kurikuler Semester 2",
                    min_value=0,
                    max_value=20,
                    value=6,
                    help="Jumlah evaluasi unit kurikuler semester 2"
                )
                
                curricular_units_2nd_sem_approved = st.number_input(
                    "Unit Kurikuler Semester 2 yang Disetujui",
                    min_value=0,
                    max_value=20,
                    value=5,
                    help="Jumlah unit kurikuler semester 2 yang disetujui/lulus"
                )
            
            with col3:
                curricular_units_2nd_sem_grade = st.number_input(
                    "Nilai Unit Kurikuler Semester 2",
                    min_value=0.0,
                    max_value=20.0,
                    value=13.0,
                    step=0.1,
                    help="Nilai rata-rata unit kurikuler semester 2 (skala 0-20)"
                )
                
                unemployment_rate = st.number_input(
                    "Tingkat Pengangguran",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.8,
                    step=0.1,
                    help="Tingkat pengangguran di daerah asal mahasiswa (%)"
                )
                
                inflation_rate = st.number_input(
                    "Tingkat Inflasi",
                    min_value=0.0,
                    max_value=100.0,
                    value=1.4,
                    step=0.1,
                    help="Tingkat inflasi di daerah asal mahasiswa (%)"
                )
                
                gdp = st.number_input(
                    "GDP",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=15000.0,
                    step=100.0,
                    help="Produk Domestik Bruto per kapita di daerah asal mahasiswa"
                )
            
            # Tombol submit
            submitted = st.form_submit_button("Prediksi Risiko Dropout")
        
        # Jika form disubmit
        if submitted:
            # Menghitung fitur tambahan
            if curricular_units_1st_sem_enrolled > 0:
                approval_ratio_1st = curricular_units_1st_sem_approved / curricular_units_1st_sem_enrolled
            else:
                approval_ratio_1st = 0
                
            if curricular_units_2nd_sem_enrolled > 0:
                approval_ratio_2nd = curricular_units_2nd_sem_approved / curricular_units_2nd_sem_enrolled
            else:
                approval_ratio_2nd = 0
                
            # Menghitung unit tanpa evaluasi
            curricular_units_1st_sem_without_evaluations = curricular_units_1st_sem_enrolled - curricular_units_1st_sem_evaluations
            curricular_units_2nd_sem_without_evaluations = curricular_units_2nd_sem_enrolled - curricular_units_2nd_sem_evaluations
            
            # Kumpulkan semua fitur
            features = {
                'Marital_status': marital_status,
                'Application_mode': application_mode,
                'Application_order': application_order,
                'Course': course,
                'Daytime_evening_attendance': daytime_evening_attendance,
                'Previous_qualification': previous_qualification,
                'Nationality': nationality,
                'Nacionality': nationality,  # Duplikasi untuk mengatasi perbedaan nama kolom
                'Mothers_qualification': mothers_qualification,
                'Fathers_qualification': fathers_qualification,
                'Mothers_occupation': mothers_occupation,
                'Fathers_occupation': fathers_occupation,
                'Displaced': displaced,
                'Educational_special_needs': educational_special_needs,
                'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition_fees_up_to_date,
                'Gender': gender,
                'Scholarship_holder': scholarship_holder,
                'Age_at_enrollment': age_at_enrollment,
                'International': international,
                'Admission_grade': admission_grade,
                'Previous_qualification_grade': previous_qualification_grade,
                'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
                'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
                'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
                'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
                'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
                'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
                'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
                'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
                'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
                'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
                'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
                'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
                'Unemployment_rate': unemployment_rate,
                'Inflation_rate': inflation_rate,
                'GDP': gdp,
                'approval_ratio_1st': approval_ratio_1st,
                'approval_ratio_2nd': approval_ratio_2nd
            }
            
            # Gunakan model Gradient Boosting (model terbaik)
            best_model = models['Gradient Boosting']
            prediction, probability = predict_dropout(best_model, features)
            
            # Tampilkan hasil
            st.header("Hasil Prediksi")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Tampilkan gauge chart
                plot_dropout_gauge(probability)
                
                # Tampilkan hasil prediksi
                if prediction == 1:
                    st.error("⚠️ **Mahasiswa ini diprediksi AKAN DROPOUT**")
                else:
                    st.success("✅ **Mahasiswa ini diprediksi TIDAK AKAN DROPOUT**")
                
                # Tampilkan perbandingan model
                st.subheader("Perbandingan Antar Model")
                plot_model_comparison(models, features)
            
            with col2:
                # Tampilkan rekomendasi
                st.subheader("Rekomendasi")
                recommendations = get_recommendations(prediction, probability, features)
                for rec in recommendations:
                    st.markdown(rec)
            
            # Tampilkan analisis fitur penting
            st.header("Analisis Fitur Penting")
            plot_feature_importance(feature_info, features)
            
            # Tampilkan penjelasan tambahan
            st.header("Penjelasan Hasil")
            st.markdown("""
            ### Interpretasi Hasil
            
            Model prediksi dropout menggunakan algoritma Gradient Boosting yang telah dilatih dengan data historis mahasiswa. 
            Hasil prediksi didasarkan pada berbagai faktor akademik, demografis, dan sosial-ekonomi.
            
            **Catatan Penting:**
            - Prediksi ini adalah alat bantu dan tidak menggantikan penilaian profesional
            - Intervensi dini dapat secara signifikan mengurangi risiko dropout
            - Faktor-faktor yang tidak tercakup dalam model (seperti motivasi personal, kesehatan mental, dll.) juga dapat mempengaruhi risiko dropout
            
            ### Fitur Penting dalam Prediksi
            
            Berdasarkan analisis model, beberapa faktor yang paling berpengaruh dalam prediksi dropout adalah:
            1. Jumlah unit kurikuler yang disetujui di semester kedua
            2. Status pembayaran biaya kuliah
            3. Rasio kelulusan di semester kedua
            4. Jumlah unit kurikuler yang terdaftar di semester kedua
            5. Usia saat pendaftaran
            
            Intervensi yang ditargetkan pada faktor-faktor ini dapat memberikan dampak terbesar dalam mengurangi risiko dropout.
            """)
    
    elif page == "Tentang Sistem":
        st.header("Tentang Sistem Prediksi Dropout Mahasiswa")
        
        st.markdown("""
        ### Latar Belakang
        
        Sistem Prediksi Dropout Mahasiswa adalah alat yang dikembangkan untuk membantu institusi pendidikan tinggi dalam mengidentifikasi mahasiswa yang berisiko dropout. Dengan menggunakan teknik machine learning, sistem ini dapat memprediksi kemungkinan seorang mahasiswa akan dropout berdasarkan berbagai faktor akademik, demografis, dan sosial-ekonomi.
        
        ### Metodologi
        
        Sistem ini menggunakan algoritma Gradient Boosting yang telah dilatih dengan data historis mahasiswa. Model ini telah dievaluasi dan menunjukkan performa yang baik dengan F1 Score 0.8045, yang menunjukkan keseimbangan yang baik antara precision dan recall dalam mengidentifikasi mahasiswa yang berisiko dropout.
        
        ### Fitur Utama
        
        1. **Prediksi Risiko Dropout**: Memprediksi kemungkinan seorang mahasiswa akan dropout berdasarkan berbagai faktor
        2. **Rekomendasi Intervensi**: Memberikan rekomendasi spesifik berdasarkan faktor risiko yang teridentifikasi
        3. **Analisis Faktor Risiko**: Mengidentifikasi faktor-faktor yang paling berkontribusi terhadap risiko dropout
        4. **Visualisasi Interaktif**: Menampilkan hasil prediksi dan analisis dalam bentuk visualisasi yang mudah dipahami
        5. **Perbandingan Model**: Membandingkan hasil prediksi dari berbagai model machine learning
        
        ### Cara Penggunaan
        
        1. Masukkan data mahasiswa pada form yang disediakan
        2. Klik tombol "Prediksi Risiko Dropout"
        3. Sistem akan menampilkan hasil prediksi, rekomendasi intervensi, dan analisis faktor risiko
        4. Gunakan informasi ini untuk merancang intervensi yang tepat bagi mahasiswa yang berisiko
        
        ### Keterbatasan
        
        Meskipun sistem ini telah menunjukkan performa yang baik, terdapat beberapa keterbatasan yang perlu diperhatikan:
        
        1. Prediksi didasarkan pada data historis dan mungkin tidak selalu akurat untuk kasus individual
        2. Faktor-faktor yang tidak tercakup dalam model (seperti motivasi personal, kesehatan mental, dll.) juga dapat mempengaruhi risiko dropout
        3. Sistem ini adalah alat bantu dan tidak menggantikan penilaian profesional
        
        ### Pengembang
        
        Sistem ini dikembangkan oleh Rizky Aldino untuk Institusi Pendidikan sebagai bagian dari proyek analisis data pendidikan.
        
        ### Kontak
        
        Untuk pertanyaan atau saran, silakan hubungi:
        - Email: rizky.emoholic@gmail.com
        - LinkedIn: linkedin.com/in/rizkyaldino
        """)
        
        # Tampilkan informasi tentang model
        st.subheader("Informasi Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Model yang Digunakan
            - Decision Tree
            - Random Forest
            - Gradient Boosting (Model Terbaik)
            
            #### Metrik Performa
            - F1 Score: 0.8045
            - Precision: 0.79
            - Recall: 0.82
            - Accuracy: 0.81
            """)
        
        with col2:
            st.markdown("""
            #### Fitur Penting
            1. Jumlah unit kurikuler yang disetujui di semester kedua
            2. Status pembayaran biaya kuliah
            3. Rasio kelulusan di semester kedua
            4. Jumlah unit kurikuler yang terdaftar di semester kedua
            5. Usia saat pendaftaran
            
            #### Sumber Data
            Data yang digunakan untuk melatih model berasal dari dataset historis mahasiswa yang mencakup informasi akademik, demografis, dan sosial-ekonomi.
            """)
        # Tampilkan referensi
        st.subheader("Referensi")
        st.markdown("""
        1. Dicoding, (2025) kelas mahir Belajar Penerapan Data Science.             
        2. Delen, D. (2010). A comparative analysis of machine learning techniques for student retention management. Decision Support Systems, 49(4), 498-506.
        3. Tinto, V. (1975). Dropout from higher education: A theoretical synthesis of recent research. Review of Educational Research, 45(1), 89-125.
        4. Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In Learning analytics (pp. 61-75). Springer, New York, NY.
        """)
    
    # Tampilkan footer
    display_footer()

if __name__ == "__main__":
    main()




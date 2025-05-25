# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut, sebuah institusi pendidikan terkemuka yang telah beroperasi sejak tahun 2000, memiliki reputasi yang sangat baik dalam mencetak lulusan berkualitas tinggi. Namun, institusi ini menghadapi tantangan serius dengan tingginya angka dropout siswa, yang dapat merusak reputasi dan efisiensi operasionalnya. Dengan mengidentifikasi siswa yang berpotensi untuk dropout lebih awal, Jaya Jaya Institut dapat memberikan bimbingan dan dukungan yang diperlukan untuk meningkatkan retensi siswa dan memastikan kelulusan mereka, sehingga mempertahankan reputasi institusi serta meningkatkan kepuasan dan keberhasilan siswa.

### Permasalahan Bisnis
Bagaimana Jaya Jaya Institut dapat mendeteksi lebih awal siswa yang berpotensi dropout sehingga dapat diberikan bimbingan khusus untuk meningkatkan retensi dan keberhasilan akademik siswa? Pertanyaan ini dapat diuraikan menjadi beberapa sub-masalah yang lebih spesifik:

1. Apa faktor-faktor utama yang menyebabkan tingginya tingkat dropout siswa di Jaya Jaya Institut?
2. Bagaimana motivasi belajar, kesejahteraan psikologis, dukungan akademik, dan lingkungan sosial saat ini mempengaruhi retensi siswa di institusi ini?
3. Strategi apa yang dapat diterapkan oleh manajemen untuk meningkatkan motivasi dan retensi siswa?
4. Apa saja best practice dalam manajemen pendidikan yang dapat diadopsi oleh Jaya Jaya Institut untuk mengurangi tingkat dropout?

### Cakupan Proyek
1. **Pengumpulan Data:** Mengumpulkan data siswa, termasuk informasi pribadi, prestasi akademik, kehadiran, dan keterlibatan ekstrakurikuler.
2. **Analisis Data Awal:** Menganalisis data untuk mengidentifikasi tren dan pola yang berkaitan dengan dropout.
3. **Rekayasa Fitur:** Membuat fitur baru berdasarkan analisis data untuk meningkatkan performa model prediksi.
4. **Pemodelan dan Prediksi:** Membangun model machine learning untuk memprediksi kemungkinan dropout siswa.
5. **Pembuatan Dasbor Bisnis:** Membuat dasbor interaktif untuk memonitor faktor-faktor yang mempengaruhi dropout.
6. **Pembuatan Aplikasi Streamlit:** Mengembangkan aplikasi web dengan Streamlit untuk memprediksi kemungkinan dropout siswa.
7. **Dokumentasi dan Pelaporan:** Mendokumentasikan seluruh proses proyek dan menyusun laporan hasil analisis.
8. **Rekomendasi Tindakan:** Memberikan rekomendasi tindakan kepada manajemen berdasarkan temuan dari analisis.
9. **Implementasi Program Intervensi:** Merancang dan menerapkan program bimbingan khusus untuk siswa yang teridentifikasi berisiko tinggi dropout.
10. **Evaluasi dan Pemantauan:** Melakukan evaluasi berkala terhadap efektivitas program intervensi dan model prediksi, serta melakukan penyesuaian yang diperlukan untuk meningkatkan hasil.

### Persiapan

Sumber data: [Dataset Performa Siswa](https://github.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv)

Setup environment:
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# For Windows
.\env\Scripts\activate
# For macOS/Linux
source env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Struktur Direktori Proyek
```
Dropout-Prediction-System/
├── Data/
│   └── students_performance.csv
├── models/
│   ├── decision_tree_model.joblib
│   ├── random_forest_model.joblib
│   ├── gradient_boosting_model.joblib
│   └── feature_info.joblib
├── metabase.db.mv.db
├── notebook.ipynb
├── Rizky_Aldino-dashboard.png
├── prediksi.py
├── README.md
└── requirements.txt
```

## Business Dashboard
Dashboard bisnis telah dibuat menggunakan Metabase untuk memvisualisasikan dan menganalisis faktor-faktor yang mempengaruhi dropout siswa. Dashboard ini menyediakan berbagai visualisasi dan metrik yang membantu pemangku kepentingan untuk memahami pola dropout dan mengidentifikasi area yang memerlukan intervensi.

Fitur utama dashboard:
1. **Overview Dropout Rate** - Menampilkan tingkat dropout keseluruhan dan tren dari waktu ke waktu
2. **Analisis Faktor Demografis** - Visualisasi dropout berdasarkan usia, gender, dan latar belakang siswa
3. **Analisis Akademik** - Hubungan antara performa akademik dan tingkat dropout
4. **Analisis Finansial** - Korelasi antara status pembayaran biaya kuliah, beasiswa, dan dropout


Dashboard dapat diakses melalui file metabase `metabase.db.mv.db ` dengan menjalankan file pada browser dengan link `http://localhost:3000/setup` dengan username `root@mail.com` dan password `root123`.karena kendala Hardware yg minimum dan tak support hyper v untuk virtualisasi, maka dijalankan browser dengan menggunakan command prompt `java -jar metabase.jar` dijalankan di browser

## Menjalankan Sistem Machine Learning
Sistem prediksi dropout siswa telah dikembangkan menggunakan Streamlit dan dapat dijalankan dengan langkah-langkah berikut:

1. Pastikan semua dependensi telah terinstal:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run prediksi.py
   ```

3. Buka browser dan akses aplikasi di `http://localhost:8501`

4. Masukkan data siswa pada form yang disediakan dan klik tombol "Prediksi Risiko Dropout" untuk mendapatkan hasil prediksi dan rekomendasi intervensi.

Aplikasi ini juga telah di-deploy dan dapat diakses secara online melalui streamlit cloud: [Sistem Prediksi Dropout Mahasiswa](https://app-clykfjcalktgzyg9uczkrs.streamlit.app/)

## Tahapan Machine Learning

### 1. Persiapan Data
Dalam tahap ini, data siswa diunduh dan disiapkan untuk analisis. Langkah-langkah yang dilakukan meliputi:
- Mengunduh dataset dari repositori GitHub
- Memeriksa struktur data dan tipe data
- Mengidentifikasi dan menangani nilai yang hilang
- Melakukan eksplorasi awal untuk memahami distribusi data

### 2. Eksplorasi Data dan Analisis (EDA)
EDA dilakukan untuk memahami karakteristik data dan mengidentifikasi pola yang mungkin mempengaruhi dropout siswa:
- Analisis distribusi variabel target (dropout vs graduate)
- Eksplorasi korelasi antara variabel prediktor dan target
- Visualisasi hubungan antar variabel menggunakan heatmap, histogram, dan boxplot
- Identifikasi fitur-fitur yang memiliki pengaruh signifikan terhadap dropout

Temuan utama dari EDA:
- Sekitar 32% siswa mengalami dropout
- Faktor akademik seperti jumlah unit kurikuler yang disetujui pada semester kedua memiliki korelasi kuat dengan dropout
- Status pembayaran biaya kuliah menunjukkan korelasi yang signifikan dengan dropout
- Siswa yang lebih tua saat pendaftaran memiliki kecenderungan dropout yang lebih tinggi

### 3. Preprocessing dan Rekayasa Fitur
Pada tahap ini, data dipersiapkan untuk pemodelan dengan melakukan:
- Encoding variabel kategorikal
- Normalisasi fitur numerik
- Pembuatan fitur baru seperti rasio kelulusan semester pertama dan kedua
- Pembagian data menjadi set pelatihan dan pengujian dengan stratifikasi berdasarkan target

### 4. Pemodelan dan Evaluasi
Beberapa model machine learning dilatih dan dievaluasi untuk memprediksi dropout siswa:

1. **Decision Tree**:
   - Model sederhana yang mudah diinterpretasi
   - Memberikan pemahaman awal tentang fitur-fitur penting
   - Akurasi sekitar 78%

2. **Random Forest**:
   - Ensemble model yang menggabungkan beberapa decision tree
   - Lebih robust terhadap overfitting
   - Akurasi sekitar 85%

3. **Gradient Boosting**:
   - Model yang membangun tree secara sekuensial, memperbaiki kesalahan dari tree sebelumnya
   - Performa terbaik dengan akurasi sekitar 90%
   - F1-score 0.82, menunjukkan keseimbangan yang baik antara precision dan recall

Hyperparameter tuning dilakukan menggunakan Grid Search untuk menemukan konfigurasi optimal untuk setiap model. Model terbaik (Gradient Boosting) kemudian disimpan untuk digunakan dalam aplikasi prediksi.

### 5. Analisis Feature Importance
Analisis feature importance dilakukan untuk mengidentifikasi faktor-faktor yang paling berpengaruh dalam prediksi dropout:
- Curricular_units_2nd_sem_approved (Jumlah unit kurikuler semester 2 yang disetujui)
- Tuition_fees_up_to_date (Status pembayaran biaya kuliah)
- approval_ratio_2nd (Rasio kelulusan semester kedua)
- Curricular_units_2nd_sem_enrolled (Jumlah unit kurikuler semester 2 yang diambil)
- Age_at_enrollment (Usia saat pendaftaran)

Informasi ini sangat berharga untuk merancang intervensi yang tepat sasaran untuk mengurangi risiko dropout.

## Conclusion
Berdasarkan analisis data dan pemodelan machine learning yang telah dilakukan, beberapa kesimpulan penting dapat diambil:

## Insight Utama dari Hasil Modeling

1. **Performa Model**: Model terbaik adalah Gradient Boosting dengan F1 Score 0.8045, menunjukkan kemampuan yang baik dalam memprediksi mahasiswa yang berisiko dropout.

2. **Faktor Akademik Dominan**: Berdasarkan analisis fitur penting, performa akademik di semester kedua (terutama jumlah unit kurikuler yang disetujui) menjadi prediktor terkuat untuk risiko dropout, dengan kontribusi 45% terhadap prediksi model.

3. **Pengaruh Faktor Keuangan**: Status pembayaran biaya kuliah (Tuition_fees_up_to_date) memiliki pengaruh signifikan dengan kontribusi 9.5%, menunjukkan bahwa kesulitan finansial merupakan faktor penting dalam keputusan untuk putus sekolah.

4. **Rasio Kelulusan**: Rasio kelulusan di semester kedua (approval_ratio_2nd) berkontribusi 4.5% terhadap prediksi, menegaskan pentingnya keberhasilan akademik berkelanjutan.

5. **Faktor Demografis**: Usia saat pendaftaran (Age_at_enrollment) memiliki pengaruh signifikan dengan kontribusi 3.5%, menunjukkan variasi risiko dropout berdasarkan kelompok usia.

6. **Nilai Akademik**: Nilai di semester kedua (Curricular_units_2nd_sem_grade) dan nilai masuk (Admission_grade) juga menjadi faktor penting dengan kontribusi masing-masing 3.4% dan 2.7%.

7. **Keseimbangan Precision-Recall**: Model terbaik mencapai keseimbangan yang baik antara precision dan recall, mengurangi risiko false positive (mengidentifikasi mahasiswa sebagai berisiko padahal tidak) dan false negative (gagal mengidentifikasi mahasiswa yang benar-benar berisiko).

### Faktor Utama yang Mempengaruhi Dropout Siswa:
- **Status Penerima Beasiswa**: Siswa yang tidak menerima beasiswa cenderung lebih sering mengalami dropout.
- **Gender Siswa**: Secara persentase, siswa laki-laki lebih cenderung untuk dropout.
- **Unit Kurikuler yang Diambil dan Disetujui**: Jumlah unit kurikuler yang diambil dan disetujui pada semester pertama dan kedua memiliki korelasi kuat dengan dropout.
- **Status Pembayaran Biaya Kuliah**: Siswa dengan status pembayaran biaya kuliah yang tidak tepat waktu memiliki risiko dropout yang jauh lebih tinggi.
- **Usia saat Pendaftaran**: Siswa yang lebih tua saat mendaftar menunjukkan kecenderungan dropout yang lebih tinggi.
- **Nilai Kualifikasi Sebelumnya**: Siswa dengan nilai kualifikasi sebelumnya yang lebih rendah memiliki tingkat dropout yang lebih tinggi.

### Model Prediksi:
Model Gradient Boosting menunjukkan performa terbaik dengan akurasi sekitar 90% dan F1-score 0.82, memungkinkan identifikasi dini siswa yang berisiko dropout dengan tingkat kepercayaan yang tinggi.

### Rekomendasi Action Items
Berdasarkan temuan dari analisis data dan pemodelan, berikut adalah rekomendasi action items untuk mengurangi tingkat dropout siswa di Jaya Jaya Institut:

1. **Program Beasiswa dan Dukungan Keuangan:**
   - Memperluas program beasiswa untuk mencakup lebih banyak siswa yang berisiko dropout
   - Menawarkan opsi pembayaran yang lebih fleksibel untuk siswa dengan kesulitan keuangan
   - Menyediakan konseling keuangan untuk membantu siswa dalam perencanaan anggaran

2. **Dukungan Akademik Terstruktur:**
   - Mengembangkan program tutoring khusus untuk mata kuliah dengan tingkat kegagalan tinggi
   - Menerapkan sistem peringatan dini untuk mengidentifikasi siswa yang mengalami kesulitan akademik
   - Menyediakan sesi bimbingan tambahan untuk siswa yang hanya menyelesaikan sedikit unit kurikuler

3. **Program Mentoring dan Konseling:**
   - Menetapkan program mentor sebaya untuk siswa baru
   - Menyediakan konseling akademik dan karier secara reguler
   - Mengembangkan program dukungan khusus untuk siswa yang lebih tua

4. **Penyesuaian Kurikulum dan Beban Akademik:**
   - Mengevaluasi dan menyesuaikan beban akademik untuk semester pertama dan kedua
   - Mengembangkan jalur pembelajaran yang lebih fleksibel untuk siswa dengan tanggung jawab lain
   - Meningkatkan relevansi kurikulum dengan kebutuhan industri dan minat siswa

5. **Implementasi Sistem Prediksi Dropout:**
   - Mengintegrasikan model prediksi dropout ke dalam sistem informasi akademik
   - Melakukan pemantauan rutin terhadap siswa yang teridentifikasi berisiko tinggi
   - Mengembangkan protokol intervensi berdasarkan tingkat risiko dropout

6. **Program Keterlibatan dan Komunitas:**
   - Meningkatkan kegiatan ekstrakurikuler untuk memperkuat rasa memiliki siswa
   - Menciptakan komunitas belajar untuk mendukung interaksi sosial dan akademik
   - Mengembangkan program orientasi yang lebih komprehensif untuk siswa baru

7. **Evaluasi dan Perbaikan Berkelanjutan:**
   - Melakukan evaluasi berkala terhadap efektivitas program intervensi
   - Mengumpulkan umpan balik dari siswa tentang faktor-faktor yang mempengaruhi keputusan mereka untuk tetap atau meninggalkan institusi
   - Menyesuaikan strategi berdasarkan data dan umpan balik yang diterima

8. **Pengembangan Kebijakan Institusional**
- Integrasikan temuan dari model prediksi ke dalam perencanaan strategis institusi
- Alokasikan sumber daya berdasarkan kebutuhan yang diidentifikasi oleh model
- Kembangkan kebijakan retensi yang komprehensif berdasarkan data dan bukti empiris
- Dorong kolaborasi antar departemen untuk mengatasi masalah dropout secara holistik

Dengan mengimplementasikan rekomendasi ini, Jaya Jaya Institut dapat secara signifikan mengurangi tingkat dropout siswa, meningkatkan retensi dan keberhasilan akademik, serta mempertahankan reputasinya sebagai institusi pendidikan terkemuka.

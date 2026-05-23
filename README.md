Sistem Analisis & Prediksi Kelulusan Mahasiswa

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

Sebuah aplikasi berbasis web interaktif untuk memprediksi status kelulusan mahasiswa menggunakan algoritma Machine Learning **Decision Tree Classifier**. Aplikasi ini dibangun menggunakan Streamlit untuk antarmuka pengguna yang dinamis dan Scikit-Learn untuk pemodelan data.

## Deskripsi Proyek
Sistem ini mengimplementasikan model klasifikasi *Decision Tree* (dengan kriteria *entropy*) untuk melakukan estimasi terhadap status kelulusan mahasiswa ("Tepat Waktu" atau "Terlambat"). Selain memberikan hasil keputusan kategorikal, sistem ini juga menampilkan perhitungan probabilitas (Tingkat Keyakinan) dari model dalam bentuk persentase dan visualisasi diagram batang.

## Fitur Utama
* **Antarmuka Interaktif:** Pengguna dapat memasukkan parameter data mahasiswa melalui *dropdown* yang intuitif.
* **Analisis Prediktif Real-time:** Hasil prediksi kelulusan langsung keluar setelah tombol analisis ditekan.
* **Metrik Kepercayaan (Confidence Level):** Menampilkan persentase tingkat keyakinan model terhadap prediksi yang diberikan.
* **Visualisasi Probabilitas:** Distribusi probabilitas kelas target ditampilkan dalam bentuk *Bar Chart* yang mudah dibaca.
* **Encoding Otomatis:** Input kategorikal (Jenis Kelamin, Asal SMA, Status Pernikahan, Ukuran Program) secara otomatis diubah menggunakan `LabelEncoder` sebelum diproses oleh model.

## Parameter Input
Model ini menggunakan 4 variabel independen (fitur) untuk melakukan prediksi:
1. **Jenis Kelamin** (Laki-laki / Perempuan)
2. **Asal SMA** (SMA / SMK)
3. **Status Pernikahan** (Belum / Sudah)
4. **Ukuran Program Studi** (Reguler / Ekstensi)

## Cara Menjalankan Aplikasi Lokal

1. **Clone repositori ini**
   ```bash
   git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
   cd nama-repo-kamu
2. Instal dependensi yang dibutuhkan streamlit -> run MachineLearning_DecisionTree_GUI.py
3. Jalankan aplikasi Streamlit -> streamlit run MachineLearning_DecisionTree_GUI.py
4. Buka browser dan akses -> http://localhost:8501

## Metodelogi Penelitian
Sistem ini menggunakan DecisionTreeClassifier(criterion='entropy'). Nilai persentase probabilitas yang ditampilkan pada antarmuka dihitung berdasarkan rasio distribusi instans kelas pada simpul daun (leaf node) tempat data uji direpresentasikan dalam struktur pohon keputusan.

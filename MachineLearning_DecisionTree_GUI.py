import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Sistem Analisis Kelulusan", layout="wide")

st.markdown("""
<style>
.metric-container {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    padding: 15px;
    background-color: #fdfdfd;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.header-title {
    color: #2c3e50;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 10px;
    margin-bottom: 20px;
    font-family: 'Times New Roman', Times, serif;
}
.section-header {
    color: #34495e;
    font-family: 'Times New Roman', Times, serif;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

data = {
    'Jenis Kelamin': ['Laki-laki', 'Laki-laki', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Laki-laki', 'Laki-laki', 'Laki-laki', 'Laki-laki', 'Perempuan'],
    'Asal SMA': ['SMA', 'SMA', 'SMK', 'SMA', 'SMA', 'SMA', 'SMK', 'SMK', 'SMK', 'SMK'],
    'Nikah': ['Belum', 'Belum', 'Belum', 'Belum', 'Belum', 'Belum', 'Sudah', 'Belum', 'Belum', 'Belum'],
    'UkuranProgram': ['Reguler', 'Reguler', 'Ekstensi', 'Reguler', 'Reguler', 'Reguler', 'Ekstensi', 'Ekstensi', 'Reguler', 'Reguler'],
    'Kelulusan': ['Tepat', 'Tepat', 'Terlambat', 'Tepat', 'Tepat', 'Tepat', 'Terlambat', 'Terlambat', 'Tepat', 'Terlambat']
}
df = pd.DataFrame(data)

le_jk = LabelEncoder()
le_sma = LabelEncoder()
le_nikah = LabelEncoder()
le_prog = LabelEncoder()

df['Jenis Kelamin'] = le_jk.fit_transform(df['Jenis Kelamin'])
df['Asal SMA'] = le_sma.fit_transform(df['Asal SMA'])
df['Nikah'] = le_nikah.fit_transform(df['Nikah'])
df['UkuranProgram'] = le_prog.fit_transform(df['UkuranProgram'])

X = df.drop('Kelulusan', axis=1)
y = df['Kelulusan']

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)

st.markdown("<h2 class='header-title'>Sistem Analisis dan Prediksi Kelulusan Mahasiswa Berbasis Decision Tree</h2>", unsafe_allow_html=True)
st.write("Sistem ini mengimplementasikan model klasifikasi Machine Learning (Decision Tree Classifier) untuk melakukan estimasi terhadap status kelulusan mahasiswa. Analisis mencakup hasil keputusan kategorikal serta perhitungan probabilitas keyakinan model secara persentase.")

st.markdown("<h4 class='section-header'>Parameter Input Data Mahasiswa</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    jk_input = st.selectbox("Jenis Kelamin", le_jk.classes_)
    sma_input = st.selectbox("Asal Sekolah Menengah", le_sma.classes_)

with col2:
    nikah_input = st.selectbox("Status Pernikahan", le_nikah.classes_)
    prog_input = st.selectbox("Ukuran Program Studi", le_prog.classes_)

st.write("---")

if st.button("Jalankan Analisis Prediktif", type="primary"):
    input_data = pd.DataFrame([[
        le_jk.transform([jk_input])[0],
        le_sma.transform([sma_input])[0],
        le_nikah.transform([nikah_input])[0],
        le_prog.transform([prog_input])[0]
    ]], columns=X.columns)
    
    prediksi = model.predict(input_data)[0]
    probabilitas = model.predict_proba(input_data)[0]
    
    kelas_target = model.classes_
    prob_tepat = probabilitas[list(kelas_target).index('Tepat')] * 100
    prob_terlambat = probabilitas[list(kelas_target).index('Terlambat')] * 100
    
    st.markdown("<h4 class='section-header'>Hasil Analisis Model</h4>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric(label="Keputusan Klasifikasi", value=f"Lulus {prediksi}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with res_col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        tingkat_keyakinan = prob_tepat if prediksi == 'Tepat' else prob_terlambat
        st.metric(label="Tingkat Keyakinan (Confidence Level)", value=f"{tingkat_keyakinan:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h5 class='section-header'>Distribusi Probabilitas Kelas Target</h5>", unsafe_allow_html=True)
    
    prob_df = pd.DataFrame({
        'Kategori Kelulusan': ['Tepat Waktu', 'Terlambat'],
        'Persentase Probabilitas (%)': [prob_tepat, prob_terlambat]
    })
    
    st.bar_chart(prob_df.set_index('Kategori Kelulusan'))
    
    st.info("Catatan Metodologi: Nilai persentase probabilitas dihitung berdasarkan rasio distribusi instans kelas pada simpul daun (leaf node) tempat data uji direpresentasikan dalam struktur pohon keputusan.")
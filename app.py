
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# =========================
# LOAD MODEL
# =========================

def load_pickle(filename: str):
    file_path = Path(filename)
    if not file_path.is_file():
        st.error(
            f"File model tidak ditemukan: '{filename}'.\n" 
            "Pastikan file model berada di folder yang sama dengan app.py dan coba lagi."
        )
        st.stop()
    try:
        return joblib.load(file_path)
    except Exception as exc:
        st.error(
            f"Gagal memuat '{filename}': {exc}.\n" 
            "Periksa apakah file model valid dan bukan rusak."
        )
        st.stop()

rf_model = load_pickle('rf_model.pkl')
scaler = load_pickle('scaler.pkl')
feature_columns = load_pickle('feature_columns.pkl')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Stroke",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #E63946;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
}
.stButton>button:hover {
    background-color: #d62839;
    color: white;
}
.indicator-box {
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 14px;
}
.indicator-danger {
    background-color: #FCEBEB;
    border-left: 4px solid #E24B4A;
    color: #501313;
}
.indicator-normal {
    background-color: #EAF3DE;
    border-left: 4px solid #639922;
    color: #173404;
}
.indicator-warning {
    background-color: #FAEEDA;
    border-left: 4px solid #BA7517;
    color: #412402;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI ANALISIS INDIKATOR
# Fungsi ini menganalisis setiap nilai input pasien
# dan memberikan keterangan apakah normal atau berisiko
# =========================
def analyze_indicators(age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status):
    """
    Menganalisis indikator kesehatan pasien.
    Mengembalikan list dari dict yang berisi:
    - status: 'danger', 'warning', atau 'normal'
    - icon: emoji indikator
    - label: nama indikator
    - nilai: nilai aktual pasien
    - keterangan: penjelasan kondisi
    - normal: rentang nilai normal
    """
    indicators = []

    # --- Usia ---
    if age >= 75:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Usia Sangat Tinggi",
            "nilai": f"{age} tahun",
            "keterangan": "Usia sangat tua (≥75 tahun) merupakan faktor risiko utama stroke. Risiko stroke meningkat pesat setelah usia 55 tahun.",
            "normal": "Risiko meningkat signifikan di atas 55 tahun"
        })
    elif age >= 60:
        indicators.append({
            "status": "warning",
            "icon": "🟡",
            "label": "Usia Berisiko",
            "nilai": f"{age} tahun",
            "keterangan": "Usia di atas 60 tahun meningkatkan risiko stroke karena pembuluh darah mulai mengalami penurunan elastisitas.",
            "normal": "Risiko lebih rendah pada usia < 55 tahun"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "Usia",
            "nilai": f"{age} tahun",
            "keterangan": "Usia dalam rentang relatif rendah risiko stroke.",
            "normal": "Risiko rendah pada usia < 55 tahun"
        })

    # --- Kadar Glukosa Darah ---
    if avg_glucose_level >= 200:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Kadar Glukosa Sangat Tinggi",
            "nilai": f"{avg_glucose_level:.1f} mg/dL",
            "keterangan": "Kadar glukosa ≥200 mg/dL menandakan diabetes tidak terkontrol. Gula darah tinggi merusak dinding pembuluh darah dan meningkatkan risiko stroke secara signifikan.",
            "normal": "Normal: 70–99 mg/dL | Prediabetes: 100–125 mg/dL | Diabetes: ≥126 mg/dL"
        })
    elif avg_glucose_level >= 126:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Kadar Glukosa Tinggi (Diabetes)",
            "nilai": f"{avg_glucose_level:.1f} mg/dL",
            "keterangan": "Kadar glukosa ≥126 mg/dL mengindikasikan diabetes. Penderita diabetes memiliki risiko stroke 1.5–3 kali lebih tinggi dari orang normal.",
            "normal": "Normal: 70–99 mg/dL (puasa)"
        })
    elif avg_glucose_level >= 100:
        indicators.append({
            "status": "warning",
            "icon": "🟡",
            "label": "Kadar Glukosa Batas (Prediabetes)",
            "nilai": f"{avg_glucose_level:.1f} mg/dL",
            "keterangan": "Kadar glukosa 100–125 mg/dL adalah kondisi prediabetes yang perlu diwaspadai.",
            "normal": "Normal: 70–99 mg/dL (puasa)"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "Kadar Glukosa",
            "nilai": f"{avg_glucose_level:.1f} mg/dL",
            "keterangan": "Kadar glukosa dalam batas normal.",
            "normal": "Normal: 70–99 mg/dL (puasa)"
        })

    # --- BMI (Indeks Massa Tubuh) ---
    if bmi >= 35:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "BMI Obesitas Berat",
            "nilai": f"{bmi:.1f} kg/m²",
            "keterangan": "BMI ≥35 (obesitas berat) meningkatkan tekanan pada jantung dan pembuluh darah, risiko pembentukan gumpalan darah, dan risiko stroke.",
            "normal": "Normal: 18.5–24.9 | Overweight: 25–29.9 | Obesitas: ≥30"
        })
    elif bmi >= 30:
        indicators.append({
            "status": "warning",
            "icon": "🟡",
            "label": "BMI Obesitas",
            "nilai": f"{bmi:.1f} kg/m²",
            "keterangan": "BMI ≥30 (obesitas) berkaitan dengan tekanan darah tinggi dan kolesterol, yang merupakan faktor risiko stroke.",
            "normal": "Normal: 18.5–24.9 kg/m²"
        })
    elif bmi >= 25:
        indicators.append({
            "status": "warning",
            "icon": "🟡",
            "label": "BMI Overweight",
            "nilai": f"{bmi:.1f} kg/m²",
            "keterangan": "BMI 25–29.9 (overweight) sedikit meningkatkan risiko penyakit kardiovaskular.",
            "normal": "Normal: 18.5–24.9 kg/m²"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "BMI",
            "nilai": f"{bmi:.1f} kg/m²",
            "keterangan": "BMI dalam rentang normal.",
            "normal": "Normal: 18.5–24.9 kg/m²"
        })

    # --- Hipertensi ---
    if hypertension == 1:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Hipertensi (Tekanan Darah Tinggi)",
            "nilai": "Positif",
            "keterangan": "Hipertensi adalah faktor risiko terbesar stroke. Tekanan darah tinggi merusak dan melemahkan pembuluh darah otak, sehingga mudah pecah atau tersumbat.",
            "normal": "Normal: Tekanan darah < 120/80 mmHg"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "Hipertensi",
            "nilai": "Tidak Ada",
            "keterangan": "Tidak terdeteksi hipertensi.",
            "normal": "Normal: Tekanan darah < 120/80 mmHg"
        })

    # --- Penyakit Jantung ---
    if heart_disease == 1:
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Penyakit Jantung",
            "nilai": "Positif",
            "keterangan": "Riwayat penyakit jantung (seperti atrial fibrilasi atau gagal jantung) meningkatkan risiko stroke karena gumpalan darah dari jantung dapat masuk ke aliran darah otak.",
            "normal": "Tidak ada riwayat penyakit jantung = risiko lebih rendah"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "Penyakit Jantung",
            "nilai": "Tidak Ada",
            "keterangan": "Tidak terdeteksi penyakit jantung.",
            "normal": "Tidak ada riwayat penyakit jantung = risiko lebih rendah"
        })

    # --- Status Merokok ---
    if smoking_status == "smokes":
        indicators.append({
            "status": "danger",
            "icon": "🔴",
            "label": "Perokok Aktif",
            "nilai": "Merokok",
            "keterangan": "Merokok aktif meningkatkan risiko stroke 2–4 kali lipat. Nikotin dan karbon monoksida merusak dinding pembuluh darah dan meningkatkan pembekuan darah.",
            "normal": "Tidak merokok = risiko lebih rendah"
        })
    elif smoking_status == "formerly smoked":
        indicators.append({
            "status": "warning",
            "icon": "🟡",
            "label": "Mantan Perokok",
            "nilai": "Pernah Merokok",
            "keterangan": "Mantan perokok masih memiliki risiko lebih tinggi dari yang tidak pernah merokok, namun risiko berkurang seiring waktu berhenti merokok.",
            "normal": "Tidak pernah merokok = risiko paling rendah"
        })
    else:
        indicators.append({
            "status": "normal",
            "icon": "🟢",
            "label": "Status Merokok",
            "nilai": "Tidak Merokok / Tidak Diketahui",
            "keterangan": "Tidak terdeteksi kebiasaan merokok aktif.",
            "normal": "Tidak merokok = risiko lebih rendah"
        })

    return indicators


# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center; color:#E63946;'>
🩺 Prediksi Risiko Stroke
</h1>
<p style='text-align:center; font-size:18px;'>
Sistem Prediksi Stroke Menggunakan Machine Learning Random Forest
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================
# OPTIONS
# =========================
gender_options = ['Male', 'Female', 'Other']
ever_married_options = ['Yes', 'No']
work_type_options = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
residence_type_options = ['Urban', 'Rural']
smoking_status_options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

# =========================
# FORM INPUT
# =========================
st.subheader("📋 Informasi Pasien")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Jenis Kelamin', gender_options)
    age = st.slider('Usia', 0, 100, 30)
    hypertension = st.radio('Hipertensi', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    heart_disease = st.radio('Penyakit Jantung', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    ever_married = st.selectbox('Sudah Menikah', ever_married_options)

with col2:
    work_type = st.selectbox('Tipe Pekerjaan', work_type_options)
    residence_type = st.selectbox('Tipe Tempat Tinggal', residence_type_options)
    avg_glucose_level = st.number_input('Kadar Glukosa (mg/dL)', min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input('BMI (kg/m²)', min_value=10.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox('Status Merokok', smoking_status_options)

st.divider()

# =========================
# PREPROCESSING INPUT
# =========================
user_input = {
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}

input_df_processed = pd.DataFrame(0, index=[0], columns=feature_columns)

input_df_processed['age'] = float(user_input['age'])
input_df_processed['hypertension'] = int(user_input['hypertension'])
input_df_processed['heart_disease'] = int(user_input['heart_disease'])
input_df_processed['avg_glucose_level'] = float(user_input['avg_glucose_level'])
input_df_processed['bmi'] = float(user_input['bmi'])

for gender_opt in gender_options:
    col_name = f"gender_{gender_opt}"
    if col_name in input_df_processed.columns:
        input_df_processed[col_name] = 1 if user_input['gender'] == gender_opt else 0

for married_opt in ever_married_options:
    col_name = f"ever_married_{married_opt}"
    if col_name in input_df_processed.columns:
        input_df_processed[col_name] = 1 if user_input['ever_married'] == married_opt else 0

for work_opt in work_type_options:
    col_name = f"work_type_{work_opt}"
    if col_name in input_df_processed.columns:
        input_df_processed[col_name] = 1 if user_input['work_type'] == work_opt else 0

for residence_opt in residence_type_options:
    col_name = f"Residence_type_{residence_opt}"
    if col_name in input_df_processed.columns:
        input_df_processed[col_name] = 1 if user_input['Residence_type'] == residence_opt else 0

for smoking_opt in smoking_status_options:
    col_name = f"smoking_status_{smoking_opt}"
    if col_name in input_df_processed.columns:
        input_df_processed[col_name] = 1 if user_input['smoking_status'] == smoking_opt else 0

numerical_cols_for_scaler = ['age', 'avg_glucose_level', 'bmi']
df_to_scale = input_df_processed[numerical_cols_for_scaler].copy()
scaled_data = scaler.transform(df_to_scale)
input_df_processed[numerical_cols_for_scaler] = scaled_data
final_input_for_model = input_df_processed

# =========================
# PREDICTION & TAMPILAN HASIL
# =========================
if st.button('🔍 Prediksi Risiko Stroke'):

    prediction = rf_model.predict(final_input_for_model)
    prediction_proba = rf_model.predict_proba(final_input_for_model)
    prob_no_stroke = prediction_proba[0][0]
    prob_stroke = prediction_proba[0][1]

    # --- Hasil utama ---
    st.subheader("📊 Hasil Prediksi")

    if prediction[0] == 1:
        st.error("⚠️ **Pasien Berpotensi Mengalami Stroke**")
    else:
        st.success("✅ **Pasien Tidak Berpotensi Stroke**")

    # --- Probabilitas ---
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric(label="Probabilitas Tidak Stroke", value=f"{prob_no_stroke*100:.1f}%")
    with col_p2:
        st.metric(label="Probabilitas Stroke", value=f"{prob_stroke*100:.1f}%")

    st.progress(float(prob_stroke), text=f"Tingkat risiko stroke: {prob_stroke*100:.1f}%")

    st.divider()

    # --- Analisis Indikator ---
    st.subheader("🔎 Analisis Detail Indikator Kesehatan Pasien")
    st.caption("Berikut penjelasan kondisi setiap indikator kesehatan pasien berdasarkan input yang diberikan:")

    indicators = analyze_indicators(
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        avg_glucose_level=avg_glucose_level,
        bmi=bmi,
        smoking_status=smoking_status
    )

    # Tampilkan indikator bahaya dulu, lalu warning, lalu normal
    for status_filter in ["danger", "warning", "normal"]:
        filtered = [i for i in indicators if i["status"] == status_filter]
        for ind in filtered:
            css_class = f"indicator-{ind['status']}"
            st.markdown(f"""
            <div class="{css_class} indicator-box">
                <strong>{ind['icon']} {ind['label']}: {ind['nilai']}</strong><br>
                 {ind['keterangan']}<br>
                <span style='font-size:12px; opacity:0.8;'>📏 Acuan: {ind['normal']}</span>
            </div>
            """, unsafe_allow_html=True)

    # --- Ringkasan Faktor Risiko ---
    st.divider()
    danger_count = sum(1 for i in indicators if i["status"] == "danger")
    warning_count = sum(1 for i in indicators if i["status"] == "warning")

    st.subheader("📋 Ringkasan Faktor Risiko")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("🔴 Indikator Bahaya", f"{danger_count} faktor")
    with col_s2:
        st.metric("🟡 Indikator Waspada", f"{warning_count} faktor")
    with col_s3:
        normal_count = len(indicators) - danger_count - warning_count
        st.metric("🟢 Indikator Normal", f"{normal_count} faktor")

    if danger_count >= 3:
        st.warning("⚠️ Pasien memiliki banyak faktor risiko berbahaya. Disarankan segera konsultasi ke dokter spesialis.")
    elif danger_count >= 1:
        st.info("ℹ️ Pasien memiliki beberapa faktor risiko yang perlu mendapat perhatian medis.")
    else:
        st.success("✅ Tidak ada indikator berbahaya. Tetap jaga gaya hidup sehat!")

st.divider()
st.caption("Machine Learning Stroke Prediction System | Random Forest | Tugas AI")

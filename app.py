
import streamlit as st
import numpy as np
import joblib
import torch
import re
import pandas as pd


from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# KONFIGURASI (DRIVE)
# =========================
BASE_PATH = "/content/drive/MyDrive/Praktikum/UAP/model"
ENCODER_PATH = "/content/drive/MyDrive/Praktikum/UAP/encoder"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Sentiment Analysis UAP",
    layout="centered"
)

st.title("Sentiment Analysis Ulasan Game (UAP)")
st.write("Analisis sentimen ulasan game **Clash Royale** Bahasa Indonesia")

# =========================
# LOAD LABEL ENCODER
# =========================
@st.cache_resource
def load_label_encoder():
    return joblib.load(f"{ENCODER_PATH}/label_encoder.pkl")

label_encoder = load_label_encoder()

# =========================
# LOAD MODELS (DI AWAL)
# =========================
@st.cache_resource
def load_indobert_model():
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/indobert")
    model = AutoModelForSequenceClassification.from_pretrained(f"{BASE_PATH}/indobert")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_distilbert_model():
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/distilbert")
    model = AutoModelForSequenceClassification.from_pretrained(f"{BASE_PATH}/distilbert")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# LOAD SEKALI SAAT APP DIBUKA
with st.spinner("Memuat model..."):
    tokenizer_indobert, model_indobert = load_indobert_model()
    tokenizer_distilbert, model_distilbert = load_distilbert_model()

st.success("Model berhasil dimuat")

# =========================
# PREPROCESS TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

# =========================
# UI INPUT
# =========================
model_choice = st.selectbox(
    "Pilih Model",
    ["IndoBERT", "DistilBERT"]
)

user_text = st.text_area(
    "Masukkan Ulasan Game",
    height=120,
    placeholder="Contoh: game ini sangat seru dan grafiknya bagus..."
)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Sentimen"):

    if user_text.strip() == "":
        st.warning("⚠️ Ulasan tidak boleh kosong")
    else:
        clean = clean_text(user_text)

        # PILIH MODEL YANG SUDAH DI-LOAD
        if model_choice == "IndoBERT":
            tokenizer = tokenizer_indobert
            model = model_indobert
        else:
            tokenizer = tokenizer_distilbert
            model = model_distilbert

        inputs = tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

        label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        # =========================
        # OUTPUT
        # =========================
        
        st.success(f"**Prediksi Sentimen:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # =========================
        # GRAFIK PROBABILITAS
        # =========================
        st.subheader("Grafik Probabilitas Sentimen")

        prob_df = pd.DataFrame({
            "Sentimen": label_encoder.classes_,
            "Probabilitas (%)": probs * 100
        })

        st.bar_chart(
            prob_df.set_index("Sentimen"),
            height=300
        )

        # =========================
        # DETAIL ANGKA
        # =========================
        st.subheader("Probabilitas Detail")
        for i, cls in enumerate(label_encoder.classes_):
            st.write(f"- **{cls}**: {probs[i]*100:.2f}%")



import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# LOAD MODEL (DARI GDRIVE)
# =========================
MODEL_PATH = "rice_leaf_disease_cnn_model.keras"
FILE_ID = "1OVBofQX8flV7DK_69MMZya6MAirLkwpv"

@st.cache_resource
def load_model():

    # Download model jika belum ada
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={1OVBofQX8flV7DK_69MMZya6MAirLkwpv}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    return model
labels = ["Healthy", "Blas", "Hawar Daun", "Tungro"]

# =========================
# UI
# =========================
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")

st.markdown("""
    <style>
    .title {text-align:center;font-size:36px;font-weight:bold;color:#2c3e50;}
    .subtitle {text-align:center;font-size:18px;color:#7f8c8d;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌾 Rice Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload gambar daun padi untuk mendeteksi penyakit</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Gambar Daun Padi", type=["jpg","jpeg","png"])

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    image = image.resize((299, 299))  # InceptionV3
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar Input", use_column_width=True)

    with col2:
        st.write("🔍 Menganalisis...")

        img_array = preprocess_image(image)
        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        st.success(f"✅ {labels[predicted_class]}")
        st.metric("Confidence", f"{confidence:.2%}")
        st.progress(float(confidence))

        st.write("### Detail Probabilitas:")
        for i, label in enumerate(labels):
            st.write(f"{label}: {predictions[0][i]:.2%}")

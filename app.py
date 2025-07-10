import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("deepfake_detector_model.h5")

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🧠 Deepfake Image Detector")
st.write("Upload a face image to check if it's REAL or FAKE")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "REAL" if prediction > 0.5 else "FAKE"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader(f"🧾 Prediction: **{label}**")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: `{confidence*100:.2f}%`")

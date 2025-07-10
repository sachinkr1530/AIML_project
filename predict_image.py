from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("deepfake_detector_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "REAL" if prediction > 0.5 else "FAKE"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

predict_image("real.png")

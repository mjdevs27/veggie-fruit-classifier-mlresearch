import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, UnidentifiedImageError

# -------------------
# STREAMLIT CONFIG  (must be the first Streamlit command)
# -------------------
st.set_page_config(page_title="AgriVision Classifier", page_icon="🍎")

# -------------------
# CONFIG
# -------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "agrivision_resnet_best.keras"
CLASS_NAMES_PATH = "class_names.json"

# -------------------
# LOAD MODEL & CLASSES
# -------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)
    # handle list or dict
    if isinstance(data, dict):
        try:
            max_idx = max(data.values())
            class_list = [""] * (max_idx + 1)
            for name, idx in data.items():
                class_list[idx] = name
            return class_list
        except Exception:
            return list(data.keys())
    return data

model = load_model()
class_names = load_class_names()
num_classes = model.output_shape[-1]

# -------------------
# PREDICTION FUNCTION
# -------------------
def predict_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]  # (num_classes,)
    idx = int(np.argmax(preds))

    if isinstance(class_names, list) and idx < len(class_names):
        pred_class = class_names[idx]
    else:
        pred_class = f"class_{idx}"

    confidence = float(preds[idx]) * 100.0
    return pred_class, confidence, preds

# -------------------
# STREAMLIT UI
# -------------------
st.title(" AgriVision – Fruit & Vegetable Classifier")
st.write("Upload a fruit/vegetable image to get the predicted class and confidence score.")

uploaded = st.file_uploader(
    "Upload a fruit/vegetable image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG.")
        st.stop()

    # IMPORTANT: use_column_width works on Streamlit 1.37.0 (Cloud)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            with st.spinner("Classifying..."):
                label, conf, preds = predict_image(image)

            st.success(f"Predicted Class: **{label}**")
            st.write(f"Confidence: **{conf:.2f}%**")

            st.subheader("Top 3 probabilities")
            top_k = min(3, len(preds))
            top_idx = np.argsort(preds)[::-1][:top_k]
            for i in top_idx:
                cname = class_names[i] if isinstance(class_names, list) and i < len(class_names) else f"class_{i}"
                st.write(f"- {cname}: {preds[i] * 100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info(" Click **Browse files** to pick an image from your file explorer.")

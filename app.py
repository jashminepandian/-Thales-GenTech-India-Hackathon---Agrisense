import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
import json

# -----------------------------
# ✅ Streamlit page settings
# -----------------------------
st.set_page_config(
    page_title="AgriSense - Crop Monitor",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------
# ✅ Load your trained model
# -----------------------------
@st.cache_resource(show_spinner="⏳ Loading trained leaf model…")
def load_model():
    # load your keras model
    model = keras.models.load_model("models/leaf_model.keras")

    # load class names from json
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names

model, class_names = load_model()

# -----------------------------
# ✅ Defects dictionary
# -----------------------------
DEFECTS = {
    "Bacterial_spot": {
        "desc": "Dark, water-soaked spots spreading on the leaf surface.",
        "remedy": "Use copper-based sprays, remove heavily infected leaves, avoid overhead watering."
    },
    "Rust": {
        "desc": "Orange-brown pustules on the underside of leaves.",
        "remedy": "Apply preventive fungicides early, use resistant varieties, and rotate crops."
    },
    "Early_blight": {
        "desc": "Concentric brown rings on lower leaves.",
        "remedy": "Spray mancozeb/chlorothalonil, remove old debris, improve airflow."
    },
    "Late_blight": {
        "desc": "Large, irregular dark lesions often with white mold under humid conditions.",
        "remedy": "Remove infected plants quickly, apply systemic fungicides."
    },
    "Leaf_Mold": {
        "desc": "Yellow patches on top and olive-green mold beneath the leaves.",
        "remedy": "Improve ventilation, reduce humidity, apply protective sprays."
    },
    "Septoria": {
        "desc": "Circular brown spots with gray centers on older leaves.",
        "remedy": "Remove spotted leaves, apply fungicides, avoid leaf wetness."
    },
    "Healthy": {
        "desc": "No major issues detected. Leaf looks healthy.",
        "remedy": "Maintain proper irrigation, fertilization, and monitoring."
    }
}

# -----------------------------
# ✅ Helper: map class → defect
# -----------------------------
def map_to_defect(label: str):
    for defect in DEFECTS.keys():
        if defect.lower() in label.lower():
            return defect
    return "Healthy"

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("🌱 AgriSense - Smart Crop Monitor")
st.write("Upload a leaf image to detect defects and get remedy suggestions.")

uploaded = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Show uploaded image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # Preprocess for your model
    img_array = np.array(img.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    st.write("🔎 Analyzing…")
    preds = model.predict(img_array)
    top_idx = np.argmax(preds)
    top_label = class_names[top_idx]
    top_score = float(np.max(preds))

    # Show prediction
    st.subheader("📊 Prediction")
    st.write(f"- **{top_label}** ({top_score:.2f})")

    # Suggest remedy
    defect = map_to_defect(top_label)

    st.subheader("🌿 Detected Defect")
    st.info(DEFECTS[defect]["desc"])

    st.subheader("💡 Suggested Remedy")
    st.success(DEFECTS[defect]["remedy"])

else:
    st.info("👉 Upload a clear, close-up photo of a single leaf against a plain background for best results.")

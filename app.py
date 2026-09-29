import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_disease_model.keras")

model = load_model()

# Class names
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Title
st.title("🍅 Tomato Disease Detection")
st.write("Upload a tomato leaf image to detect possible diseases using Machine Learning.")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        use_container_width=True
    )

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    # Clean disease name
    disease_name = predicted_class.replace("Tomato___", "").replace("_", " ")

    st.divider()

    st.subheader("🔍 Prediction Result")

    if disease_name.lower() == "healthy":
        st.success("✅ The tomato leaf appears to be HEALTHY.")
    else:
        st.warning(f"⚠️ Possible Disease: {disease_name}")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(min(confidence / 100, 1.0))

    st.caption(
        "Note: This is an AI-based prediction and should not be considered a professional agricultural diagnosis."
    )

else:
    st.info("👆 Please upload a tomato leaf image to start prediction.")
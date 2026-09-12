import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="centered"
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "fruit_freshness_mobilenetv2.h5"
    )
    return model


model = load_model()


# -----------------------------------
# CLASS NAMES
# -----------------------------------

class_names = [
    "fresh_peaches_done",
    "fresh_pomegranates_done",
    "fresh_strawberries_done",
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rotten_peaches_done",
    "rotten_pomegranates_done",
    "rotten_strawberries_done",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]


# -----------------------------------
# TITLE
# -----------------------------------

st.title("🍎 Fruit Freshness Detector")

st.write(
    "Upload an image of a fruit and the AI model will predict "
    "whether the fruit is fresh or rotten."
)


# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Freshness"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

        # Prediction
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        predicted_class = class_names[predicted_index]

        # Clean class name
        display_name = predicted_class.replace("_done", "")
        display_name = display_name.replace("_", " ")
        display_name = display_name.title()


        # -----------------------------------
        # RESULT
        # -----------------------------------

        st.subheader("Prediction Result")

        if "fresh" in predicted_class.lower():

            st.success(
                f"🟢 Fresh Fruit Detected: {display_name}"
            )

        else:

            st.error(
                f"🔴 Rotten Fruit Detected: {display_name}"
            )

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )

        st.progress(int(confidence))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="MRI / X-Ray Enhancer", layout="wide")

IMG_SIZE = 256
UPSCALE_FACTOR = 4
INPUT_SIZE = IMG_SIZE // UPSCALE_FACTOR


# ==========================================================
# SESSION STATE
# ==========================================================
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# ==========================================================
# CUSTOM PIXEL SHUFFLE
# ==========================================================
class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models():
    classifier = tf.keras.models.load_model("mri_xray_classifier.h5")

    xray_sr = tf.keras.models.load_model(
        "best_espcn_model1.keras",
        compile=False,
        custom_objects={"PixelShuffle": PixelShuffle}
    )

    mri_sr = tf.keras.models.load_model(
        "best_espcn_model2 (1).keras",
        compile=False,
        custom_objects={"PixelShuffle": PixelShuffle}
    )

    return classifier, xray_sr, mri_sr


classifier, xray_sr, mri_sr = load_models()


# ==========================================================
# PREPROCESS
# ==========================================================
def preprocess_classifier(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def enhance_image(model, image):
    img = image.convert("L")
    hr = img.resize((IMG_SIZE, IMG_SIZE))
    lr = hr.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)

    lr_array = np.array(lr) / 255.0
    lr_array = np.expand_dims(lr_array, axis=(0, -1))

    sr = model.predict(lr_array, verbose=0)[0]
    sr = np.clip(sr, 0, 1)

    sr_img = (sr.squeeze() * 255).astype(np.uint8)
    return sr_img


# ==========================================================
# SKELETON HIGHLIGHT FUNCTION (FINAL TUNED)
# ==========================================================
def highlight_region(image, region):
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size

    if region == "head":
        # 🔹 Moved UP + Smaller (tight skull highlight)
        draw.ellipse(
            [(w*0.44, h*0.01), (w*0.56, h*0.12)],
            fill=(255, 0, 0, 120)
        )

    elif region == "chest":
        # 🔹 Proper lung region
        draw.ellipse(
            [(w*0.35, h*0.18), (w*0.65, h*0.38)],
            fill=(255, 0, 0, 120)
        )

    return Image.alpha_composite(img, overlay)


# ==========================================================
# MAIN PAGE INPUT
# ==========================================================
st.title("🧠 MRI / X-Ray Enhancer")

if not st.session_state.submitted:

    st.subheader("Enter Patient Details")

    patient_name = st.text_input("Patient Name")
    patient_gender = st.selectbox(
        "Gender",
        ["Select Gender", "Male", "Female", "Other"]
    )

    uploaded_file = st.file_uploader(
        "Upload MRI or X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Submit"):
        if uploaded_file is not None and patient_gender != "Select Gender" and patient_name != "":
            st.session_state.submitted = True
            st.session_state.patient_name = patient_name
            st.session_state.patient_gender = patient_gender
            st.session_state.uploaded_file = uploaded_file
            st.rerun()
        else:
            st.warning("Please fill all details and upload image.")


# ==========================================================
# AFTER SUBMISSION
# ==========================================================
else:

    # Sidebar
    st.sidebar.title("📝 Patient Information")
    st.sidebar.write(f"**Name:** {st.session_state.patient_name}")
    st.sidebar.write(f"**Gender:** {st.session_state.patient_gender}")

    image = Image.open(st.session_state.uploaded_file)

    # Classification
    input_img = preprocess_classifier(image)
    prediction = classifier.predict(input_img, verbose=0)[0][0]

    if prediction > 0.5:
        label = "MRI"
        confidence = float(prediction)
        sr_model = mri_sr
        region = "head"
    else:
        label = "X-Ray"
        confidence = float(1 - prediction)
        sr_model = xray_sr
        region = "chest"

    # ===============================
    # RESULT HEADER
    # ===============================
    st.subheader("🧾 Analysis Result")

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.4f}")

    st.divider()

    # ===============================
    # TWO COLUMN LAYOUT
    # ===============================
    left_col, right_col = st.columns([1, 1], gap="large")

    # LEFT COLUMN
    with left_col:

        st.markdown("### Original Image")
        st.image(image, width=350)

        st.markdown("### Enhanced Image (ESPCN)")
        with st.spinner("Enhancing image quality..."):
            sr_img = enhance_image(sr_model, image)

        st.image(sr_img, width=350)

    # RIGHT COLUMN
    with right_col:

        st.markdown("### Detected Body Region")
        skeleton = Image.open("human-skeletal-system.jpg")
        highlighted = highlight_region(skeleton, region)

        st.image(highlighted, width=350)

    # Reset
    if st.sidebar.button("New Patient"):
        st.session_state.submitted = False
        st.rerun()
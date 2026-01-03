import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# Background Image Function
# -----------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        /* Remove default padding & margins */
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .glass {{
            background: rgba(0, 0, 0, 0.85);
            padding: 25px;
            border-radius: 18px;
            text-align: center;
            color: white;
            margin-bottom: 15px;
        }}

        .result-good {{
            font-size: 48px;
            font-weight: bold;
            color: #00FF7F;
        }}

        .result-avg {{
            font-size: 48px;
            font-weight: bold;
            color: #FFD700;
        }}

        .result-poor {{
            font-size: 48px;
            font-weight: bold;
            color: #FF4500;
        }}

        /* Remove extra spacing from markdown */
        .stMarkdown {{
            margin-bottom: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background image
add_bg_from_local("wine.png")

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_RFmodel.sav", "rb"))
    scaler = pickle.load(open("scaler_model.sav", "rb"))
    return model, scaler

RF_model, scaler = load_model()

# -----------------------------
# App Title
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h1>🍷 Wine Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p>Predict wine quality using <b>Machine Learning</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 Wine Parameters")

def user_input_features():
    return pd.DataFrame([{
        'fixed acidity': st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.8),
        'volatile acidity': st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.9),
        'citric acid': st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.1),
        'residual sugar': st.sidebar.slider("Residual Sugar (log)", -2.0, 3.0, 0.74),
        'chlorides': st.sidebar.slider("Chlorides (log)", -5.0, 1.0, 1.9),
        'free sulfur dioxide': st.sidebar.slider("Free Sulfur Dioxide (log)", -2.0, 5.0, 1.56),
        'total sulfur dioxide': st.sidebar.slider("Total Sulfur Dioxide (log)", 0.0, 6.0, 4.6),
        'density': st.sidebar.slider("Density", 0.990, 1.005, 1.000),
        'pH': st.sidebar.slider("pH", 2.5, 4.5, 3.3),
        'sulphates': st.sidebar.slider("Sulphates (log)", -2.0, 3.0, 0.7),
        'alcohol': st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)
    }])

user_df = user_input_features()

# -----------------------------
# Display Inputs
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h3>📊 Input Parameters</h3>", unsafe_allow_html=True)
st.dataframe(user_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction Button
# -----------------------------
predict = st.button("🍷 Predict Wine Quality", use_container_width=True)

# -----------------------------
# Prediction Output
# -----------------------------
if predict:
    with st.spinner("🍇 Analyzing wine quality..."):
        time.sleep(2)
        user_scaled = scaler.transform(user_df)
        prediction = RF_model.predict(user_scaled)
        predicted_quality = int(np.round(prediction[0]))

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<h3>🍷 Prediction Result</h3>", unsafe_allow_html=True)

    if predicted_quality <= 4:
        st.markdown(
            f"<div class='result-poor'>Quality: {predicted_quality}<br>Poor 🍂</div>",
            unsafe_allow_html=True
        )
    elif predicted_quality <= 6:
        st.markdown(
            f"<div class='result-avg'>Quality: {predicted_quality}<br>Average 🍷</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-good'>Quality: {predicted_quality}<br>Good ⭐</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<div class='glass'>Developed with ❤️ using Streamlit & Machine Learning</div>",
    unsafe_allow_html=True
)

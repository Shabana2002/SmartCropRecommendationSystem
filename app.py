# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
import os

# ------------------------------
# Paths for model and label encoder
# ------------------------------
MODEL_PATH = os.path.join("model", "crop_model.pkl")
LE_PATH = os.path.join("model", "label_encoder.pkl")


# ------------------------------
# Load model and label encoder once
# ------------------------------
@st.cache_data
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.error(
            "Model or label encoder not found! Make sure the 'model/' folder with .pkl files is in the repo."
        )
        return None, None
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le


model, le = load_model()


# ------------------------------
# Predict function
# ------------------------------
def predict_top3(input_features):
    if model is None or le is None:
        return []

    # Reshape input for prediction
    input_array = np.array(input_features).reshape(1, -1)

    # Get probabilities
    probs = model.predict_proba(input_array)

    # Top 3 indices
    top3_indices = np.argsort(probs[0])[-3:][::-1]

    # Convert indices back to crop names
    top3_crops = le.inverse_transform(top3_indices)
    return top3_crops


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌾 Smart Crop Recommendation System")
st.write("Enter your soil and weather details to get the top 3 crop recommendations.")

# User inputs
N = st.number_input("Nitrogen content in soil (N)", min_value=0.0)
P = st.number_input("Phosphorus content in soil (P)", min_value=0.0)
K = st.number_input("Potassium content in soil (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Predict Crops"):
    features = [N, P, K, temperature, humidity, ph, rainfall]
    top_crops = predict_top3(features)

    if top_crops is not None and len(top_crops) > 0:
        st.success("Top 3 recommended crops:")
        for i, crop in enumerate(top_crops, start=1):
            st.write(f"{i}. {crop}")
    else:
        st.warning("Could not predict crops. Check the model and input values.")

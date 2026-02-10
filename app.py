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
def predict_top3(input_f

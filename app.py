import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

# ------------------------------
# Load model and label encoder once
# ------------------------------
@st.cache_data
def load_model():
    model = joblib.load("model/crop_model.pkl")
    le = joblib.load("model/label_encoder.pkl")
    return model, le

model, le = load_model()

# ------------------------------
# Predict function
# ------------------------------
def predict_top3(input_features):
    probs = model.predict_proba([input_features])[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = le.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]
    return list(zip(top3_crops, top3_probs)), probs

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="wide")
st.title("🌾 Smart Crop Recommendation System")
st.write("Enter your soil and weather details to get crop recommendations.")

# ------------------------------
# Sidebar Inputs
# ------------------------------
with st.sidebar:
    st.header("Input Parameters")

    with st.expander("🌱 Soil Nutrients", expanded=True):
        N = st.slider("Nitrogen (N) 🌱", 0, 140, 90)
        P = st.slider("Phosphorus (P) 🌿", 0, 140, 42)
        K = st.slider("Potassium (K) 🍃", 0, 140, 43)

    with st.expander("🌡️ Weather Parameters", expanded=True):
        temperature = st.slider("Temperature (°C) 🌡️", 0, 50, 30)
        humidity = st.slider("Humidity (%) 💧", 0, 100, 85)
        ph = st.slider("Soil pH ⚗️", 0.0, 14.0, 6.5, 0.1)
        rainfall = st.slider("Rainfall (mm) 🌧️", 0.0, 500.0, 200.0, 1.0)

features = [N, P, K, temperature, humidity, ph, rainfall]

# ------------------------------
# Predict button
# ------------------------------
if st.button("Recommend Crops"):
    top3, all_probs = predict_top3(features)

    # Highlight best crop
    best_crop, best_prob = top3[0]
    st.success(f"🌟 Best Crop Recommendation: {best_crop} ({best_prob * 100:.2f}% probability)")

    # Top 3 display
    st.subheader("Top 3 Recommended Crops:")
    for crop, prob in top3:
        st.write(f"{crop} - {prob * 100:.2f}% probability")

    # Bar chart for top 3
    df_top3 = pd.DataFrame(top3, columns=["Crop", "Probability"])
    df_top3["Probability"] *= 100
    chart = alt.Chart(df_top3).mark_bar(color="#4CAF50").encode(
        x="Crop",
        y="Probability"
    )
    st.altair_chart(chart, use_container_width=True)

    # NPK visual chart
    st.subheader("🌱 Soil Nutrient Levels (NPK)")
    df_npk = pd.DataFrame({
        "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
        "Value": [N, P, K]
    })
    npk_chart = alt.Chart(df_npk).mark_bar(color="#FFA500").encode(
        x="Nutrient",
        y="Value"
    )
    st.altair_chart(npk_chart, use_container_width=True)

    # Fertilizer suggestions
    st.subheader("💡 Fertilizer Suggestions")
    if N < 50 or P < 30 or K < 30:
        st.info("Consider adding fertilizer to balance NPK levels.")
    else:
        st.success("NPK levels are optimal ✅")

    # All crops table
    st.subheader("All Crop Probabilities:")
    all_crops = le.inverse_transform(np.arange(len(all_probs)))
    df_all = pd.DataFrame({"Crop": all_crops, "Probability": all_probs * 100})
    df_all = df_all.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    st.dataframe(df_all)

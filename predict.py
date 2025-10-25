import joblib
model = joblib.load("model/crop_model.pkl")
le = joblib.load("model/label_encoder.pkl")
def predict_top3(input_features):
    probs = model.predict_proba([input_features])[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = le.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]
    return list(zip(top3_crops, top3_probs))

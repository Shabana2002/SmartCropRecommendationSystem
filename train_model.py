import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv("data/crop_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "model/label_encoder.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "model/crop_model.pkl")
accuracy = model.score(X_test, y_test)
print(f"Model trained. Test Accuracy: {accuracy*100:.2f}%")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv("data.csv")

X = data[["speed", "acceleration", "braking", "turn_rate"]]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)
# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")
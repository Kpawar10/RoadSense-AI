import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# PAGE CONFIG (APP-LIKE UI)
# -----------------------------
st.set_page_config(page_title="Smart Rider Score", layout="centered")

st.title("🚗 Smart Rider Score")
st.caption("Real-time AI-based driving safety assistant")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
features = ["speed", "acceleration", "turn_rate", "braking"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# SESSION STATE (REAL-TIME FEEL)
# -----------------------------
if "speed_history" not in st.session_state:
    st.session_state.speed_history = []

if "score_history" not in st.session_state:
    st.session_state.score_history = []
st.subheader("📂 Upload Ride Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    csv_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", csv_data.head())

    st.subheader("📊 Processing Uploaded Data")

    scores = []

    for _, row in csv_data.iterrows():
        score = int((row["speed"]/120)*30 +
                    (row["acceleration"]/5)*20 +
                    (row["braking"]/5)*30 +
                    (row["turn_rate"]/5)*20)
        scores.append(score)

    csv_data["Risk Score"] = scores

    st.line_chart(csv_data[["Risk Score"]])  
# -----------------------------
# INPUT SECTION (CLEAN UI)
# -----------------------------
st.subheader("📥 Enter Ride Data")
   
col1, col2 = st.columns(2)
  
with col1:
    speed = st.slider("Speed (km/h)", 0, 120, 50)
    acceleration = st.slider("Acceleration", 0.0, 5.0, 1.0)

with col2:
    turning = st.slider("Turning Intensity", 0.0, 5.0, 1.0)
    braking = st.slider("Braking Intensity", 0.0, 5.0, 1.0)

st.caption("Simulated real-time sensor data (speed, acceleration, braking, turning)")

input_data = pd.DataFrame([[speed, acceleration, turning, braking]], columns=features)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0]

# -----------------------------
# RISK SCORE
# -----------------------------
risk_score = int((speed/120)*30 + (acceleration/5)*20 + (braking/5)*30 + (turning/5)*20)

if risk_score < 40:
    category = "🟢 Safe"

elif risk_score < 70:
    category = "🟡 Risky"

else:
    category = "🔴 Dangerous"

# -----------------------------
# SAVE HISTORY
# -----------------------------
st.session_state.speed_history.append(speed)
st.session_state.score_history.append(risk_score)

# -----------------------------
# OUTPUT (PRODUCT STYLE)
# -----------------------------
st.markdown("## 🎯 Your Driving Score")

st.markdown(f"# {risk_score}/100")
st.markdown(f"### {category}")

# Alerts
if risk_score < 40:
    st.success("✅ Safe Driving")

elif risk_score < 70:
    st.warning("⚠️ Drive Carefully")

else:
    st.error("🚨 Dangerous Driving! Please slow down.")

# -----------------------------
# EXPLANATION
# -----------------------------
st.subheader("🧠 Why this score?")

reasons = []

if speed > 80:
    reasons.append("High speed detected")
if braking > 3:
    reasons.append("Harsh braking")
if turning > 3:
    reasons.append("Sharp turning")
if acceleration > 3:
    reasons.append("Aggressive acceleration")

if not reasons:
    reasons.append("Normal driving behavior")

for r in reasons:
    st.write(f"- {r}")

# -----------------------------
# REAL-TIME GRAPH
# -----------------------------
st.subheader("📊 Driving Trend")

chart_data = pd.DataFrame({
    "Speed": st.session_state.speed_history,
    "Risk Score": st.session_state.score_history
})

st.line_chart(chart_data)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📈 Model Performance")

st.write(f"**Accuracy:** {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
st.write("**Confusion Matrix:**")
st.write(cm)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🔥 Feature Importance")

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit")
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.components.v1.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXX"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-8H38CSX9T3');
</script>
""", height=0)

# -----------------------------------------------
# PAGE CONFIG (APP-LIKE UI)
# -----------------------------------------------
st.set_page_config(page_title="Smart Rider Score", layout="centered")

st.title("🚗 Smart Rider Score")
st.caption("Real-time AI-based driving safety assistant")

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
@st.cache_data
def load_data():
    # Make sure to have data.csv in your working directory
    return pd.read_csv("data.csv")

try:
    df = load_data()
except FileNotFoundError:
    # Dummy fallback in case data.csv isn't found locally
    st.error("Please ensure 'data.csv' is in the project folder.")
    st.stop()

# -----------------------------------------------
# TRAIN MODEL
# -----------------------------------------------
features = ["speed", "acceleration", "turn_rate", "braking"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# -----------------------------------------------
# EVALUATION
# -----------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------------------------
# FLOWCHART OPTION B: PENALTY SCORING ENGINE
# -----------------------------------------------
def calculate_safety_score(speed, accel, turning, braking):
    """
    Calculates a safety score starting at 100 based on detected penalties.
    Matches the project's system architecture flowchart.
    """
    base_score = 100
    penalties = 0
    
    # Flowchart Rule 1: Overspeeding
    if speed > 80:
        penalties += 20
        
    # Flowchart Rule 2: Hard Braking
    if speed > 60 and braking > 3.5:
        penalties += 15
        
    # Flowchart Rule 3: Sharp Turning
    if turning > 3.5:
        penalties += 15
        
    # Flowchart Rule 4: Aggressive Acceleration
    if accel > 3.5:
        penalties += 10
        
    final_score = base_score - penalties
    return max(0, final_score) # Score cannot drop below zero

# -----------------------------------------------
# SESSION STATE (REAL-TIME FEEL)
# -----------------------------------------------
if "speed_history" not in st.session_state:
    st.session_state.speed_history = []

if "score_history" not in st.session_state:
    st.session_state.score_history = []

# -----------------------------------------------
# FILE UPLOAD SECTION
# -----------------------------------------------
st.subheader("📂 Upload Ride Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    csv_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", csv_data.head())

    st.subheader("📊 Processing Uploaded Data")

    scores = []
    # Apply the penalty-based logic to every row in the uploaded CSV
    for _, row in csv_data.iterrows():
        score = calculate_safety_score(
            row["speed"], 
            row["acceleration"], 
            row["turn_rate"], 
            row["braking"]
        )
        scores.append(score)

    csv_data["Safety Score"] = scores
    st.line_chart(csv_data[["Safety Score"]])  

# -----------------------------------------------
# INPUT SECTION (CLEAN UI)
# -----------------------------------------------
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

# -----------------------------------------------
# PREDICTION & REAL-TIME SCORE
# -----------------------------------------------
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0]

# Generate score via the new penalty-based engine
final_safety_score = calculate_safety_score(speed, acceleration, turning, braking)

# Adjusting category thresholds to reflect a 100-down scoring system
if final_safety_score >= 80:
    category = "🟢 Safe"
elif final_safety_score >= 50:
    category = "🟡 Risky"
else:
    category = "🔴 Dangerous"

# Save history for graphs
st.session_state.speed_history.append(speed)
st.session_state.score_history.append(final_safety_score)

# -----------------------------------------------
# OUTPUT (PRODUCT STYLE)
# -----------------------------------------------
st.markdown("## 🎯 Your Driving Score")

st.markdown(f"# {final_safety_score}/100")
st.markdown(f"### {category}")

# Alerts based on penalty-deducted score
if final_safety_score >= 80:
    st.success("✅ Safe Driving")
elif final_safety_score >= 50:
    st.warning("⚠️ Drive Carefully")
else:
    st.error("🚨 Dangerous Driving! Please slow down.")

# -----------------------------------------------
# EXPLANATION
# -----------------------------------------------
st.subheader("🧠 Why this score?")

reasons = []

# This matches the flowchart and penalty function thresholds perfectly
if speed > 80:
    reasons.append("High speed detected (-20 points)")
if speed > 60 and braking > 3.5:
    reasons.append("Harsh braking (-15 points)")
if turning > 3.5:
    reasons.append("Sharp turning (-15 points)")
if acceleration > 3.5:
    reasons.append("Aggressive acceleration (-10 points)")

if not reasons:
    reasons.append("Normal driving behavior. Keep it up!")

for r in reasons:
    st.write(f"- {r}")

# -----------------------------------------------
# REAL-TIME GRAPH
# -----------------------------------------------
st.subheader("📊 Driving Trend")

chart_data = pd.DataFrame({
    "Speed": st.session_state.speed_history,
    "Safety Score": st.session_state.score_history
})

st.line_chart(chart_data)

# -----------------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------------
st.subheader("📈 Model Performance")

st.write(f"**Accuracy:** {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
st.write("**Confusion Matrix:**")
st.write(cm)

# -----------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------
st.subheader("🔥 Feature Importance")

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# -----------------------------------------------
# FOOTER
# -----------------------------------------------
st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit")
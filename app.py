import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 🛠 PAGE CONFIG
st.set_page_config(
    page_title="Advanced Admission Simulator",
    page_icon="🎓",
    layout="centered"
)

# 🌈 ANIMATED BACKGROUND (CSS)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#a1c4fd,#c2e9fb,#fbc2eb,#a6c0fe);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
""", unsafe_allow_html=True)

# 📦 LOAD MODEL + SCALER
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# 🧿 TITLE & INTRO
st.markdown("<h1 style='text-align:center; color:#1e293b;'>🎓 Advanced University Admission Simulator</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#334155; font-size:17px;'>"
    "Explore how changes in GRE, TOEFL, CGPA, SOP, LOR, and Research affect your admission probability."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# 🟦 INPUT SLIDERS
st.subheader("🧪 Adjust Your Academic Profile")

col1, col2 = st.columns(2)

with col1:
    GRE = st.slider("GRE Score", 260, 340, 310)
    TOEFL = st.slider("TOEFL Score", 80, 120, 100)
    UR = st.slider("University Rating (1–5)", 1, 5, 3)
    Research = st.selectbox("Research Experience", [0, 1], index=1)

with col2:
    SOP = st.slider("SOP Strength (1–5)", 1, 5, 3)
    LOR = st.slider("LOR Strength (1–5)", 1, 5, 3)
    CGPA = st.slider("CGPA (0–10)", 5.0, 10.0, 8.2, step=0.1)

# Create input DF
input_df = pd.DataFrame([{
    "GRE Score": GRE,
    "TOEFL Score": TOEFL,
    "University Rating": UR,
    "SOP": SOP,
    "LOR": LOR,
    "CGPA": CGPA,
    "Research": Research
}])

# 🔮 PREDICT PROBABILITY
scaled = scaler.transform(input_df)
prob = float(model.predict(scaled)[0])
prob = max(0.0, min(1.0, prob))  # clamp
prob_percent = round(prob * 100, 2)

# 🎓 MAP PROBABILITY → TIER
def prob_to_tier(p):
    if p < 0.20: return 1
    elif p < 0.40: return 2
    elif p < 0.60: return 3
    elif p < 0.80: return 4
    else: return 5

tier = prob_to_tier(prob)

tier_labels = {
    1: "🔴 Tier 1 – Very High Risk",
    2: "🟠 Tier 2 – Moderate Risk",
    3: "🟡 Tier 3 – Good Chances",
    4: "🟢 Tier 4 – Strong Chances",
    5: "💚 Tier 5 – Excellent Chances"
}

tier_universities = {
    5: ["MIT", "Stanford", "Harvard"],
    4: ["UT Austin", "UCLA", "UIUC"],
    3: ["ASU", "UTD", "Iowa State"],
    2: ["UNT", "Texas Tech"],
    1: ["Backup / Safer Programs"]
}

# 📊 FEATURE SENSITIVITY EXPLORER
st.markdown("## 📊 Feature Sensitivity Explorer")

feature = st.selectbox(
    "Choose a feature to analyze:",
    ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
)

# Generate values per feature
if feature == "GRE Score":
    values = np.arange(260, 341, 5)
elif feature == "TOEFL Score":
    values = np.arange(80, 121, 2)
elif feature == "University Rating":
    values = np.arange(1, 6, 1)
elif feature == "SOP":
    values = np.arange(1, 6, 1)
elif feature == "LOR":
    values = np.arange(1, 6, 1)
elif feature == "CGPA":
    values = np.arange(5.0, 10.01, 0.1)
elif feature == "Research":
    values = [0, 1]

probs = []
for val in values:
    temp = input_df.copy()
    temp[feature] = val
    scaled_temp = scaler.transform(temp)
    p = float(model.predict(scaled_temp)[0])
    probs.append(max(0, min(1, p)))

# Plot curve
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(values, probs, linewidth=2.5, color="#2563eb")
ax1.set_xlabel(feature, fontsize=12)
ax1.set_ylabel("Admission Probability", fontsize=12)
ax1.set_title(f"{feature} vs Admission Probability", fontsize=14)
ax1.grid(alpha=0.3)
st.pyplot(fig1)

# 🟣 RADAR CHART
st.markdown("## 🟣 Academic Profile Radar Chart")

labels = ["GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA"]
stats = [GRE/340, TOEFL/120, UR/5, SOP/5, LOR/5, CGPA/10]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats += stats[:1]
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax2.plot(angles, stats, color="#7c3aed", linewidth=2)
ax2.fill(angles, stats, color="#c4b5fd", alpha=0.4)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels)
ax2.set_yticklabels([])
ax2.set_title("Academic Strength Radar", fontsize=14)
st.pyplot(fig2)

# 🎯 FINAL PREDICTION
st.markdown("---")
st.markdown("## 🎯 Final Admission Prediction")

st.subheader("📈 Admission Probability")
st.progress(prob)
st.markdown(f"<h2 style='color:#1e293b'>{prob_percent}%</h2>", unsafe_allow_html=True)

st.subheader("🏛 Predicted Tier")
st.markdown(
    f"<div style='padding:14px;border-radius:12px;background:linear-gradient(120deg,#6366f1,#22c55e);color:white;font-size:20px;'>"
    f"{tier_labels[tier]}</div>",
    unsafe_allow_html=True
)

st.subheader("🏫 Suggested Universities")
for uni in tier_universities[tier]:
    st.write("✔", uni)

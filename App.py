import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(
    page_title="EduPilot AI - SDG 4 Quality Education", page_icon="🎓"
)

st.title("🎓 EduPilot AI: Quality Education & Career Readiness Model")
st.write(
    "UN SDG 4 & 8: AI-powered skill gap analysis, personalized learning"
    " paths, and at-risk student detection."
)

st.sidebar.header("📊 Student Assessment Inputs")
academic_score = st.sidebar.slider("Academic Baseline Score (%)", 0, 100, 72)
quiz_completion = st.sidebar.slider("Micro-Lesson Completion (%)", 0, 100, 85)
study_hours = st.sidebar.slider("Weekly Self-Study Hours", 0, 40, 14)
industry_match = st.sidebar.slider(
    "Industry Skill Alignment Score (%)", 0, 100, 60
)

# Training Dataset
X_train = np.array([
    [90, 95, 20, 90],
    [80, 85, 15, 75],
    [60, 50, 8, 45],
    [40, 30, 4, 30],
    [75, 80, 12, 70],
])
y_train = np.array([1, 1, 0, 0, 1])

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

if st.button("🚀 Run Skill Gap Analysis & Prediction"):
  input_data = [[
      academic_score,
      quiz_completion,
      study_hours,
      industry_match,
  ]]
  prediction = model.predict(input_data)[0]
  readiness_score = round(
      (
          academic_score * 0.3
          + quiz_completion * 0.3
          + industry_match * 0.4
      ),
      1,
  )

  st.subheader("🎯 Model Output Summary")
  st.metric(label="Calculated Career Readiness Score", value=f"{readiness_score}%")

  if prediction == 1:
    st.success("✅ **Status:** Student is on track for target industry roles.")
  else:
    st.error(
        "⚠️ **Status:** At-Risk Student — Targeted Academic Intervention"
        " Required."
    )

  st.subheader("🧠 Identified Skill Gaps & Recommended Path")
  if industry_match < 70:
    st.warning("• **Identified Gap:** Practical Industry Tools & Frameworks")
    st.info(
        "• **Actionable Micro-Lesson Path:** Complete Module 3 (Real-World"
        " Case Studies & Project Portfolio)."
    )
  else:
    st.info(
        "• **Actionable Micro-Lesson Path:** Advanced Specialization & Mock"
        " Industry Assessments."
    )

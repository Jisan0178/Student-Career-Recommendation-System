
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("models/best_synthetic_model.pkl")

feature_names = joblib.load(
    "models/synthetic_feature_names.pkl"
)

# Page configuration
st.set_page_config(
    page_title="Student Career Recommendation",
    page_icon="🎓",
    layout="centered"
)

# Application title
st.title("🎓 Student Career Recommendation System")

st.write(
    "Enter your skill levels and interests "
    "to get a career recommendation."
)

st.info("Select a score from 1 to 5 for each skill.")

# Student skill inputs
st.subheader("Student Skill Assessment")

python_score = st.slider("Python", 1, 5, 3)
java_score = st.slider("Java", 1, 5, 3)
math_score = st.slider("Mathematics", 1, 5, 3)
communication = st.slider("Communication", 1, 5, 3)
problem_solving = st.slider("Problem Solving", 1, 5, 3)
data_analysis = st.slider("Data Analysis", 1, 5, 3)
ai_interest = st.slider("AI/ML Interest", 1, 5, 3)
web_interest = st.slider("Web Development Interest", 1, 5, 3)

# Prediction button
if st.button("Recommend My Career"):

    student_data = {
        "Python": [python_score],
        "Java": [java_score],
        "Mathematics": [math_score],
        "Communication": [communication],
        "Problem_Solving": [problem_solving],
        "Data_Analysis": [data_analysis],
        "AI_ML_Interest": [ai_interest],
        "Web_Dev_Interest": [web_interest]
    }

    student_df = pd.DataFrame(student_data)
    student_df = student_df[feature_names]

    prediction = model.predict(student_df)[0]

    st.success("Recommended Career: " + prediction)

    probabilities = model.predict_proba(student_df)[0]
    confidence = max(probabilities) * 100

    st.write(
        "Model Confidence:",
        round(confidence, 2),
        "%"
    )

    st.warning(
    "This application uses a synthetic dataset. "
    "Its recommendations are for educational demonstration "
    "and should not be treated as professional career advice."
    )

# Footer
st.divider()
st.caption("Student Career Recommendation System | Machine Learning Mini Project")

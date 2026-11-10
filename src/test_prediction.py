
import pandas as pd
import joblib

# Load the best trained model
model = joblib.load("models/best_model.pkl")

# Load feature names
feature_names = joblib.load("models/feature_names.pkl")

# New student information
student_data = {
    "Python": [5],
    "Java": [2],
    "Mathematics": [5],
    "Communication": [3],
    "Problem_Solving": [5],
    "Data_Analysis": [4],
    "AI_ML_Interest": [5],
    "Web_Dev_Interest": [2]
}

# Convert data into DataFrame
student_df = pd.DataFrame(student_data)

# Arrange columns in the correct order
student_df = student_df[feature_names]

# Predict career
prediction = model.predict(student_df)[0]

print("Student Skill Details:")
print(student_df)

print("\nPredicted Career:", prediction)

# Display model confidence
probabilities = model.predict_proba(student_df)[0]
confidence = max(probabilities) * 100

print("Model Confidence:", round(confidence, 2), "%")

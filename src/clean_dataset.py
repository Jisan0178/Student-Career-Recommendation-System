
import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/student_career_dataset.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

# Check valid score range
features = [
    "Python", "Java", "Mathematics", "Communication",
    "Problem_Solving", "Data_Analysis",
    "AI_ML_Interest", "Web_Dev_Interest"
]

for column in features:
    df = df[df[column].between(1, 5)]

# Check target values
valid_careers = [
    "Software Developer",
    "Data Scientist",
    "AI/ML Engineer",
    "Web Developer"
]

df = df[df["Preferred_Career"].isin(valid_careers)]

# Save cleaned dataset
df.to_csv("data/processed/cleaned_student_career.csv", index=False)

print("DATA CLEANING COMPLETED!")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("\nCleaned Dataset:")
print(df.head())

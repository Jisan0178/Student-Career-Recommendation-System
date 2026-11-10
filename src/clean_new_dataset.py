
import pandas as pd

# Load the new dataset
df = pd.read_csv("data/raw/real_student_career_dataset.csv")

print("Original Dataset Shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

# Define valid score columns
score_columns = [
    "Python",
    "Java",
    "Mathematics",
    "Communication",
    "Problem_Solving",
    "Data_Analysis",
    "AI_ML_Interest",
    "Web_Dev_Interest"
]

# Keep scores between 1 and 5
for column in score_columns:
    df = df[(df[column] >= 1) & (df[column] <= 5)]

# Define valid career categories
valid_careers = [
    "Software Developer",
    "Data Scientist",
    "AI/ML Engineer",
    "Web Developer"
]

df = df[df["Preferred_Career"].isin(valid_careers)]

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_student_career_synthetic.csv",
    index=False
)

print("\nCleaning completed!")
print("Cleaned Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nCareer Distribution:")
print(df["Preferred_Career"].value_counts())
print("\nCleaned dataset saved successfully!")

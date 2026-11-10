
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/student_career_dataset.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Display number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Display career distribution
print("\nCAREER DISTRIBUTION:")
print(df["Preferred_Career"].value_counts())

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nVERIFYING DATASET:")

print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumn Names:")
for column in df.columns:
    print(column)

print("\nFirst 10 Records:")
print(df.head(10).to_string(index=False))


# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# Check incorrect scores
features = [
    "Python", "Java", "Mathematics", "Communication",
    "Problem_Solving", "Data_Analysis",
    "AI_ML_Interest", "Web_Dev_Interest"
]

print("\nINVALID SCORES:")
for column in features:
    invalid = df[~df[column].between(1, 5)]
    print(column, ":", len(invalid))

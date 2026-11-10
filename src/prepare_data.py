
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_student_career.csv")

# Select input features
X = df.drop("Preferred_Career", axis=1)

# Select target column
y = df["Preferred_Career"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("DATA PREPARATION COMPLETED!")

print("\nOriginal Dataset:", df.shape)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Column:")
print(y.name)

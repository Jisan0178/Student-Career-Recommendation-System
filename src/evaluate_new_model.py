
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_student_career_synthetic.csv"
)

X = df.drop("Preferred_Career", axis=1)
y = df["Preferred_Career"]

# Split dataset exactly as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Load saved best model
model = joblib.load("models/best_synthetic_model.pkl")

# Predict test data
predictions = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted Career")
plt.ylabel("Actual Career")
plt.tight_layout()

plt.savefig("reports/new_dataset_confusion_matrix.png")
plt.show()

# Display classification report
print("Classification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Load model comparison results
results = pd.read_csv(
    "reports/new_dataset_model_comparison.csv"
)

# Plot model accuracy
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("reports/new_dataset_accuracy_comparison.png")
plt.show()

# Plot F1 score
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="F1 Score"
)

plt.title("Model F1 Score Comparison")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("reports/new_dataset_f1_comparison.png")
plt.show()

print("\nEvaluation completed successfully!")

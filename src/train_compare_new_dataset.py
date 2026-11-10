
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load cleaned synthetic dataset
df = pd.read_csv(
    "data/processed/cleaned_student_career_synthetic.csv"
)

# Separate features and target
X = df.drop("Preferred_Career", axis=1)
y = df["Preferred_Career"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define three ML models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

results = []

best_model = None
best_name = ""
best_score = -1

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n", name)
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

# Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/new_dataset_model_comparison.csv",
    index=False
)

# Save the best model separately
joblib.dump(
    best_model,
    "models/best_synthetic_model.pkl"
)

joblib.dump(
    list(X.columns),
    "models/synthetic_feature_names.pkl"
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBest Model:", best_name)
print("Best F1 Score:", round(best_score, 4))

print("\nNew dataset model training completed!")

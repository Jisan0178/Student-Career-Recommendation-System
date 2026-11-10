
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

# Load dataset
df = pd.read_csv("data/processed/cleaned_student_career.csv")

# Separate features and target
X = df.drop("Preferred_Career", axis=1)
y = df["Preferred_Career"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define models
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

best_model = None
best_name = ""
best_score = -1

# Train and select best model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(name, "F1-score:", score)

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save feature names
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("\nBEST MODEL:", best_name)
print("BEST F1-SCORE:", best_score)
print("\nBest model saved successfully!")

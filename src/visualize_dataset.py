
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_student_career.csv")

# Set visualization style
sns.set_theme(style="whitegrid")

# 1. Career Distribution
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Preferred_Career"
)

plt.title("Student Career Distribution")
plt.xlabel("Career")
plt.ylabel("Number of Students")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/career_distribution.png")
plt.show()


# 2. Average Skills by Career
features = [
    "Python", "Java", "Mathematics", "Communication",
    "Problem_Solving", "Data_Analysis",
    "AI_ML_Interest", "Web_Dev_Interest"
]

career_skills = df.groupby("Preferred_Career")[features].mean()

career_skills.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Average Skills by Career")
plt.xlabel("Career")
plt.ylabel("Average Score")
plt.xticks(rotation=20)
plt.legend(title="Skills", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig("reports/average_skills.png")
plt.show()


# 3. Feature Correlation Heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.show()

print("DATA VISUALIZATION COMPLETED!")

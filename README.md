# Student Career Recommendation System

## Project Overview
The Student Career Recommendation System is a Machine Learning mini-project that recommends a suitable career category based on a student's skills and interests.

## Objectives
- Analyze student skills and interests.
- Apply Machine Learning classification algorithms.
- Compare different ML models.
- Predict a suitable career category.
- Provide an interactive web application.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset
The project currently uses a synthetic dataset containing 200 student records.

Each record contains eight skill and interest features:
1. Python
2. Java
3. Mathematics
4. Communication
5. Problem Solving
6. Data Analysis
7. AI/ML Interest
8. Web Development Interest

Target column: `Preferred_Career`

Career categories:
- Software Developer
- Data Scientist
- AI/ML Engineer
- Web Developer

**Dataset limitation:** The dataset is simulated and does not represent real student survey responses.

## Machine Learning Algorithms
1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)

Models are compared using Accuracy, Precision, Recall and F1-score.

## Application Features
- Student skill assessment using sliders.
- Career prediction.
- Model confidence display.
- Interactive Streamlit interface.

## Project Structure
```text
Student_Career_Recommendation/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

1. Clone the repository:
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project folder:
```bash
cd Student_Career_Recommendation
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app/app.py
```

## Limitations
- The current dataset is synthetic.
- Predictions are for educational demonstration only.
- Recommendations should not be treated as professional career advice.

## Future Improvements
- Collect real student survey data.
- Improve model evaluation.
- Add more student skill features.
- Improve the user interface.
- Evaluate generalization using real student data.

## Author
Jisan Ali

B.Tech - Computer Science and Engineering (AI & ML)
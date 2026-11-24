
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import json
from pathlib import Path
from datetime import datetime


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_synthetic_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "synthetic_feature_names.pkl"
DB_PATH = BASE_DIR / "data" / "career_recommendations.db"


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🎯",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
.stApp {
    background: #100D1D;
    color: white;
}

[data-testid="stSidebar"] {
    background: #19142B;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #B69CFF;
}

.subtitle {
    color: #B8B0CC;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #211A35;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #3A2D59;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


try:

    model, feature_names = load_model()

except Exception as e:

    st.error(f"Could not load model: {e}")
    st.stop()


# ==============================
# DATABASE CONNECTION
# ==============================

def get_connection():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DB_PATH)


# ==============================
# CREATE / UPDATE DATABASE
# ==============================

def create_table():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            programming_skills INTEGER,
            mathematics INTEGER,
            analytical_thinking INTEGER,
            problem_solving INTEGER,
            creativity INTEGER,
            communication INTEGER,
            technical_knowledge INTEGER,
            interest_in_technology INTEGER,
            recommended_career TEXT,
            confidence REAL,
            probabilities TEXT
        )
    """)

    # Read existing column names
    existing_columns = {
        row[1].lower()
        for row in conn.execute(
            "PRAGMA table_info(recommendations)"
        ).fetchall()
    }

    required_columns = {
        "created_at": "TEXT",
        "programming_skills": "INTEGER",
        "mathematics": "INTEGER",
        "analytical_thinking": "INTEGER",
        "problem_solving": "INTEGER",
        "creativity": "INTEGER",
        "communication": "INTEGER",
        "technical_knowledge": "INTEGER",
        "interest_in_technology": "INTEGER",
        "recommended_career": "TEXT",
        "confidence": "REAL",
        "probabilities": "TEXT"
    }

    # Add only genuinely missing columns
    for column, data_type in required_columns.items():

        if column.lower() not in existing_columns:

            conn.execute(
                f"ALTER TABLE recommendations "
                f"ADD COLUMN {column} {data_type}"
            )

    conn.commit()
    conn.close()


create_table()


# ==============================
# SAVE PREDICTION
# ==============================

def save_prediction(
    values,
    career,
    confidence,
    probabilities
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO recommendations (
            created_at,
            programming_skills,
            mathematics,
            analytical_thinking,
            problem_solving,
            creativity,
            communication,
            technical_knowledge,
            interest_in_technology,
            recommended_career,
            confidence,
            probabilities
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        career,
        confidence,
        json.dumps(probabilities)
    ))

    conn.commit()
    conn.close()


# ==============================
# GET HISTORY
# ==============================

def get_history():

    conn = get_connection()

    history = pd.read_sql_query(
        "SELECT * FROM recommendations ORDER BY id DESC",
        conn
    )

    conn.close()

    return history


# ==============================
# DELETE PREDICTION
# ==============================

def delete_prediction(record_id):

    conn = get_connection()

    conn.execute(
        "DELETE FROM recommendations WHERE id = ?",
        (record_id,)
    )

    conn.commit()
    conn.close()


# ==============================
# SIDEBAR
# ==============================

st.sidebar.markdown("# 🎯 CareerPath AI")

st.sidebar.caption(
    "Student Career Recommendation System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "New Recommendation",
        "History",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning Mini Project"
)


# ==============================
# DASHBOARD
# ==============================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Welcome to CareerPath AI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Discover a suitable career path based on your skills and interests.</p>',
        unsafe_allow_html=True
    )

    history = get_history()

    total = len(history)

    if total > 0:

        career_counts = history[
            "recommended_career"
        ].dropna()

        if not career_counts.empty:
            top_career = career_counts.mode().iloc[0]
        else:
            top_career = "No data yet"

        avg_confidence = history[
            "confidence"
        ].mean()

        if pd.isna(avg_confidence):
            avg_confidence = 0

    else:

        top_career = "No data yet"
        avg_confidence = 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total
    )

    col2.metric(
        "Most Recommended Career",
        top_career
    )

    col3.metric(
        "Average Model Score",
        f"{avg_confidence:.1%}"
    )

    st.markdown("---")

    st.subheader("About CareerPath AI")

    st.write("""
    CareerPath AI is a machine learning-based application
    that recommends possible career paths based on student
    skills and interests.

    The application uses a trained Random Forest model to
    predict one of four career categories.
    """)

    st.info(
        "This project uses a simulated dataset for academic demonstration. "
        "Its predictions are suggestions, not guaranteed career outcomes."
    )


# ==============================
# NEW RECOMMENDATION
# ==============================

elif page == "New Recommendation":

    st.markdown(
        '<p class="main-title">Career Recommendation</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Adjust the sliders according to your skills and interests."
    )

    labels = {
        "programming_skills": "Programming Skills",
        "mathematics": "Mathematics",
        "analytical_thinking": "Analytical Thinking",
        "problem_solving": "Problem Solving",
        "creativity": "Creativity",
        "communication": "Communication",
        "technical_knowledge": "Technical Knowledge",
        "interest_in_technology": "Interest in Technology"
    }

    values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        label = labels.get(
            str(feature),
            str(feature).replace("_", " ").title()
        )

        with col1 if i % 2 == 0 else col2:

            value = st.slider(
                label,
                min_value=1,
                max_value=5,
                value=3,
                key=f"slider_{feature}"
            )

        values.append(value)

    st.markdown("---")

    if st.button(
        "🎯 Recommend My Career",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [values],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        probabilities_array = model.predict_proba(
            input_data
        )[0]

        probabilities = {
            str(name): float(prob)
            for name, prob in zip(
                model.classes_,
                probabilities_array
            )
        }

        confidence = float(
            max(probabilities_array)
        )

        save_prediction(
            values,
            str(prediction),
            confidence,
            probabilities
        )

        st.success(
            "Career recommendation generated successfully!"
        )

        st.markdown("### Your Recommended Career")

        st.markdown(
            f"<h2 style='color:#B69CFF'>{prediction}</h2>",
            unsafe_allow_html=True
        )

        st.metric(
            "Highest Prediction Probability",
            f"{confidence:.2%}"
        )

        st.markdown(
            "### Career Prediction Probabilities"
        )

        probability_df = pd.DataFrame({
            "Career": list(probabilities.keys()),
            "Probability": list(probabilities.values())
        })

        st.bar_chart(
            probability_df.set_index("Career")
        )

        st.info(
            "Prediction probabilities are model outputs, "
            "not guarantees of career success."
        )


# ==============================
# HISTORY
# ==============================

elif page == "History":

    st.markdown(
        '<p class="main-title">Prediction History</p>',
        unsafe_allow_html=True
    )

    st.write(
        "View and manage your previous career recommendations."
    )

    history = get_history()

    if history.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        st.metric(
            "Total Saved Predictions",
            len(history)
        )

        st.markdown("---")

        st.subheader("Saved Recommendations")

        for _, record in history.iterrows():

            record_id = int(record["id"])

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(
                        f"### {record['recommended_career']}"
                    )

                    st.caption(
                        f"Date: {record['created_at']}"
                    )

                    if pd.notna(record["confidence"]):

                        st.write(
                            f"Prediction Probability: "
                            f"{record['confidence']:.2%}"
                        )

                with col2:

                    st.write("")

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{record_id}",
                        use_container_width=True
                    ):

                        delete_prediction(record_id)

                        st.toast(
                            "Prediction deleted successfully!"
                        )

                        st.rerun()

                with st.expander("View Details"):

                    st.write("**Student Input Values**")

                    # Display all input columns dynamically
                    excluded_columns = [
                        "id",
                        "created_at",
                        "recommended_career",
                        "confidence",
                        "probabilities"
                    ]

                    input_details = record.drop(
                        labels=excluded_columns,
                        errors="ignore"
                    )

                    details_df = pd.DataFrame({
                        "Feature": input_details.index,
                        "Score": input_details.values
                    })

                    st.dataframe(
                        details_df,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.write("**Career Probabilities**")

                    try:

                        if pd.notna(record["probabilities"]):

                            probabilities = json.loads(
                                record["probabilities"]
                            )

                            probability_df = pd.DataFrame.from_dict(
                                probabilities,
                                orient="index",
                                columns=["Probability"]
                            )

                            st.bar_chart(probability_df)

                        else:

                            st.info(
                                "Probability details are not available."
                            )

                    except (TypeError, ValueError, json.JSONDecodeError):

                        st.info(
                            "Probability details are not available "
                            "for this saved record."
                        )

        st.markdown("---")

        st.subheader("Download History")

        download_data = history.drop(
            columns=["probabilities"],
            errors="ignore"
        )

        csv_data = download_data.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="career_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================
# ABOUT PROJECT
# ==============================

elif page == "About Project":

    st.markdown(
        '<p class="main-title">About the Project</p>',
        unsafe_allow_html=True
    )

    st.subheader("Project Information")

    st.write("""
    **Project Name:** Student Career Recommendation System

    **Course:** Introduction to Machine Learning

    **Algorithm:** Random Forest

    **Framework:** Streamlit

    **Programming Language:** Python

    **Dataset:** 200 simulated student records
    """)

    st.markdown("---")

    st.subheader(
        "Machine Learning Model Performance"
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            0.7750,
            0.7707,
            0.7583,
            0.7565
        ]
    })

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.info("""
    The dataset used in this project is simulated.
    The model's performance on real-world student data
    may be different.
    """)

    st.markdown("---")

    st.caption(
        "Developed by Jisan Ali | B.Tech CSE"
    )

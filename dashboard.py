"""
=========================================================
Dashboard Home
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import streamlit as st

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Saint-Gobain Welding AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

@st.cache_data
def load_data():

    loader = WeldingDataLoader()

    df = loader.load_data()

    df = DataPreprocessor.preprocess(df)

    return df


df = load_data()

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("🏭 Intelligent Welding Time Estimation & Process Analytics System")

st.caption(
    "Saint-Gobain | Machine Learning Based Welding Cycle Time Prediction"
)

st.divider()

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Production Samples",
    len(df),
)

c2.metric(
    "Average Cycle Time",
    f"{df['total_time'].mean():.2f} sec",
)

c3.metric(
    "Fastest Cycle",
    f"{df['total_time'].min():.2f} sec",
)

c4.metric(
    "Slowest Cycle",
    f"{df['total_time'].max():.2f} sec",
)

st.divider()

# -------------------------------------------------------
# Project Overview
# -------------------------------------------------------

st.subheader("Project Overview")

st.markdown("""
This system predicts welding cycle time for PVC window profiles using a
CatBoost Machine Learning model trained on production data collected from
the welding station.

The application provides:

- 📊 Production Analytics
- 🤖 Cycle Time Prediction
- 📈 Model Performance Evaluation
- 🧠 AI Explainability using SHAP
- ☁️ Live Google Sheets Integration
""")

st.divider()

# -------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Dataset Summary")

    summary = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Operators": df["operator"].nunique(),
        "Production Lines": df["line_no"].nunique(),
        "Articles": df["article_no"].nunique(),
    }

    st.dataframe(
        summary,
        width="stretch",
    )

with right:

    st.subheader("Current Features")

    st.markdown("""
**Machine Inputs**

- Window Length
- Window Width
- Cross Section Length
- Cross Section Width
- Line Number
- Operator

**Predicted Output**

- Total Welding Cycle Time
""")

st.divider()

# -------------------------------------------------------
# Navigation
# -------------------------------------------------------

st.subheader("Dashboard Modules")

m1, m2 = st.columns(2)

with m1:

    st.success("📊 Analytics")

    st.write(
        "- Cycle Time Distribution\n"
        "- Operator Comparison\n"
        "- Correlation Matrix\n"
        "- Scatter Analysis"
    )

    st.success("🤖 Prediction")

    st.write(
        "- Predict New Window Cycle Time\n"
        "- Production Summary"
    )

with m2:

    st.success("📈 Model")

    st.write(
        "- MAE\n"
        "- RMSE\n"
        "- R² Score\n"
        "- Feature Importance"
    )

    st.success("🧠 Explainability")

    st.write(
        "- SHAP Feature Importance\n"
        "- Prediction Explanation\n"
        "- Local Feature Contributions"
    )

st.divider()

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.info(
    "Use the navigation menu on the left to access Analytics, Prediction, Model Performance, and Explainability."
)
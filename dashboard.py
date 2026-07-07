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

st.info(
    """
    This dashboard serves as the landing page for the Intelligent Welding Time Estimation & Process Analytics system.
    It loads live production data directly from Google Sheets to provide up-to-date insights.
    Here, you can view a summary of overall plant performance before navigating to detailed modules for deeper analysis.
    """
)

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Production Samples",
    len(df),
)

c2.metric(
    "Unique Articles",
    df['article_no'].nunique(),
)

c3.metric(
    "Average Cycle Time",
    f"{df['total_time'].mean():.2f} sec",
)

c4.metric(
    "Fastest Cycle",
    f"{df['total_time'].min():.2f} sec",
)

c5.metric(
    "Slowest Cycle",
    f"{df['total_time'].max():.2f} sec",
)

st.divider()

st.markdown("""
### Importance of Article-wise Comparison

Different article numbers represent distinct window geometries and materials, each with unique expected welding cycle times.  
Therefore, comparing overall average cycle times across all articles can be misleading.  
To gain meaningful insights and identify true performance variations, all analyses and comparisons should be conducted within the same article number or between specific article numbers.  
This article-wise approach ensures accurate benchmarking and effective process optimization.
""")

st.divider()

# -------------------------------------------------------
# Project Overview
# -------------------------------------------------------

st.subheader("Project Overview")

st.markdown("""
This system primarily focuses on comparing production performance between different article numbers while predicting welding cycle time for PVC window profiles.  
It uses a CatBoost Machine Learning model trained on production data collected from the welding station.

The application provides:

- 📊 Production Analytics
- 🤖 Cycle Time Prediction
- 📈 Model Performance Evaluation
- 🧠 AI Explainability using SHAP
- ☁️ Live Google Sheets Integration
""")

st.markdown("""
### What This System Helps Engineers Do

- Compare production performance article-wise, recognizing that different article numbers represent different geometries and materials, and thus have different expected cycle times.
- Monitor production performance in real-time.
- Identify bottlenecks and areas for process improvement.
- Evaluate operator efficiency and line performance.
- Predict welding cycle time for new window profiles before production.
- Validate AI model accuracy and reliability.
- Improve overall productivity through data-driven insights.
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

    st.markdown("""
- **Rows**: Number of individual production records.
- **Columns**: Number of features and variables captured.
- **Operators**: Count of unique personnel performing welding.
- **Production Lines**: Number of distinct welding stations.
- **Articles**: Number of unique product identifiers (article numbers). This is the most important manufacturing grouping for analysis, as it defines different window geometries and materials, which directly impact welding cycle times.
""")

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

st.markdown("""
### Understanding the Dataset

- **Rows** represent individual production records.
- **Columns** denote the number of features and variables captured.
- **Operators** count the unique personnel performing welding.
- **Production Lines** indicate the number of distinct welding stations.
- **Articles** correspond to unique product identifiers (article numbers).

Article Number is a critical manufacturing identifier as different article numbers correspond to different window geometries and materials, which naturally result in varying welding times.

**Important Observation:** Comparisons and analyses should always be performed within the same article number whenever possible to ensure meaningful conclusions.
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
        "- Article-wise Production Analysis\n"
        "- Article Comparison\n"
        "- Operator Comparison\n"
        "- Correlation Analysis\n"
        "- Bottleneck Identification"
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

st.markdown("""
### How to Use This Dashboard

Step 1: Select or compare specific article numbers to focus your analysis.  
Step 2: Analyze production performance and operator contributions within those articles.  
Step 3: Use Prediction modules to estimate cycle times for new window profiles of selected articles.  
Step 4: Validate Model Performance for the chosen articles to ensure AI accuracy.  
Step 5: Utilize Explainability tools to understand prediction drivers and build trust in the model.
""")

st.divider()

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("""
### Final Remarks

For accurate and meaningful insights, article number should always be the first level of comparison before evaluating operators, production lines, or overall process performance.  
Recognizing the inherent differences in window designs and materials across articles ensures that analyses reflect true production dynamics and support effective decision-making.
""")
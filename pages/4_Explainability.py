"""
=========================================================
Model Explainability
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.trainer import ModelTrainer


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🧠",
    layout="wide",
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

trainer = ModelTrainer()

feature_columns = trainer.feature_columns

X = df[feature_columns]

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = joblib.load("models/catboost_model.pkl")

# -------------------------------------------------------
# Feature Importance
# -------------------------------------------------------

importance = model.get_feature_importance()

importance_df = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": importance
    }
).sort_values(
    by="Importance",
    ascending=False
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🧠 AI Model Explainability Dashboard")

st.markdown("---")

st.write("""
This page explains which production parameters have the greatest influence
on the CatBoost model's welding time prediction.

Instead of SHAP, CatBoost's built-in Feature Importance is used, providing
a fast and reliable explanation that is fully compatible with Streamlit Cloud.
""")

# -------------------------------------------------------
# Sample Prediction
# -------------------------------------------------------

sample_index = st.slider(
    "Select Production Sample",
    0,
    len(X) - 1,
    0
)

selected = X.iloc[[sample_index]]

prediction = model.predict(selected)[0]

actual = df.iloc[sample_index]["total_time"]

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "Predicted Time",
    f"{prediction:.2f} sec"
)

c2.metric(
    "Actual Time",
    f"{actual:.2f} sec"
)

c3.metric(
    "Prediction Error",
    f"{prediction-actual:.2f} sec"
)

st.divider()

# -------------------------------------------------------
# Feature Importance Chart
# -------------------------------------------------------

st.subheader("Overall Feature Importance")

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="CatBoost Feature Importance"
)

fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=550
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------------
# Feature Importance Table
# -------------------------------------------------------

st.subheader("Importance Ranking")

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)

# -------------------------------------------------------
# Selected Input
# -------------------------------------------------------

st.subheader("Selected Production Sample")

st.dataframe(
    selected,
    use_container_width=True,
    hide_index=True
)

# -------------------------------------------------------
# Interpretation
# -------------------------------------------------------

st.subheader("Interpretation")

top_feature = importance_df.iloc[0]["Feature"]
top_value = importance_df.iloc[0]["Importance"]

st.success(
    f"Most influential feature: **{top_feature}** "
    f"(Importance Score = {top_value:.2f})"
)

st.info(
    "Features with higher importance contribute more to the model's prediction "
    "of welding cycle time. Lower-ranked features have comparatively less influence."
)

st.markdown("""
### Key Insights

- Higher importance indicates a stronger impact on welding time prediction.
- Feature Importance is calculated directly from the trained CatBoost model.
- The ranking helps identify the process parameters that most influence production performance.
- This information can support process optimization, bottleneck analysis, and continuous improvement initiatives.
""")


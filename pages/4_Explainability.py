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
import shap
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
# SHAP Explainer
# -------------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🧠 AI Explainability Dashboard")

st.markdown("---")

st.write("""
This page explains **why** the CatBoost model predicts a particular
cycle time by showing the contribution of each input feature.
""")

# -------------------------------------------------------
# Select Sample
# -------------------------------------------------------

sample_index = st.slider(
    "Select Production Sample",
    0,
    len(X)-1,
    0,
)

selected = X.iloc[[sample_index]]

prediction = model.predict(selected)[0]

actual = df.iloc[sample_index]["total_time"]

# -------------------------------------------------------
# KPI
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
# Global SHAP Importance
# -------------------------------------------------------

importance = pd.DataFrame({

    "Feature":feature_columns,

    "Importance":abs(shap_values).mean(axis=0)

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    title="Global SHAP Feature Importance"

)

st.plotly_chart(
    fig,
    width="stretch"
)

# -------------------------------------------------------
# Local Explanation
# -------------------------------------------------------

local = pd.DataFrame({

    "Feature":feature_columns,

    "Contribution":shap_values[sample_index]

})

local["Impact"] = local["Contribution"].abs()

local = local.sort_values(
    by="Impact",
    ascending=False
)

fig = px.bar(

    local,

    x="Contribution",

    y="Feature",

    color="Contribution",

    orientation="h",

    title="Contribution of Each Feature"

)

st.plotly_chart(
    fig,
    width="stretch"
)

# -------------------------------------------------------
# Selected Input
# -------------------------------------------------------

st.subheader("Selected Window Details")

st.dataframe(
    selected,
    width="stretch",
    hide_index=True,
)

# -------------------------------------------------------
# SHAP Values
# -------------------------------------------------------

st.subheader("Feature Contributions")

st.dataframe(
    local,
    width="stretch",
    hide_index=True,
)

# -------------------------------------------------------
# Interpretation
# -------------------------------------------------------

st.subheader("Interpretation")

positive = local[local["Contribution"] > 0]

negative = local[local["Contribution"] < 0]

st.success(
    f"{len(positive)} feature(s) increased the predicted cycle time."
)

st.info(
    f"{len(negative)} feature(s) reduced the predicted cycle time."
)

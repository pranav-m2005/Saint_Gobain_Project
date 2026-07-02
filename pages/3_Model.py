"""
=========================================================
Model Performance
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
import joblib
import plotly.express as px

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.trainer import ModelTrainer


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
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

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = joblib.load("models/catboost_model.pkl")

trainer = ModelTrainer()

X = df[trainer.feature_columns]
y = df[trainer.target_column]

predictions = model.predict(X)

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

mae = mean_absolute_error(y, predictions)

rmse = mean_squared_error(
    y,
    predictions,
) ** 0.5


r2 = r2_score(
    y,
    predictions,
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("📈 Model Performance Dashboard")

st.markdown("---")

# -------------------------------------------------------
# KPI
# -------------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "MAE",
    f"{mae:.2f} sec"
)

c2.metric(
    "RMSE",
    f"{rmse:.2f} sec"
)

c3.metric(
    "R² Score",
    f"{r2:.3f}"
)

st.divider()

# -------------------------------------------------------
# Actual vs Predicted
# -------------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

fig = px.scatter(
    comparison,
    x="Actual",
    y="Predicted",
    title="Actual vs Predicted"

)

st.plotly_chart(
    fig,
    width="stretch"
)

# -------------------------------------------------------
# Error Distribution
# -------------------------------------------------------

comparison["Error"] = (
    comparison["Predicted"]
    - comparison["Actual"]
)

fig = px.histogram(
    comparison,
    x="Error",
    nbins=20,
    title="Prediction Error Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# -------------------------------------------------------
# Feature Importance
# -------------------------------------------------------

importance = pd.DataFrame({
    "Feature": trainer.feature_columns,
    "Importance": model.get_feature_importance(),
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
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# -------------------------------------------------------
# Results Table
# -------------------------------------------------------

results = X.copy()

results["Actual"] = y.values

results["Predicted"] = predictions

results["Error"] = (
    results["Predicted"]
    - results["Actual"]
)

st.subheader("Prediction Results")

st.dataframe(
    results,
    width="stretch",
    hide_index=True,
)
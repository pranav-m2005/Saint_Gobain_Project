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
import plotly.graph_objects as go

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

st.info(
    "The model's performance should be primarily evaluated on an article-wise basis because different article numbers represent distinct product geometries, materials, and welding requirements. "
    "Evaluating the model globally without considering article differences can mask specific strengths or weaknesses tied to particular products."
)

st.markdown(
    """
    ### Model Validation Workflow

    - Train the model using historical production data covering multiple articles.
    - Validate predictions separately for each article number to account for product-specific characteristics.
    - Compare actual vs predicted production times article-wise.
    - Analyze prediction errors to detect systematic biases per article.
    - Review feature importance to understand key drivers of model predictions.
    - Retrain the model whenever new articles are introduced to maintain accuracy.
    """
)

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

st.markdown(
    """
    ### Model Evaluation Metrics

    The overall MAE and RMSE provide a general sense of prediction error; however, these metrics should ideally also be evaluated separately for each article number. Different products naturally have varying cycle time ranges due to their unique geometries and materials, so article-wise error analysis offers a more accurate assessment of model performance.
    """
)

# -------------------------------------------------------
# Actual vs Predicted (Article Colored)
# -------------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions,
    "article_no": df["article_no"].values,
})

fig = px.scatter(
    comparison,
    x="Actual",
    y="Predicted",
    color="article_no",
    title="Actual vs Predicted Production Time by Article",
    labels={"article_no": "Article Number"},
    hover_data=["article_no"],
)

# Add dashed y=x reference line
fig.add_shape(
    type="line",
    x0=comparison["Actual"].min(),
    y0=comparison["Actual"].min(),
    x1=comparison["Actual"].max(),
    y1=comparison["Actual"].max(),
    line=dict(
        color="Black",
        width=1,
        dash="dash",
    ),
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.markdown(
    """
    ### What this graph shows

    - Each point represents one production sample, colored by its article number.
    - The x-axis shows the actual production time, and the y-axis shows the AI predicted production time.
    - The dashed diagonal line represents perfect prediction (Predicted = Actual).
    - Clustering of points by article number is expected due to differing product characteristics.
    - Comparing different articles together can be misleading because each article has its own natural cycle time range and variation.
    - Evaluating prediction accuracy within each article cluster provides clearer insights into model reliability.
    """
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
    title="Prediction Error Distribution Across Articles"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.markdown(
    """
    This histogram shows the distribution of prediction errors aggregated across all articles.

    - Systematic over- or under-prediction for certain article numbers can lead to skewed error distributions.
    - Investigating error patterns by article helps identify specific products where the model may be biased or less accurate.
    - Balanced error distributions centered around zero within each article indicate unbiased predictions.
    """
)

# -------------------------------------------------------
# Article-wise Model Performance
# -------------------------------------------------------

article_summary = comparison.groupby("article_no").apply(
    lambda x: pd.Series({
        "Samples": len(x),
        "Average Actual Time": x["Actual"].mean(),
        "Average Predicted Time": x["Predicted"].mean(),
        "MAE": mean_absolute_error(x["Actual"], x["Predicted"]),
        "Average Error": x["Error"].mean(),
    })
).reset_index()

article_summary = article_summary.sort_values(by="MAE", ascending=False)

st.markdown(
    """
    ### Article-wise Model Performance Summary
    """
)

st.dataframe(
    article_summary,
    width="stretch",
    hide_index=True,
)

st.markdown(
    """
    Engineers should use this summary to identify articles with high prediction errors or systematic biases. These articles may require focused model retraining, additional feature engineering, or data quality checks to improve prediction accuracy for critical products.
    """
)

# -------------------------------------------------------
# Feature Importance
# -------------------------------------------------------

st.markdown(
    """
    ### Feature Contribution Analysis

    CatBoost automatically determines how much each manufacturing variable contributes to the prediction accuracy.

    The article number is expected to be one of the most influential variables because it captures inherent differences in product geometry, welding length, and material characteristics that strongly affect welding time.
    """
)

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

st.markdown(
    """
    Feature importance quantifies the impact of each input variable on the model's predictions.

    This graph helps engineers identify dominant process variables influencing welding time.

    Note that the article number often becomes highly important because different window geometries naturally require different welding durations.
    """
)

# -------------------------------------------------------
# Results Table
# -------------------------------------------------------

st.markdown(
    """
    ### Sample Prediction Validation

    This table compares every actual production record with the AI prediction and the calculated error.
    """
)

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

st.markdown(
    """
    ### How to Interpret the Results Table

    - **Actual**: The true production time recorded.
    - **Predicted**: The AI model's estimated production time.
    - **Error**: The difference between predicted and actual times.

    Positive error values indicate over-prediction by the model, while negative values indicate under-prediction. This helps identify where the model may be less accurate.
    """
)

st.success(
    """
    ### Overall Model Assessment

    While overall metrics like MAE, RMSE, and R² provide useful general insights, article-wise validation remains the most important criterion for judging deployment readiness. Different articles represent distinct products with unique characteristics, so ensuring accurate predictions within each article cluster is critical for reliable manufacturing decision-making.

    It is recommended to perform periodic retraining and focused evaluation whenever substantial new production data or new article numbers are introduced to maintain and improve model accuracy.
    """
)
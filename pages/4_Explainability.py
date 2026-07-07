
import sys
from pathlib import Path

# Add project root to sys.path for imports
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
# Load Data and Model
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

# Ensure only current schema columns are used
expected_columns = [
    "article_no",
    "line_no",
    "operator",
    "total_time",
    "window_length",
    "window_width",
    "cross_section_length",
    "cross_section_width"
]

# Only keep columns that are in expected_columns to avoid KeyError
df = df[[col for col in expected_columns if col in df.columns]]

# Model input features (exclude 'total_time')
input_features = [col for col in feature_columns if col in df.columns and col != "total_time"]
X = df[input_features]

# -------------------------------------------------------
# Load CatBoost Model
# -------------------------------------------------------
model = joblib.load("models/catboost_model.pkl")

# -------------------------------------------------------
# Feature Importance
# -------------------------------------------------------
importance = model.get_feature_importance()
importance_df = pd.DataFrame(
    {
        "Feature": input_features,
        "Importance": importance
    }
).sort_values(by="Importance", ascending=False)

# -------------------------------------------------------
# Title and Introduction
# -------------------------------------------------------
st.title("🧠 AI Model Explainability Dashboard")
st.markdown("---")
st.info("""
This page provides transparency into how the CatBoost AI model predicts welding cycle times. By examining feature importance and analyzing predictions for specific production samples, engineers can understand which factors most influence the model's output.

**Important:** Always compare and interpret results within the same Article Number. Different articles represent different geometries, materials, and manufacturing complexities, so direct comparison across articles is not meaningful.
""")

st.markdown("""
### Explainability Workflow
1. Select a production sample using the slider.
2. Review the predicted and actual cycle time for that sample.
3. Analyze feature importance for the model.
4. View historical statistics for the selected Article Number.
5. Interpret the engineering implications for process optimization.
""")

# -------------------------------------------------------
# Sample Selection and Prediction
# -------------------------------------------------------
sample_index = st.slider(
    "Select Production Sample",
    0,
    len(X) - 1,
    0
)
selected = X.iloc[[sample_index]]
selected_full = df.iloc[[sample_index]]
selected_article = selected_full.iloc[0]["article_no"]
selected_operator = selected_full.iloc[0]["operator"]
selected_line = selected_full.iloc[0]["line_no"]
actual_time = selected_full.iloc[0]["total_time"]
predicted_time = model.predict(selected)[0]
prediction_error = predicted_time - actual_time

# -------------------------------------------------------
# Display Sample Details and Historical Stats
# -------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Selected Sample Details")
    st.markdown(f"- **Article Number:** `{selected_article}`")
    st.markdown(f"- **Operator:** `{selected_operator}`")
    st.markdown(f"- **Line Number:** `{selected_line}`")
    st.markdown(f"- **Predicted Time:** `{predicted_time:.2f} sec`")
    st.markdown(f"- **Actual Time:** `{actual_time:.2f} sec`")
    st.markdown(f"- **Prediction Error:** `{prediction_error:.2f} sec`")
    st.markdown("""
    **Interpretation:**  
    The model prediction is compared to the actual recorded cycle time for this production sample.  
    **Engineers should only compare prediction errors within the same Article Number** to ensure meaningful assessment, due to differences in product geometry and complexity.
    """)
with c2:
    st.subheader("Historical Statistics for Selected Article")
    article_data = df[df["article_no"] == selected_article]
    st.markdown(f"- **Sample Count:** `{len(article_data)}`")
    st.markdown(f"- **Average Cycle Time:** `{article_data['total_time'].mean():.2f} sec`")
    st.markdown(f"- **Min Cycle Time:** `{article_data['total_time'].min():.2f} sec`")
    st.markdown(f"- **Max Cycle Time:** `{article_data['total_time'].max():.2f} sec`")
    st.markdown("""
    **What does this show?**  
    These statistics summarize the historical welding times for the selected Article Number.  
    Use these as a baseline to compare individual sample performance and prediction accuracy, always within the same article.
    """)

st.divider()

# -------------------------------------------------------
# Feature Importance Visualization
# -------------------------------------------------------
st.subheader("Global Feature Importance")
fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="Feature Importance for Welding Time Prediction"
)
fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=500
)
fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**What does this chart show?**  
This horizontal bar chart ranks all input features by their influence on the model's welding time predictions.

- Each bar's length indicates the feature's relative importance.
- Features with higher scores have a stronger impact on predicted cycle time.
- For example, if `article_no` ranks highest, it means the model relies most on the article identity, which encodes geometry, material, and process complexity.

**How should engineers interpret this?**  
Use this chart to see which input parameters most drive the AI model's decisions.  
Focus process improvement efforts on the most important features.  
**Remember:** Only compare importance and prediction results within the same Article Number.
""")

# -------------------------------------------------------
# Feature Importance Table
# -------------------------------------------------------
st.subheader("Feature Importance Ranking Table")
st.dataframe(
    importance_df.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
st.markdown("""
**How to use this table:**  
This table lists all input features in descending order of their influence on the AI model.  
Engineers should prioritize process control and optimization efforts on features at the top of this list, as these have the greatest impact on welding cycle time predictions.
""")

# -------------------------------------------------------
# Selected Production Sample Display
# -------------------------------------------------------
st.subheader("Selected Production Sample")
st.dataframe(
    selected_full.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
st.markdown("""
**What does this show?**  
This table displays the exact input features for the selected production sample.  
These values are what the AI model uses to make its prediction.

**Interpretation:**  
When investigating process issues or prediction errors, always compare samples with the same `article_no`.  
Comparing across different article numbers is not meaningful due to differences in geometry and manufacturing complexity.
""")

# -------------------------------------------------------
# Engineering Conclusion and Interpretation
# -------------------------------------------------------
st.subheader("Engineering Interpretation and Conclusion")
top_feature = importance_df.iloc[0]["Feature"]
top_importance = importance_df.iloc[0]["Importance"]
st.success(
    f"The most influential feature is **{top_feature}** (Importance Score = {top_importance:.2f})."
)
st.markdown("""
**Why is this important?**  
The top-ranked feature is the primary driver of the model's predictions.  
For example, if `article_no` is highest, it reflects the fact that article identity (and thus geometry, material, and complexity) dominates welding cycle time.

**How should engineers use this information?**
- Focus on controlling and optimizing the highest-importance features for the best impact on cycle time.
- Use feature importance as a guide to prioritize process improvement projects.
- Always interpret results and draw conclusions within the same `article_no` to avoid misleading comparisons.

**Conclusion:**  
Feature importance analysis builds trust in AI-assisted manufacturing by making model decisions transparent.  
Engineers should use these insights to support robust, article-specific process optimization, confident that the AI model's predictions are grounded in the most relevant production factors.
""")

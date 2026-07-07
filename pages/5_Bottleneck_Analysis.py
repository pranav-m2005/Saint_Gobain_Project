"""
=========================================================
AI Bottleneck Analysis
Saint-Gobain Welding Time Predictor
=========================================================
"""

import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.trainer import ModelTrainer

st.set_page_config(page_title="Bottleneck Analysis", page_icon="🚦", layout="wide")

@st.cache_data
def load_data():
    loader = WeldingDataLoader()
    df = loader.load_data()
    return DataPreprocessor.preprocess(df)

@st.cache_data
def load_model():
    model_path = ROOT / 'models' / 'catboost_model.pkl'
    model = joblib.load(model_path)
    return model

raw_df = load_data()
model = load_model()
trainer = ModelTrainer()

# Prepare feature matrix and predict
X = raw_df[trainer.feature_columns]
raw_df['Predicted_Time'] = model.predict(X)

# Calculate Time_Loss and Efficiency
raw_df['Time_Loss'] = (raw_df['total_time'] - raw_df['Predicted_Time']).abs()
raw_df['Efficiency'] = (raw_df['Predicted_Time'] / raw_df['total_time']) * 100

st.title("🚦 AI Bottleneck Analysis")
st.markdown("Analyse production losses by comparing actual production time with AI predicted time. Comparisons are most meaningful within the same Article Number.")

articles = ["All"] + sorted(raw_df["article_no"].astype(str).unique().tolist())
selected = st.selectbox("Select Article Number", articles)

if selected != "All":
    df = raw_df[raw_df["article_no"].astype(str) == selected].copy()
else:
    df = raw_df.copy()

avg_actual = df["total_time"].mean()
avg_pred = df["Predicted_Time"].mean()
avg_loss = df["Time_Loss"].mean()
avg_efficiency = df["Efficiency"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Actual Time", f"{avg_actual:.2f} s")
c2.metric("Average Predicted Time", f"{avg_pred:.2f} s")
c3.metric("Average Time Loss", f"{avg_loss:.2f} s")
c4.metric("Average Efficiency", f"{avg_efficiency:.1f}%")

st.markdown("### Average Time Loss by Article Number")
article_loss = df.groupby("article_no", as_index=False)["Time_Loss"].mean().sort_values("Time_Loss", ascending=False)
fig1 = px.bar(article_loss, x="article_no", y="Time_Loss", title="Average Time Loss by Article Number")
st.plotly_chart(fig1, use_container_width=True)
st.markdown(
    """
    This bar chart displays the average time loss for each Article Number.
    Time loss represents the absolute difference between actual production time and AI predicted time.
    Higher values indicate articles where actual production time deviates most from expected time, highlighting potential bottlenecks specific to certain products.
    Analyzing this helps prioritize process improvements for articles with the largest discrepancies.
    """
)

st.markdown("### Average Time Loss by Operator")
operator_loss = df.groupby("operator", as_index=False)["Time_Loss"].mean().sort_values("Time_Loss", ascending=False)
fig2 = px.bar(operator_loss, x="operator", y="Time_Loss", title="Average Time Loss by Operator")
st.plotly_chart(fig2, use_container_width=True)
st.markdown(
    """
    This chart shows the average time loss for each operator.
    Operators with consistently high time loss may indicate variability in performance or training needs.
    Identifying such operators allows targeted interventions to improve efficiency and reduce production delays.
    """
)

st.markdown("### Average Time Loss by Production Line")
line_loss = df.groupby("line_no", as_index=False)["Time_Loss"].mean().sort_values("Time_Loss", ascending=False)
fig3 = px.bar(line_loss, x="line_no", y="Time_Loss", title="Average Time Loss by Production Line")
st.plotly_chart(fig3, use_container_width=True)
st.markdown(
    """
    This bar chart highlights production lines with the highest average time loss.
    Lines with elevated time loss may suffer from equipment issues, workflow inefficiencies, or other bottlenecks.
    Focusing on these lines helps prioritize maintenance and process optimization efforts.
    """
)

st.markdown("### Actual vs Predicted Production Time")
fig4 = px.scatter(
    df,
    x="Predicted_Time",
    y="total_time",
    color="article_no",
    title="Actual vs Predicted Production Time",
    labels={"Predicted_Time": "Predicted Time (s)", "total_time": "Actual Time (s)", "article_no": "Article Number"},
    hover_data=["operator", "line_no", "Time_Loss", "Efficiency"]
)
fig4.add_shape(
    type="line",
    x0=df["Predicted_Time"].min(),
    y0=df["Predicted_Time"].min(),
    x1=df["Predicted_Time"].max(),
    y1=df["Predicted_Time"].max(),
    line=dict(color="Red", dash="dash"),
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown(
    """
    The scatter plot compares actual production times against AI predicted times for each record.
    Points along the red dashed line indicate perfect prediction.
    Deviations from this line represent inefficiencies or delays.
    Coloring by Article Number helps identify if certain articles consistently deviate, revealing product-specific bottlenecks.
    """
)

st.markdown("### Detailed Production Records")
display_cols = ["article_no", "operator", "line_no", "total_time", "Predicted_Time", "Time_Loss", "Efficiency"]
st.dataframe(df[display_cols].sort_values("Time_Loss", ascending=False), use_container_width=True, hide_index=True)

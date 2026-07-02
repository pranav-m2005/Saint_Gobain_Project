
"""
Version 1 placeholder due to response size limits.
This file should be replaced by the full implementation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.bottleneck_analysis import BottleneckAnalysis


st.set_page_config(
    page_title="AI Bottleneck Analysis",
    page_icon="🚦",
    layout="wide"
)


@st.cache_data
def load_data():
    loader = WeldingDataLoader()
    df = loader.load_data()
    return DataPreprocessor.preprocess(df)


df = load_data()

engine = BottleneckAnalysis()
df = engine.predict_dataset(df)

summary = engine.overall_summary(df)

st.title("🚦 AI Bottleneck Analysis")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Avg Actual", f"{summary['Average Actual Time']:.2f}s")
c2.metric("Avg Predicted", f"{summary['Average Predicted Time']:.2f}s")
c3.metric("Avg Loss", f"{summary['Average Time Loss']:.2f}s")
c4.metric("Efficiency", f"{summary['Average Efficiency']:.2f}%")

st.divider()

st.subheader("Top Articles with Time Loss")
article = engine.article_ranking(df)
fig = px.bar(article.head(10),
             x="Average_Loss",
             y="article_no",
             orientation="h",
             title="Average Time Loss by Article")
st.plotly_chart(fig, width="stretch")
st.dataframe(article, width="stretch", hide_index=True)

st.subheader("Operator Ranking")
operator = engine.operator_ranking(df)
fig = px.bar(operator,
             x="Average_Loss",
             y="operator",
             orientation="h")
st.plotly_chart(fig, width="stretch")

st.subheader("Production Line Ranking")
line = engine.line_ranking(df)
fig = px.bar(line,
             x="Average_Loss",
             y="line_no",
             orientation="h")
st.plotly_chart(fig, width="stretch")

st.subheader("Cross Section Ranking")
cross = engine.cross_section_ranking(df)
cross["Cross Section"] = (
    cross["cross_section_length"].astype(str)
    + " × "
    + cross["cross_section_width"].astype(str)
)
fig = px.bar(
    cross.head(10),
    x="Average_Loss",
    y="Cross Section",
    orientation="h"
)
st.plotly_chart(fig, width="stretch")

st.subheader("Worst Production Records")
st.dataframe(engine.worst_cases(df,10), width="stretch", hide_index=True)

st.subheader("Best Production Records")
st.dataframe(engine.best_cases(df,10), width="stretch", hide_index=True)

impact = engine.production_impact(df)
st.subheader("Production Impact")
st.write(impact)

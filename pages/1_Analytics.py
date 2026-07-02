"""
=========================================================
Analytics Page
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
import plotly.express as px

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.validator import DataValidator


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Production Analytics",
    page_icon="📊",
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

DataValidator.validate(df)

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------

st.sidebar.title("Filters")

operator = st.sidebar.selectbox(
    "Operator",
    ["All"] + sorted(df["operator"].astype(str).unique().tolist()),
)

line = st.sidebar.selectbox(
    "Production Line",
    ["All"] + sorted(df["line_no"].unique().tolist()),
)

# -------------------------------------------------------
# Apply Filters
# -------------------------------------------------------

filtered_df = df.copy()

if operator != "All":
    filtered_df = filtered_df[
        filtered_df["operator"].astype(str) == operator
    ]

if line != "All":
    filtered_df = filtered_df[
        filtered_df["line_no"] == line
    ]

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("📊 Production Analytics Dashboard")

st.markdown("---")

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Samples",
    len(filtered_df),
)

c2.metric(
    "Average Cycle Time",
    f"{filtered_df['total_time'].mean():.2f} sec",
)

c3.metric(
    "Fastest Cycle",
    f"{filtered_df['total_time'].min():.2f} sec",
)

c4.metric(
    "Slowest Cycle",
    f"{filtered_df['total_time'].max():.2f} sec",
)

st.divider()

# -------------------------------------------------------
# Row 1
# -------------------------------------------------------

left, right = st.columns(2)

with left:

    fig = px.histogram(
        filtered_df,
        x="total_time",
        nbins=12,
        title="Cycle Time Distribution",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

with right:

    fig = px.box(
        filtered_df,
        x="operator",
        y="total_time",
        color="operator",
        title="Operator Comparison",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

# -------------------------------------------------------
# Row 2
# -------------------------------------------------------

left, right = st.columns(2)

with left:

    fig = px.scatter(
        filtered_df,
        x="window_length",
        y="total_time",
        color="operator",
        size="cross_section_width",
        hover_data=["article_no"],
        title="Window Length vs Total Time",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

with right:

    fig = px.scatter(
        filtered_df,
        x="cross_section_length",
        y="total_time",
        color="operator",
        size="window_width",
        hover_data=["article_no"],
        title="Cross Section Length vs Total Time",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

# -------------------------------------------------------
# Correlation Matrix
# -------------------------------------------------------

corr = filtered_df.select_dtypes(
    include=["int64", "float64"]
).corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix",
)

st.plotly_chart(
    fig,
    width="stretch",
)

# -------------------------------------------------------
# Raw Dataset
# -------------------------------------------------------

st.subheader("Production Dataset")

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True,
)
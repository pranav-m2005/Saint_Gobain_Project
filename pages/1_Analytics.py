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

article = st.sidebar.selectbox(
    "Article Number",
    ["All"] + sorted(df["article_no"].unique().tolist()),
)

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

if article != "All":
    filtered_df = filtered_df[
        filtered_df["article_no"] == article
    ]

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

st.info(
    """
    This page analyzes production behavior using live data sourced directly from Google Sheets.
    It helps identify process variations and operator performance across the production lines.
    Since different article numbers correspond to different geometries and materials,
    article-wise comparisons are essential for meaningful analysis and interpretation.
    """
)

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

st.markdown(
    """
    ### KPI Interpretation

    - **Samples**: Number of production cycles considered after applying filters, indicating data volume and reliability.
    - **Average Cycle Time**: Mean welding time, reflecting overall process efficiency.
    - **Fastest Cycle**: Minimum recorded cycle time, showing best-case performance.
    - **Slowest Cycle**: Maximum recorded cycle time, highlighting potential bottlenecks or anomalies.

    **Key Insight:** KPIs should be compared within the same article number to draw meaningful conclusions due to inherent differences in product geometry and material.
    """
)

# -------------------------------------------------------
# Row 1
# -------------------------------------------------------

left, right = st.columns(2)

with left:

    avg_cycle_time_by_article = (
        filtered_df.groupby("article_no")["total_time"]
        .mean()
        .reset_index()
        .sort_values("total_time")
    )

    fig = px.bar(
        avg_cycle_time_by_article,
        x="article_no",
        y="total_time",
        title="Average Cycle Time by Article",
        labels={"article_no": "Article Number", "total_time": "Average Cycle Time (sec)"},
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown(
        """
        #### Why Article Comparison is Primary

        Different articles correspond to distinct product geometries and materials,
        which significantly influence welding time and process parameters.

        Comparing cycle times across articles provides foundational insights into manufacturing efficiency,
        enabling focused analysis on product-specific process characteristics.
        """
    )

with right:

    fig = px.box(
        filtered_df,
        x="article_no",
        y="total_time",
        color="article_no",
        title="Cycle Time Distribution by Article",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown(
        """
        This box plot reveals the distribution of cycle times for each article number.
        It highlights the consistency and variability within each product category,
        showing medians, quartiles, and outliers.

        Understanding these patterns helps pinpoint articles with stable production versus those with high variability,
        guiding quality improvement and process optimization efforts.
        """
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
        color="article_no",
        size="cross_section_width",
        hover_data=["article_no"],
        title="Window Length vs Total Time",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown(
        """
        This scatter plot explores the relationship between window length and total welding cycle time.
        Each point's size represents the cross section width, and color indicates the article number,
        allowing multidimensional analysis.

        Clustering by article number indicates the influence of product geometry on cycle times,
        helping identify product-specific process dependencies.
        """
    )

with right:

    fig = px.scatter(
        filtered_df,
        x="cross_section_length",
        y="total_time",
        color="article_no",
        size="window_width",
        hover_data=["article_no"],
        title="Cross Section Length vs Total Time",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown(
        """
        This scatter plot shows how cross section length influences production time,
        with point size representing window width and color indicating the article number.

        Clustering by article highlights product geometry's impact on welding duration,
        assisting in understanding product-specific process behavior.
        """
    )

# -------------------------------------------------------
# Correlation Matrix
# -------------------------------------------------------

st.markdown(
    """
    ### Feature Relationship Analysis

    Correlation measures the strength and direction of linear relationships between variables.
    Understanding these relationships helps identify key drivers affecting the welding process and cycle times.
    """
)

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

st.markdown(
    """
    Positive correlation indicates variables increase together, negative means one increases as the other decreases,
    and near-zero suggests little linear relationship.

    Heatmaps visually represent these correlations for quick interpretation.

    Note that correlation does not imply causation.

    Article number strongly influences several dimensional variables and should always be considered first before interpreting correlations.
    """
)

# -------------------------------------------------------
# Raw Dataset
# -------------------------------------------------------

st.markdown(
    """
    ### Filtered Production Records

    The table below reflects all active filters and serves as the data source behind every visualization on this page.
    """
)

st.subheader("Production Dataset")

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True,
)

st.success(
    """
    Summary:

    - Always compare articles first to understand product-specific cycle times.
    - Then compare operators within the same article for fair performance evaluation.
    - Use geometry-related variables to explain differences in cycle times.
    - Avoid directly comparing different products as their inherent characteristics vary.
    """
)
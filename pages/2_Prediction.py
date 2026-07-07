"""
=========================================================
Prediction Page
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

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.predictor import ModelPredictor


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Cycle Time Prediction",
    page_icon="🤖",
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

predictor = ModelPredictor()

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🤖 Welding Cycle Time Prediction")

st.markdown("---")

st.info(
    """
    Predictions are most meaningful when the correct Article Number is selected, as each article represents a unique geometry, material, and manufacturing process. 
    The AI model leverages this article-centric information to provide precise cycle time estimations tailored to the specific characteristics of each product.
    """
)

# -------------------------------------------------------
# Prediction Workflow
# -------------------------------------------------------

st.markdown(
    """
    ### Prediction Workflow
    
    - Select the correct article number representing the product.
    - Enter all relevant dimensions accurately.
    - Choose the production line and operator responsible.
    - Generate the predicted cycle time.
    - Compare the predicted cycle time with historical data of similar articles for informed production planning.
    """
)

# -------------------------------------------------------
# Input Guidelines
# -------------------------------------------------------

st.markdown(
    """
    ### Input Guidelines
    
    - **Article Number**: Select the article number corresponding to the product being manufactured.
    - **Window Length**: Enter the length of the window in millimeters.
    - **Window Width**: Enter the width of the window in millimeters.
    - **Cross Section Length**: Specify the length of the cross section in millimeters.
    - **Cross Section Width**: Specify the width of the cross section in millimeters.
    - **Production Line**: Choose the production line where welding will occur.
    - **Operator**: Select the operator responsible for the welding process.
    
    **Note:** The Article Number is the most important identifier because each article represents a different window design and material.
    """
)

# -------------------------------------------------------
# Inputs
# -------------------------------------------------------

left, right = st.columns(2)

with left:

    article_no = st.selectbox(
        "Article Number",
        sorted(df["article_no"].unique())
    )

    window_length = st.number_input(
        "Window Length (mm)",
        min_value=100,
        value=1200,
        step=1,
    )

    window_width = st.number_input(
        "Window Width (mm)",
        min_value=100,
        value=700,
        step=1,
    )

with right:

    cross_section_length = st.number_input(
        "Cross Section Length (mm)",
        min_value=10,
        value=60,
        step=1,
    )

    cross_section_width = st.number_input(
        "Cross Section Width (mm)",
        min_value=10,
        value=40,
        step=1,
    )

    line_no = st.selectbox(
        "Production Line",
        sorted(df["line_no"].unique())
    )

    operator = st.selectbox(
        "Operator",
        sorted(df["operator"].astype(str).unique())
    )

# -------------------------------------------------------
# Article Historical Data Display
# -------------------------------------------------------

hist_left, hist_right = st.columns(2)

with hist_left:
    st.subheader(f"Historical Samples for Article: {article_no}")
    df_article = df[df["article_no"] == article_no]
    st.dataframe(df_article, use_container_width=True)

with hist_right:
    st.subheader("Historical Statistics")
    sample_count = df_article.shape[0]
    avg_cycle_time = df_article["total_time"].mean() if sample_count > 0 else 0
    min_cycle_time = df_article["total_time"].min() if sample_count > 0 else 0
    max_cycle_time = df_article["total_time"].max() if sample_count > 0 else 0

    st.markdown(f"- **Sample Count:** {sample_count}")
    st.markdown(f"- **Average Cycle Time:** {avg_cycle_time:.2f} sec")
    st.markdown(f"- **Minimum Cycle Time:** {min_cycle_time:.2f} sec")
    st.markdown(f"- **Maximum Cycle Time:** {max_cycle_time:.2f} sec")

st.markdown(
    """
    Comparing the predicted cycle time against historical values for the *same article* is critical. 
    This ensures that engineers base their decisions on relevant data reflecting the specific product's characteristics, rather than on data from different articles which may have vastly different manufacturing complexities.
    """
)

# -------------------------------------------------------
# Before Predicting
# -------------------------------------------------------

st.markdown(
    """
    ### Before Predicting
    
    Please verify that all dimensions entered are correct. 
    Ensure the selected production line and operator match the actual production setup.
    Confirm that the Article Number corresponds exactly to the product being manufactured to obtain reliable predictions.
    """
)

st.divider()

# -------------------------------------------------------
# Predict
# -------------------------------------------------------

if st.button("Predict Cycle Time", use_container_width=True):

    prediction = predictor.predict(
        article_no=article_no,
        window_length=window_length,
        window_width=window_width,
        cross_section_length=cross_section_length,
        cross_section_width=cross_section_width,
        line_no=line_no,
        operator=operator,
    )

    st.success("Prediction Completed Successfully")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimated Cycle Time",
        f"{prediction:.2f} sec"
    )

    c2.metric(
        "Estimated Time (Minutes)",
        f"{prediction/60:.2f} min"
    )

    c3.metric(
        "Estimated Throughput",
        f"{3600/prediction:.2f} Windows/hr"
    )

    st.markdown(
        """
        ### Understanding the Prediction Results
        
        - **Estimated Cycle Time (seconds)**: The predicted duration for completing the welding cycle.
        - **Estimated Time (Minutes)**: The same duration converted into minutes for easier comprehension.
        - **Estimated Throughput (windows/hour)**: Number of windows that can be produced per hour based on the predicted cycle time.
        
        Throughput is crucial for production planning as it directly impacts scheduling, resource allocation, and meeting delivery deadlines.
        
        **Key Insight:** Even a small reduction in cycle time can significantly increase hourly production capacity.
        """
    )

    st.divider()

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({

        "Parameter":[
            "Article Number",
            "Window Length",
            "Window Width",
            "Cross Section Length",
            "Cross Section Width",
            "Production Line",
            "Operator",
            "Predicted Time (sec)"
        ],

        "Value":[
            article_no,
            window_length,
            window_width,
            cross_section_length,
            cross_section_width,
            line_no,
            operator,
            round(prediction,2)
        ]

    })

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        ### Why This Summary Matters
        
        This table documents all inputs used by the AI model for prediction. 
        It can be used for production planning and helps compare different article numbers before manufacturing.
        """
    )

    st.divider()

    # -------------------------------------------------------
    # Historical Comparison Section
    # -------------------------------------------------------

    st.subheader("Historical Comparison")

    diff = prediction - avg_cycle_time
    diff_percent = (diff / avg_cycle_time * 100) if avg_cycle_time != 0 else 0

    comparison_df = pd.DataFrame({
        "Metric": ["Historical Average (sec)", "Predicted Time (sec)", "Difference (sec)"],
        "Value": [f"{avg_cycle_time:.2f}", f"{prediction:.2f}", f"{diff:.2f}"]
    })

    st.table(comparison_df)

    if abs(diff_percent) <= 5:
        st.success(
            f"Prediction is within ±5% of the historical average ({diff_percent:.2f}%). This indicates strong alignment with past performance."
        )
    elif abs(diff_percent) <= 15:
        st.warning(
            f"Prediction differs by between 5% and 15% from historical average ({diff_percent:.2f}%). Review for potential process changes or anomalies."
        )
    else:
        st.error(
            f"Prediction differs by more than 15% from historical average ({diff_percent:.2f}%). Investigate possible causes such as changes in geometry, material, or process."
        )

    st.divider()

    st.markdown(
        """
        ### AI Decision Interpretation
        
        The model classifies predicted cycle times into low, moderate, and high production durations to assist engineers in quick decision making.
        """
    )

    if prediction < 180:

        st.success(
            "Expected production time is LOW. This indicates high productivity with better utilization of equipment and shorter waiting times between processes."
        )

    elif prediction < 240:

        st.warning(
            "Expected production time is MODERATE. Production is acceptable but there may be opportunities to optimize processes or reduce cycle time."
        )

    else:

        st.error(
            "Expected production time is HIGH. This may indicate bottlenecks due to complex geometry, material, or process challenges. Review dimensions, tooling, and article characteristics to improve efficiency."
        )

    st.success(
        """
        ### Practical Applications
        
        This prediction page supports article-wise production planning by providing accurate cycle time estimates tailored to each product. 
        It enables comparison of new jobs with historical performance of the same article, helping to identify deviations and optimize scheduling.
        The insights facilitate better manpower allocation, resource management, and bottleneck prevention, ultimately improving manufacturing efficiency and delivery reliability.
        """
    )
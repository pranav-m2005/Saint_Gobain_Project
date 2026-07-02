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

st.write(
    "Enter the window profile details below to estimate the total welding cycle time."
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
        "Estimated Cycle Time",
        f"{prediction/60:.2f} min"
    )

    c3.metric(
        "Estimated Throughput",
        f"{3600/prediction:.2f} Windows/hr"
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

    st.divider()

    st.subheader("Prediction Interpretation")

    if prediction < 180:

        st.success(
            "Expected production time is LOW. High throughput expected."
        )

    elif prediction < 240:

        st.warning(
            "Expected production time is MODERATE."
        )

    else:

        st.error(
            "Expected production time is HIGH. This profile may reduce production throughput."
        )
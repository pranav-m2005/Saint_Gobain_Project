"""
=========================================================
AI Bottleneck Analysis Engine
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import joblib
import pandas as pd

from core.trainer import ModelTrainer


class BottleneckAnalysis:

    def __init__(self, model_path="models/catboost_model.pkl"):

        self.model = joblib.load(model_path)

        trainer = ModelTrainer()

        self.feature_columns = trainer.feature_columns

    # --------------------------------------------------
    # Predict Entire Dataset
    # --------------------------------------------------

    def predict_dataset(self, df):

        prediction_df = df.copy()

        X = prediction_df[self.feature_columns]

        prediction_df["predicted_time"] = self.model.predict(X)

        prediction_df["time_loss"] = (
            prediction_df["total_time"]
            - prediction_df["predicted_time"]
        )

        prediction_df["efficiency"] = (
            prediction_df["predicted_time"]
            / prediction_df["total_time"]
        ) * 100

        return prediction_df

    # --------------------------------------------------
    # Overall Summary
    # --------------------------------------------------

    def overall_summary(self, df):

        return {

            "Average Actual Time":
                round(df["total_time"].mean(),2),

            "Average Predicted Time":
                round(df["predicted_time"].mean(),2),

            "Average Time Loss":
                round(df["time_loss"].mean(),2),

            "Maximum Time Loss":
                round(df["time_loss"].max(),2),

            "Average Efficiency":
                round(df["efficiency"].mean(),2)

        }

    # --------------------------------------------------
    # Article Ranking
    # --------------------------------------------------

    def article_ranking(self, df):

        article = (

            df.groupby("article_no")

            .agg(

                Samples=("article_no","count"),

                Average_Loss=("time_loss","mean"),

                Maximum_Loss=("time_loss","max"),

                Average_Efficiency=("efficiency","mean")

            )

            .reset_index()

        )

        article = article.sort_values(

            by="Average_Loss",

            ascending=False

        )

        return article

    # --------------------------------------------------
    # Operator Ranking
    # --------------------------------------------------

    def operator_ranking(self, df):

        operator = (

            df.groupby("operator")

            .agg(

                Samples=("operator","count"),

                Average_Loss=("time_loss","mean"),

                Average_Efficiency=("efficiency","mean")

            )

            .reset_index()

        )

        operator = operator.sort_values(

            by="Average_Loss",

            ascending=False

        )

        return operator

    # --------------------------------------------------
    # Line Ranking
    # --------------------------------------------------

    def line_ranking(self, df):

        line = (

            df.groupby("line_no")

            .agg(

                Samples=("line_no","count"),

                Average_Loss=("time_loss","mean"),

                Average_Efficiency=("efficiency","mean")

            )

            .reset_index()

        )

        line = line.sort_values(

            by="Average_Loss",

            ascending=False

        )

        return line

    # --------------------------------------------------
    # Cross Section Ranking
    # --------------------------------------------------

    def cross_section_ranking(self, df):

        cross = (

            df.groupby(

                [

                    "cross_section_length",

                    "cross_section_width"

                ]

            )

            .agg(

                Samples=("article_no","count"),

                Average_Loss=("time_loss","mean"),

                Average_Efficiency=("efficiency","mean")

            )

            .reset_index()

        )

        cross = cross.sort_values(

            by="Average_Loss",

            ascending=False

        )

        return cross

    # --------------------------------------------------
    # Top Worst Records
    # --------------------------------------------------

    def worst_cases(self, df, top=10):

        worst = df.sort_values(

            by="time_loss",

            ascending=False

        )

        return worst.head(top)

    # --------------------------------------------------
    # Best Records
    # --------------------------------------------------

    def best_cases(self, df, top=10):

        best = df.sort_values(

            by="time_loss",

            ascending=True

        )

        return best.head(top)

    # --------------------------------------------------
    # Production Impact
    # --------------------------------------------------

    def production_impact(self, df):

        total_loss = df["time_loss"].sum()

        return {

            "Total Time Lost (sec)":
                round(total_loss,2),

            "Total Time Lost (min)":
                round(total_loss/60,2),

            "Average Loss / Window":
                round(df["time_loss"].mean(),2)

        }


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from core.data_loader import WeldingDataLoader
    from core.preprocessing import DataPreprocessor

    loader = WeldingDataLoader()

    df = loader.load_data()

    df = DataPreprocessor.preprocess(df)

    engine = BottleneckAnalysis()

    df = engine.predict_dataset(df)

    print("\nOverall Summary\n")

    print(engine.overall_summary(df))

    print("\nTop Articles\n")

    print(engine.article_ranking(df))

    print("\nWorst Cases\n")

    print(engine.worst_cases(df))
    
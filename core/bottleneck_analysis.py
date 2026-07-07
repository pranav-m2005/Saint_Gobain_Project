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

        prediction_df["prediction_error"] = (
            prediction_df["total_time"]
            - prediction_df["predicted_time"]
        )

        # For backward compatibility
        prediction_df["time_loss"] = prediction_df["prediction_error"]

        prediction_df["efficiency"] = (
            prediction_df["predicted_time"]
            / prediction_df["total_time"]
        ) * 100

        prediction_df["delay_percentage"] = (
            (prediction_df["total_time"] - prediction_df["predicted_time"])
            / prediction_df["predicted_time"]
        ) * 100

        return prediction_df

    # --------------------------------------------------
    # Filter by Article
    # --------------------------------------------------

    @staticmethod
    def filter_article(df, article):

        if article == "All Articles":
            return df.copy()

        return df[df["article_no"] == article].reset_index(drop=True)

    # --------------------------------------------------
    # Overall Summary
    # --------------------------------------------------

    def overall_summary(self, df):

        return {
            "Total Samples": len(df),
            "Total Articles": df["article_no"].nunique(),
            "Average Actual Time": round(df["total_time"].mean(), 2),
            "Average Predicted Time": round(df["predicted_time"].mean(), 2),
            "Average Prediction Error": round(df["prediction_error"].mean(), 2),
            "Maximum Prediction Error": round(df["prediction_error"].max(), 2),
            "Average Efficiency": round(df["efficiency"].mean(), 2),
        }

    # --------------------------------------------------
    # Article Summary
    # --------------------------------------------------

    def article_summary(self, df, article):

        article_df = self.filter_article(df, article)

        if article_df.empty:
            return {}

        return {
            "Sample Count": len(article_df),
            "Average Actual Time": round(article_df["total_time"].mean(), 2),
            "Average Predicted Time": round(article_df["predicted_time"].mean(), 2),
            "Average Prediction Error": round(article_df["prediction_error"].mean(), 2),
            "Maximum Prediction Error": round(article_df["prediction_error"].max(), 2),
            "Average Efficiency": round(article_df["efficiency"].mean(), 2),
        }

    # --------------------------------------------------
    # Article Ranking
    # --------------------------------------------------

    def article_ranking(self, df):

        article = (
            df.groupby("article_no")
            .agg(
                Samples=("article_no", "count"),
                Average_Actual_Time=("total_time", "mean"),
                Average_Predicted_Time=("predicted_time", "mean"),
                Average_Error=("prediction_error", "mean"),
                Maximum_Error=("prediction_error", "max"),
                Average_Efficiency=("efficiency", "mean"),
            )
            .reset_index()
        )

        article = article.sort_values(
            by="Average_Error",
            ascending=False
        ).round(2)

        return article

    # --------------------------------------------------
    # Operator Ranking
    # --------------------------------------------------

    def operator_ranking(self, df):

        operator = (
            df.groupby("operator")
            .agg(
                Samples=("operator", "count"),
                Average_Error=("prediction_error", "mean"),
                Average_Efficiency=("efficiency", "mean")
            )
            .reset_index()
        )

        operator = operator.sort_values(
            by="Average_Error",
            ascending=False
        ).round(2)

        return operator

    # --------------------------------------------------
    # Line Ranking
    # --------------------------------------------------

    def line_ranking(self, df):

        line = (
            df.groupby("line_no")
            .agg(
                Samples=("line_no", "count"),
                Average_Error=("prediction_error", "mean"),
                Average_Efficiency=("efficiency", "mean")
            )
            .reset_index()
        )

        line = line.sort_values(
            by="Average_Error",
            ascending=False
        ).round(2)

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
                Samples=("article_no", "count"),
                Average_Error=("prediction_error", "mean"),
                Average_Efficiency=("efficiency", "mean")
            )
            .reset_index()
        )

        cross = cross.sort_values(
            by="Average_Error",
            ascending=False
        ).round(2)

        return cross

    # --------------------------------------------------
    # Top Worst Records
    # --------------------------------------------------

    def worst_cases(self, df, top=10):

        worst = df.sort_values(
            by="prediction_error",
            ascending=False
        )

        return worst.head(top)

    # --------------------------------------------------
    # Best Records
    # --------------------------------------------------

    def best_cases(self, df, top=10):

        best = df.sort_values(
            by="prediction_error",
            ascending=True
        )

        return best.head(top)

    # --------------------------------------------------
    # Production Impact
    # --------------------------------------------------

    def production_impact(self, df):

        total_error = df["prediction_error"].sum()

        return {
            "Total Time Lost (sec)": round(total_error, 2),
            "Total Time Lost (min)": round(total_error / 60, 2),
            "Average Loss / Window": round(df["prediction_error"].mean(), 2),
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
"""
=========================================================
Model Predictor
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

from pathlib import Path

import joblib
import pandas as pd


class ModelPredictor:
    """
    Loads the trained model and performs predictions for
    single samples, complete datasets and individual articles.
    """

    def __init__(self):

        root = Path(__file__).resolve().parent.parent
        model_path = root / "models" / "catboost_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}\n\n"
                "Run train_model.py first."
            )

        self.model = joblib.load(model_path)

        self.feature_columns = [
            "article_no",
            "window_length",
            "window_width",
            "cross_section_length",
            "cross_section_width",
            "line_no",
            "operator",
        ]

    # --------------------------------------------------
    # Validate Input Columns
    # --------------------------------------------------

    def _validate_dataframe(self, dataframe):

        missing = [
            col
            for col in self.feature_columns
            if col not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns : {missing}"
            )

    # --------------------------------------------------
    # Predict Single Sample
    # --------------------------------------------------

    def predict(
        self,
        article_no,
        window_length,
        window_width,
        cross_section_length,
        cross_section_width,
        line_no,
        operator,
    ):

        sample = pd.DataFrame(
            {
                "article_no": [article_no],
                "window_length": [window_length],
                "window_width": [window_width],
                "cross_section_length": [cross_section_length],
                "cross_section_width": [cross_section_width],
                "line_no": [line_no],
                "operator": [operator],
            }
        )

        prediction = self.model.predict(sample)[0]

        return round(float(prediction), 2)

    # --------------------------------------------------
    # Predict Complete Dataset
    # --------------------------------------------------

    def predict_dataframe(self, dataframe):

        self._validate_dataframe(dataframe)

        X = dataframe[self.feature_columns].copy()

        predictions = self.model.predict(X)

        return predictions

    # --------------------------------------------------
    # Append Predictions
    # --------------------------------------------------

    def predict_with_dataframe(self, dataframe):

        df = dataframe.copy()

        df["predicted_total_time"] = self.predict_dataframe(df)

        df["prediction_error"] = (
            df["predicted_total_time"] - df["total_time"]
        )

        df["absolute_error"] = (
            df["prediction_error"].abs()
        )

        return df

    # --------------------------------------------------
    # Filter Dataset by Article
    # --------------------------------------------------

    @staticmethod
    def filter_article(dataframe, article):

        if article == "All Articles":
            return dataframe.copy()

        return (
            dataframe[dataframe["article_no"] == article]
            .copy()
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # Predict Particular Article
    # --------------------------------------------------

    def predict_article(self, dataframe, article):

        df = self.filter_article(dataframe, article)

        if df.empty:
            return df

        return self.predict_with_dataframe(df)


if __name__ == "__main__":

    predictor = ModelPredictor()

    prediction = predictor.predict(
        article_no=5801005,
        window_length=1200,
        window_width=700,
        cross_section_length=56,
        cross_section_width=40,
        line_no=1,
        operator="experienced",
    )

    print(f"\nPredicted Cycle Time : {prediction:.2f} sec\n")

    
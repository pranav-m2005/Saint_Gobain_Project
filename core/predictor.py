"""
=========================================================
Model Predictor
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import joblib
import pandas as pd


class ModelPredictor:

    def __init__(self, model_path="models/catboost_model.pkl"):

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
    # Predict Single Window
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
    # Predict Multiple Windows
    # --------------------------------------------------

    def predict_dataframe(self, dataframe):

        dataframe = dataframe[self.feature_columns]

        predictions = self.model.predict(dataframe)

        return predictions

    # --------------------------------------------------
    # Predict and Append
    # --------------------------------------------------

    def predict_with_dataframe(self, dataframe):

        df = dataframe.copy()

        df["predicted_total_time"] = self.predict_dataframe(df)

        return df


# ------------------------------------------------------
# Testing
# ------------------------------------------------------

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
    
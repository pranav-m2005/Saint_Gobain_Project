"""
=========================================================
Model Trainer
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import os
import joblib
import pandas as pd

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelTrainer:

    def __init__(self):

        # ----------------------------------------------
        # Features used for prediction
        # ----------------------------------------------

        self.feature_columns = [
            "article_no",
            "window_length",
            "window_width",
            "cross_section_length",
            "cross_section_width",
            "line_no",
            "operator",
        ]

        self.target_column = "total_time"

        # ----------------------------------------------
        # Categorical Features
        # ----------------------------------------------

        self.categorical_features = [
            "article_no",
            "operator",
        ]

        self.model = None

    # --------------------------------------------------
    # Prepare Data
    # --------------------------------------------------

    def prepare_data(self, df):

        X = df[self.feature_columns]

        y = df[self.target_column]

        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    def train(self, df):

        X_train, X_test, y_train, y_test = self.prepare_data(df)

        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features,
        )

        test_pool = Pool(
            data=X_test,
            label=y_test,
            cat_features=self.categorical_features,
        )

        self.model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )

        self.model.fit(train_pool)

        predictions = self.model.predict(test_pool)

        metrics = {
            "MAE": mean_absolute_error(
                y_test,
                predictions,
            ),
            "RMSE": mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5,
            "R2": r2_score(
                y_test,
                predictions,
            ),
        }

        return (
            self.model,
            metrics,
            X_test,
            y_test,
            predictions,
        )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    def save_model(
        self,
        model,
        model_path="models/catboost_model.pkl",
    ):

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            model,
            model_path,
        )

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    def feature_importance(self):

        importance = self.model.get_feature_importance()

        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_columns,
                "Importance": importance,
            }
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False,
        )

        return importance_df

    # --------------------------------------------------
    # Print Metrics
    # --------------------------------------------------

    @staticmethod
    def print_metrics(metrics):

        print("\n========== MODEL PERFORMANCE ==========\n")

        print(f"MAE  : {metrics['MAE']:.2f} sec")
        print(f"RMSE : {metrics['RMSE']:.2f} sec")
        print(f"R²   : {metrics['R2']:.3f}")

        print("\n=======================================\n")


# ------------------------------------------------------
# Testing
# ------------------------------------------------------

if __name__ == "__main__":

    from core.data_loader import WeldingDataLoader
    from core.preprocessing import DataPreprocessor

    loader = WeldingDataLoader()

    df = loader.load_data()

    df = DataPreprocessor.preprocess(df)

    trainer = ModelTrainer()

    (
        model,
        metrics,
        X_test,
        y_test,
        predictions,
    ) = trainer.train(df)

    trainer.print_metrics(metrics)

    trainer.save_model(model)

    print("Feature Importance\n")

    print(trainer.feature_importance())

    
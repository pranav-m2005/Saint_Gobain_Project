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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:

    def __init__(self):
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
        self.categorical_features = [
            "article_no",
            "operator",
        ]
        self.model = None

    def prepare_data(self, df):
        X = df[self.feature_columns]
        y = df[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

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
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "R2": r2_score(y_test, predictions),
        }

        return self.model, metrics, X_test, y_test, predictions

    def save_model(self, model, model_path="models/catboost_model.pkl"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    def load_model(self, model_path="models/catboost_model.pkl"):
        return joblib.load(model_path)

    def feature_importance(self):
        importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame(
            {"Feature": self.feature_columns, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)
        return importance_df

    @staticmethod
    def article_performance(X_test, y_test, predictions):
        performance = X_test.copy()
        performance["Actual_Time"] = y_test.values
        performance["Predicted_Time"] = predictions
        performance["Absolute_Error"] = (performance["Actual_Time"] - performance["Predicted_Time"]).abs()

        article_summary = (
            performance.groupby("article_no")
            .agg(
                Samples=("article_no", "count"),
                Average_Actual=("Actual_Time", "mean"),
                Average_Predicted=("Predicted_Time", "mean"),
                MAE=("Absolute_Error", "mean"),
            )
            .round(2)
            .reset_index()
            .sort_values("MAE")
        )
        return article_summary

    @staticmethod
    def prediction_comparison(X_test, y_test, predictions):
        comparison = X_test.copy()
        comparison["Actual_Time"] = y_test.values
        comparison["Predicted_Time"] = predictions
        comparison["Error"] = comparison["Actual_Time"] - comparison["Predicted_Time"]
        comparison["Absolute_Error"] = comparison["Error"].abs()
        return comparison

    @staticmethod
    def print_metrics(metrics):
        print("\n========== MODEL PERFORMANCE ==========\n")
        print(f"MAE  : {metrics['MAE']:.2f} sec")
        print(f"RMSE : {metrics['RMSE']:.2f} sec")
        print(f"R²   : {metrics['R2']:.3f}")
        print("\n=======================================\n")


if __name__ == "__main__":
    from core.data_loader import WeldingDataLoader
    from core.preprocessing import DataPreprocessor

    loader = WeldingDataLoader()
    df = loader.load_data()
    df = DataPreprocessor.preprocess(df)

    trainer = ModelTrainer()
    model, metrics, X_test, y_test, predictions = trainer.train(df)

    trainer.print_metrics(metrics)
    trainer.save_model(model)

    print("Feature Importance\n")
    print(trainer.feature_importance())

    print("\nArticle-wise Model Performance\n")
    print(trainer.article_performance(X_test, y_test, predictions))
    
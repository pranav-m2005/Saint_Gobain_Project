"""
=========================================================
Train Model
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import os

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.trainer import ModelTrainer


def main():

    print("\n==============================================")
    print("      WELDING TIME MODEL TRAINING")
    print("==============================================\n")

    # -------------------------------------------------------
    # Load Data
    # -------------------------------------------------------

    print("Loading data from Google Sheets...")

    loader = WeldingDataLoader()
    df = loader.load_data()

    print(f"✓ Loaded {len(df)} records\n")

    # -------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------

    print("Preprocessing data...")

    df = DataPreprocessor.preprocess(df)

    print("✓ Data preprocessing completed\n")

    # -------------------------------------------------------
    # Train Model
    # -------------------------------------------------------

    print("Training CatBoost model...")

    trainer = ModelTrainer()

    model, metrics, X_test, y_test, predictions = trainer.train(df)

    print("✓ Model training completed\n")

    # -------------------------------------------------------
    # Model Performance
    # -------------------------------------------------------

    trainer.print_metrics(metrics)

    # -------------------------------------------------------
    # Feature Importance
    # -------------------------------------------------------

    print("Feature Importance\n")

    print(trainer.feature_importance())

    # -------------------------------------------------------
    # Article-wise Performance
    # -------------------------------------------------------

    print("\nArticle-wise Model Performance\n")

    print(
        trainer.article_performance(
            X_test,
            y_test,
            predictions,
        )
    )

    # -------------------------------------------------------
    # Save Model
    # -------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    model_path = "models/catboost_model.pkl"

    trainer.save_model(model, model_path)

    print(f"\n✓ Model saved to: {model_path}")

    print("\n==============================================")
    print("      TRAINING COMPLETED SUCCESSFULLY")
    print("==============================================\n")


if __name__ == "__main__":
    main()
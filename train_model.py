"""
=========================================================
Train CatBoost Model
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

from core.data_loader import WeldingDataLoader
from core.preprocessing import DataPreprocessor
from core.validator import DataValidator
from core.trainer import ModelTrainer


def main():

    print("\n==============================================")
    print("      WELDING TIME MODEL TRAINING")
    print("==============================================")

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    loader = WeldingDataLoader()

    df = loader.load_data()

    print("\n✓ Data Loaded Successfully")

    # --------------------------------------------------
    # Preprocess Data
    # --------------------------------------------------

    df = DataPreprocessor.preprocess(df)

    print("✓ Data Preprocessed")

    # --------------------------------------------------
    # Validate Data
    # --------------------------------------------------

    DataValidator.validate(df)

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    trainer = ModelTrainer()

    print("\nTraining CatBoost Model...\n")

    (
        model,
        metrics,
        X_test,
        y_test,
        predictions,
    ) = trainer.train(df)

    print("✓ Model Training Completed")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    trainer.save_model(model)

    print("✓ Model Saved Successfully")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    trainer.print_metrics(metrics)

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    importance = trainer.feature_importance()

    print("\n========== FEATURE IMPORTANCE ==========\n")

    print(importance)

    print("\n==============================================")
    print("Training Completed Successfully")
    print("==============================================\n")


if __name__ == "__main__":
    main()
"""
=========================================================
Data Validator
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import pandas as pd


class DataValidator:

    @staticmethod
    def validate(df):

        print("\n========== DATA VALIDATION ==========\n")

        print(f"Rows : {len(df)}")
        print(f"Columns : {len(df.columns)}")

        print("\nMissing Values")
        print(df.isnull().sum())

        print("\nColumn Types")
        print(df.dtypes)

        duplicate_rows = df.duplicated().sum()

        print(f"\nDuplicate Rows : {duplicate_rows}")

        if duplicate_rows > 0:
            print("\nDuplicate Records:\n")
            print(df[df.duplicated(keep=False)])

        print("\n=====================================\n")

        return True

    # --------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------

    @staticmethod
    def summary(df):

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
        }

        return summary

    # --------------------------------------------------
    # Missing Values
    # --------------------------------------------------

    @staticmethod
    def missing_values(df):

        return df.isnull().sum()

    # --------------------------------------------------
    # Duplicate Rows
    # --------------------------------------------------

    @staticmethod
    def duplicate_rows(df):

        return df[df.duplicated()]

    # --------------------------------------------------
    # Numeric Columns
    # --------------------------------------------------

    @staticmethod
    def numeric_columns(df):

        return df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

    # --------------------------------------------------
    # Categorical Columns
    # --------------------------------------------------

    @staticmethod
    def categorical_columns(df):

        return df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()


# ------------------------------------------------------
# Testing
# ------------------------------------------------------

if __name__ == "__main__":

    from core.data_loader import WeldingDataLoader
    from core.preprocessor import DataPreprocessor

    loader = WeldingDataLoader()

    df = loader.load_data()

    df = DataPreprocessor.preprocess(df)

    DataValidator.validate(df)

    print(DataValidator.summary(df))
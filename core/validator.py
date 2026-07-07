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
"""
=========================================================
Data Validation module
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import pandas as pd


class DataValidator:
    @staticmethod
    def validate(df):
        print("\n========== DATA VALIDATION ==========\n")
        print(f"Total Rows: {df.shape[0]}")
        print(f"Total Columns: {df.shape[1]}")
        print("\nMissing Values Per Column:")
        missing = df.isnull().sum()
        for col, val in missing.items():
            print(f"  {col}: {val}")
        print("\nDuplicate Row Count:", df.duplicated().sum())
        print("\nColumn Data Types:")
        print(df.dtypes)
        print("\nNumeric Summary (describe):")
        print(df.describe(include=[float, int]))
        print("\n=====================================\n")
        return True

    @staticmethod
    def summary(df):
        numeric_cols = DataValidator.numeric_columns(df)
        categorical_cols = DataValidator.categorical_columns(df)
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        }

    @staticmethod
    def missing_values(df):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        result = pd.DataFrame({
            "Column": missing.index,
            "Missing_Values": missing.values
        })
        result = result.sort_values("Missing_Values", ascending=False).reset_index(drop=True)
        return result

    @staticmethod
    def duplicate_rows(df):
        return df[df.duplicated(keep=False)]

    @staticmethod
    def numeric_columns(df):
        return df.select_dtypes(include=["number"]).columns.tolist()

    @staticmethod
    def categorical_columns(df):
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def outlier_summary(df):
        numeric_cols = DataValidator.numeric_columns(df)
        outlier_counts = []
        for col in numeric_cols:
            series = df[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)].count()
            outlier_counts.append({"Column": col, "Outliers": outliers})
        return pd.DataFrame(outlier_counts)


# ------------------------------------------------------
# Testing
# ------------------------------------------------------

if __name__ == "__main__":
    from core.data_loader import WeldingDataLoader
    from core.preprocessing import DataPreprocessor

    loader = WeldingDataLoader()
    df = loader.load_data()
    df = DataPreprocessor.preprocess(df)
    DataValidator.validate(df)
    print(DataValidator.summary(df))
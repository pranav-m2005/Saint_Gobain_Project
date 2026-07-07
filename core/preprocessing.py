"""
=========================================================
Data Preprocessor
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import pandas as pd


class DataPreprocessor:

    # ----------------------------------------------------
    # Main Preprocessing
    # ----------------------------------------------------

    @staticmethod
    def preprocess(df):

        df = df.copy()

        # ------------------------------------------------
        # Rename Columns
        # ------------------------------------------------

        rename_map = {
            "Date": "date",
            "window article no": "article_no",
            "window length": "window_length",
            "window width": "window_width",
            "cross section length": "cross_section_length",
            "cross section width": "cross_section_width",
            "line no": "line_no",
            "loading": "loading",
            "milling time": "milling_time",
            "bur check": "bur_check",
            "heating": "heating",
            "stamping & Cooling": "stamping_cooling",
            "unloading": "unloading",
            "time(sec)": "total_time",
            "operator": "operator"
        }

        df = df.rename(columns=rename_map)

        # ------------------------------------------------
        # Strip whitespace from object columns
        # ------------------------------------------------

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()

        # ------------------------------------------------
        # Date conversion
        # ------------------------------------------------

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        # ------------------------------------------------
        # Numeric conversion
        # ------------------------------------------------

        numeric_cols = [
            "article_no",
            "window_length",
            "window_width",
            "cross_section_length",
            "cross_section_width",
            "line_no",
            "loading",
            "milling_time",
            "heating",
            "stamping_cooling",
            "unloading",
            "total_time"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ------------------------------------------------
        # Fill missing numeric values with median
        # ------------------------------------------------

        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # ------------------------------------------------
        # Fill missing categorical values and convert to category
        # ------------------------------------------------

        for col in ["operator", "bur_check"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype("category")

        # ------------------------------------------------
        # Drop rows where article_no or total_time is missing
        # ------------------------------------------------

        if "article_no" in df.columns and "total_time" in df.columns:
            df = df.dropna(subset=["article_no", "total_time"])

        # ------------------------------------------------
        # Remove duplicates
        # ------------------------------------------------

        df = df.drop_duplicates()

        # ------------------------------------------------
        # Sort by article_no and date if both exist, else by whichever exists
        # ------------------------------------------------

        sort_cols = []
        if "article_no" in df.columns:
            sort_cols.append("article_no")
        if "date" in df.columns:
            sort_cols.append("date")

        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        return df

    # ----------------------------------------------------
    # Available Articles
    # ----------------------------------------------------

    @staticmethod
    def get_article_list(df):
        if "article_no" in df.columns:
            articles = sorted(df["article_no"].dropna().unique())
            return ["All Articles"] + list(articles)
        else:
            return ["All Articles"]

    # ----------------------------------------------------
    # Filter Dataset
    # ----------------------------------------------------

    @staticmethod
    def filter_by_article(df, article):
        if article == "All Articles":
            return df.copy()
        if "article_no" in df.columns:
            return df[df["article_no"] == article].reset_index(drop=True)
        else:
            return df.copy()

    # ----------------------------------------------------
    # Article Summary
    # ----------------------------------------------------

    @staticmethod
    def article_summary(df):
        if "article_no" not in df.columns:
            return pd.DataFrame()
        grouped = df.groupby("article_no").agg(
            Records=("article_no", "count"),
            Average_Time=("total_time", "mean"),
            Minimum_Time=("total_time", "min"),
            Maximum_Time=("total_time", "max"),
            Std_Deviation=("total_time", "std")
        ).round(2).reset_index()
        return grouped

    # ----------------------------------------------------
    # Average Stage Time
    # ----------------------------------------------------

    @staticmethod
    def stage_average(df):
        stages = [
            "loading",
            "milling_time",
            "heating",
            "stamping_cooling",
            "unloading"
        ]
        available_stages = [s for s in stages if s in df.columns]
        if not available_stages:
            return pd.Series(dtype=float)
        avg = df[available_stages].mean().round(2).sort_values(ascending=False)
        return avg

    # ----------------------------------------------------
    # Bottleneck Stage
    # ----------------------------------------------------

    @staticmethod
    def bottleneck_stage(df):
        stage_avg = DataPreprocessor.stage_average(df)
        if stage_avg.empty:
            return (None, None)
        return (stage_avg.idxmax(), stage_avg.max())


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from core.data_loader import WeldingDataLoader

    loader = WeldingDataLoader()

    df = loader.load_data()

    df = DataPreprocessor.preprocess(df)

    print(df.head())

    print("\nAvailable Articles\n")

    print(DataPreprocessor.get_article_list(df))

    print("\nArticle Summary\n")

    print(DataPreprocessor.article_summary(df))
"""
=========================================================
Data Preprocessor
Saint-Gobain Welding Time Predictor - Data Preprocessing
=========================================================
This module provides preprocessing utilities for cleaning and transforming
raw welding process data for downstream analytics and modeling.
"""

import pandas as pd


class DataPreprocessor:
    @staticmethod
    def preprocess(df):
        """
        Clean and preprocess the welding process DataFrame.
        """
        df = df.copy()
        # Rename columns to internal names
        col_map = {
            "Date": "date",
            "window article no": "article_no",
            "window length": "window_length",
            "window width": "window_width",
            "cross section length": "cross_section_length",
            "cross section width": "cross_section_width",
            "line no": "line_no",
            "loading": "loading",
            "milling time": "milling_time",
            "bur check": "bur_check",
            "heating": "heating",
            "stamping & Cooling": "stamping_cooling",
            "unloading": "unloading",
            "time(sec)": "total_time",
            "operator": "operator"
        }
        df.rename(columns=col_map, inplace=True)

        # Strip whitespace from all object columns
        obj_cols = df.select_dtypes(include="object").columns
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip()

        # Convert date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        # Convert specified columns to numeric
        num_cols = [
            "article_no", "window_length", "window_width",
            "cross_section_length", "cross_section_width", "line_no",
            "loading", "milling_time", "heating",
            "stamping_cooling", "unloading", "total_time"
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill numeric NaNs with median
        for col in num_cols:
            if col in df.columns:
                median = df[col].median()
                df[col] = df[col].fillna(median)

        # Fill missing operator and bur_check with "Unknown" and convert to category
        for col in ["operator", "bur_check"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype("category")

        # Drop rows missing article_no or total_time
        drop_cols = [c for c in ["article_no", "total_time"] if c in df.columns]
        if drop_cols:
            df = df.dropna(subset=drop_cols)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Sort by article_no then date if present
        sort_cols = []
        if "article_no" in df.columns:
            sort_cols.append("article_no")
        if "date" in df.columns:
            sort_cols.append("date")
        if sort_cols:
            df = df.sort_values(sort_cols)

        # Reset index before returning
        return df.reset_index(drop=True)

    @staticmethod
    def get_article_list(df):
        """
        Return a list of available article numbers, with 'All Articles' first.
        """
        if "article_no" in df.columns:
            articles = sorted(df["article_no"].dropna().unique())
            return ["All Articles"] + list(articles)
        return ["All Articles"]

    @staticmethod
    def filter_by_article(df, article):
        """
        Return all rows if article == 'All Articles', else only that article's rows.
        """
        if article == "All Articles":
            return df.copy()
        if "article_no" in df.columns:
            return df[df["article_no"] == article].reset_index(drop=True)
        return df.copy()

    @staticmethod
    def article_summary(df):
        """
        Return grouped statistics by article_no: Records, Average_Time, Minimum_Time, Maximum_Time, Std_Deviation.
        """
        if "article_no" not in df.columns or "total_time" not in df.columns:
            return pd.DataFrame()
        grouped = (
            df.groupby("article_no")
              .agg(
                  Records=("article_no", "count"),
                  Average_Time=("total_time", "mean"),
                  Minimum_Time=("total_time", "min"),
                  Maximum_Time=("total_time", "max"),
                  Std_Deviation=("total_time", "std")
              )
              .round(2)
              .reset_index()
        )
        return grouped

    @staticmethod
    def stage_average(df):
        """
        Return average times for each main stage, descending.
        """
        stages = ["loading", "milling_time", "heating", "stamping_cooling", "unloading"]
        available = [col for col in stages if col in df.columns]
        if not available:
            return pd.Series(dtype=float)
        avg = df[available].mean().round(2).sort_values(ascending=False)
        return avg

    @staticmethod
    def bottleneck_stage(df):
        """
        Return (stage_name, average_time) of the stage with highest average time.
        """
        avg = DataPreprocessor.stage_average(df)
        if avg.empty:
            return (None, None)
        return (avg.idxmax(), avg.max())


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    from core.data_loader import WeldingDataLoader
    loader = WeldingDataLoader()
    df = loader.load_data()
    df = DataPreprocessor.preprocess(df)
    print(df.head())
    print("\nAvailable Articles\n")
    print(DataPreprocessor.get_article_list(df))
    print("\nArticle Summary\n")
    print(DataPreprocessor.article_summary(df))
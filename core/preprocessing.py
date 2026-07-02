import pandas as pd


class DataPreprocessor:

    @staticmethod
    def preprocess(df):

        # Rename Google Sheet columns to clean Python names
        df = df.rename(columns={
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
        })

        # Numeric columns
        numeric_columns = [
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

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Categorical
        df["operator"] = df["operator"].astype("category")

        return df
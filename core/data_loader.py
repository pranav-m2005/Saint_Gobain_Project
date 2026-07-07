"""
=========================================================
Data Loader
Intelligent Welding Time Estimation & Process Analytics
Saint-Gobain Project
=========================================================
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class WeldingDataLoader:
    """
    Loads welding production data from Google Sheets.
    Supports both:
    1. Local JSON file
    2. Streamlit Cloud Secrets
    """

    def __init__(
        self,
        credentials_file="credentials/welding-time-predictor-a9b4dda965dc.json",
        sheet_name="welding analysis",
    ):
        self.credentials_file = Path(credentials_file)
        self.sheet_name = sheet_name
        self.scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        self.client = self._authenticate()
        self.sheet = self.client.open(self.sheet_name).sheet1

    def _authenticate(self):
        if STREAMLIT_AVAILABLE:
            try:
                service_account = st.secrets["gcp_service_account"]
                credentials = Credentials.from_service_account_info(
                    dict(service_account),
                    scopes=self.scopes,
                )
                print("✓ Using Streamlit Secrets")
                return gspread.authorize(credentials)
            except Exception:
                pass

        if not self.credentials_file.is_file():
            raise FileNotFoundError(
                f"Local credentials file not found: {self.credentials_file.resolve()}"
            )

        credentials = Credentials.from_service_account_file(
            str(self.credentials_file),
            scopes=self.scopes,
        )
        print("✓ Using Local Service Account")
        return gspread.authorize(credentials)

    def load_data(self):
        try:
            records = self.sheet.get_all_records()
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.replace("", pd.NA, inplace=True)

            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.strip()

            if "window article no" in df.columns:
                df = df[df["window article no"].notna()]
                df = df[df["window article no"] != ""]

            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load Google Sheet: {e}")

    def refresh_data(self):
        return self.load_data()

    def get_shape(self):
        return self.load_data().shape

    def get_columns(self):
        return list(self.load_data().columns)


if __name__ == "__main__":
    loader = WeldingDataLoader()
    df = loader.load_data()
    print(df.head())
    print(df.shape)

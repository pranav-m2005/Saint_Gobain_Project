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
        credentials_file="welding-time-predictor-a9b4dda965dc.json",
        sheet_name="welding analysis",
    ):

        self.credentials_file = credentials_file
        self.sheet_name = sheet_name

        self.scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        self.client = self._authenticate()

        self.sheet = self.client.open(self.sheet_name).sheet1

    # ----------------------------------------------------
    # Authenticate
    # ----------------------------------------------------

    def _authenticate(self):

        try:
            import streamlit as st

            # Force access to the secret so we see the real error if it fails
            service_account = st.secrets["gcp_service_account"]

            credentials = Credentials.from_service_account_info(
                dict(service_account),
                scopes=self.scopes,
            )

            print("✓ Using Streamlit Secrets")

            return gspread.authorize(credentials)

        except Exception as e:

            # Show the actual error in Streamlit logs
            raise RuntimeError(
                f"Unable to load Streamlit secret 'gcp_service_account': {e}"
            )

        # This fallback is only used locally
        credentials = Credentials.from_service_account_file(
            self.credentials_file,
            scopes=self.scopes,
        )

        return gspread.authorize(credentials)

    # ----------------------------------------------------
    # Load Data
    # ----------------------------------------------------

    def load_data(self):

        try:

            records = self.sheet.get_all_records()

            df = pd.DataFrame(records)

            # Replace blanks
            df.replace("", pd.NA, inplace=True)

            # Remove empty article rows
            if "window article no" in df.columns:

                df = df[df["window article no"].notna()]

                df = df[df["window article no"] != ""]

            # Strip strings
            for column in df.columns:

                if df[column].dtype == "object":

                    df[column] = (
                        df[column]
                        .astype(str)
                        .str.strip()
                    )

            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:

            print("\nUnable to load Google Sheet\n")

            print(e)

            return pd.DataFrame()

    # ----------------------------------------------------
    # Refresh
    # ----------------------------------------------------

    def refresh_data(self):

        return self.load_data()

    # ----------------------------------------------------
    # Shape
    # ----------------------------------------------------

    def get_shape(self):

        return self.load_data().shape

    # ----------------------------------------------------
    # Columns
    # ----------------------------------------------------

    def get_columns(self):

        return list(self.load_data().columns)


# --------------------------------------------------------
# Test
# --------------------------------------------------------

if __name__ == "__main__":

    loader = WeldingDataLoader()

    df = loader.load_data()

    print(df.head())

    print(df.shape)


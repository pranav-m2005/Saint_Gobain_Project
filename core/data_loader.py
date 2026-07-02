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


class WeldingDataLoader:
    """
    Loads welding production data from Google Sheets.
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

        credentials = Credentials.from_service_account_file(
            self.credentials_file,
            scopes=self.scopes,
        )

        client = gspread.authorize(credentials)

        return client

    # ----------------------------------------------------
    # Load Data
    # ----------------------------------------------------

    def load_data(self):

        try:

            records = self.sheet.get_all_records()

            df = pd.DataFrame(records)

            # Replace blank strings with NaN
            df.replace("", pd.NA, inplace=True)

            # Remove rows without article number
            if "window article no" in df.columns:
                df = df[df["window article no"].notna()]
                df = df[df["window article no"] != ""]

            # Remove leading/trailing spaces
            for column in df.columns:
                if df[column].dtype == "object":
                    df[column] = (
                        df[column]
                        .astype(str)
                        .str.strip()
                    )

            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as error:

            print("\nERROR : Unable to load Google Sheet.\n")

            print(error)

            return pd.DataFrame()

    # ----------------------------------------------------
    # Reload Data
    # ----------------------------------------------------

    def refresh_data(self):
        """
        Reload the latest data from Google Sheets.
        """

        return self.load_data()

    # ----------------------------------------------------
    # Dataset Information
    # ----------------------------------------------------

    def get_shape(self):

        df = self.load_data()

        return df.shape

    def get_columns(self):

        df = self.load_data()

        return list(df.columns)


# --------------------------------------------------------
# Test
# --------------------------------------------------------

if __name__ == "__main__":

    loader = WeldingDataLoader()

    df = loader.load_data()

    print("\nFirst Five Rows\n")

    print(df.head())

    print("\nShape :", df.shape)
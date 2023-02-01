"""
Reads the CSV file and cleans the dataframe.
"""

import numpy as np
import pandas as pd


def get_cleaned_data():
    """
    Reads the CSV file and returns the cleaned dataframe.
    """
    df = pd.read_csv("TARCC_data.csv")

    # Replace all empty strings with NaN, and convert all relevant columns to numeric (float).
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop all text columns.
    df = df.drop(columns=df.select_dtypes("object"))

    # TODO: Continue encoding unknown values. Should "Other" be converted to NA?
    # df["A1_HISPANIC"].replace(9, np.nan, inplace=True)
    # df["A1_HISPANIC_TYPE"].replace([99], np.nan, inplace=True)
    # df["A1_RACE"].replace([99], np.nan, inplace=True)
    # df["A1_RACESEC"].replace([88, 99], np.nan, inplace=True)

    return df

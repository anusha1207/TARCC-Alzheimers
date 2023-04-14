"""
Reads the CSV file and cleans the dataframe.
"""
import json

import numpy as np
import pandas as pd


def drop_features(df: pd.DataFrame, features_to_drop: list[str]) -> None:
    """
    Drops the given features from the input dataframe. This function modifies the data frame in place instead of
    producing a new one.

    Args:
        df: The original (or partially cleaned) TARCC dataset.
        features_to_drop: The features to drop from the input dataframe.

    Returns:
        None
    """

    df.drop(features_to_drop, axis=1, inplace=True)


def map_value_D1(value):
    """
    Helper function to map indicators to a summable weight
    """
    value_mappings = {0: 0, 1: 3, 2: 2, 3: 1}
    if type(value) == int:
        return value_mappings[value]
    else:
        return value


def sum_D1(df: pd.DataFrame):
    """
    Sums the diagnostic classifications for patients
    NOTE: to be called after clean_D1
    Inputs:
        pandas dataframe of clinical/biomarker data
    Returns:
        None, alters the inputted dataframe
    """
    cols = [col for col in df if col.startswith('D1_')]
    # only operating on set of selected columns
    window = df[cols]
    # mapping classifiers to reflect weights
    window = window.applymap(map_value_D1)
    # summing each patient row to create total risk factor
    df["D1_total"] = window.sum(axis=1)


def get_cleaned_data() -> pd.DataFrame:
    """
    Reads the CSV file and returns the cleaned dataframe.

    Returns:
        The cleaned dataframe representing the TARCC data.
    """

    df = pd.read_csv("data/TARCC_data.csv")

    # Remove all patients labeled as "Other".
    df = df[df["P1_PT_TYPE"] != 3]

    # Replace all empty strings with NaN, and convert all relevant columns to numeric (float).
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Converts all numeric columns to numeric data types.
    df = df.apply(pd.to_numeric, errors="ignore")

    # Create A3_PROP_PARENTS_DEM, which is the proportion of parents with dementia.
    df["A3_PROP_PARENTS_DEM"] = (df["A3_MOMDEM"] + df["A3_DADDEM"]) / 2

    # Define the conversion rates between mg and each of µg (1), mg (2), mL (3), and IU (4).
    vitamin_e_conversions = np.array([
        0,      # Null conversion (no vitamin E).
        0.001,  # 1 µg = 0.001 mg.
        1,      # 1 mg = 1 mg.
        -1,     # Make all mL entries negative to find them later.
        0.45,   # 1 IU ≈ 0.45 mg.
    ])

    # Convert all µg and IU strengths to mg, and negate the mL strengths.
    df["A42_VEAS"] *= vitamin_e_conversions[df["A42_VEASU"]]

    # Replace all the negative entries with the average of all the positive entries.
    df.loc[df["A42_VEAS"] < 0, "A42_VEAS"] = df.loc[df["A42_VEAS"] > 0, "A42_VEAS"].mean()

    # Replace all the missing entries with 0.
    df.loc[np.isnan(df["A42_VEAS"]), "A42_VEAS"] = 0

    # Create A5_NUM_STROKES, which is the number of strokes a patient has had.
    df["A5_NUM_STROKES"] = np.sum(
        ~pd.isna(df[["A5_STROK1YR", "A5_STROK2YR", "A5_STROK3YR", "A5_STROK4YR", "A5_STROK5YR", "A5_STROK6YR"]]),
        axis=1
    )

    # Create A5_NUM_TIA, which is the number of TIAs a patient has had.
    df["A5_NUM_TIA"] = np.sum(
        ~pd.isna(df[["A5_TIA1YR", "A5_TIA2YR", "A5_TIA3YR", "A5_TIA4YR", "A5_TIA5YR", "A5_TIA6YR"]]),
        axis=1
    )

    sum_D1(df)

    # Drop any unnecessary features without a prefix.
    df.drop(["PID", "GWAS"], axis=1, inplace=True)

    with open("config/data_codes.json") as data_codes:

        json_object = json.load(data_codes)

        # Drop features by prefix.
        filtering = json_object["filtering"]
        for prefix, filters in filtering.items():
            if filters["keep"]:
                drop_features(df, list(filter(
                    lambda feature_name: feature_name.startswith(prefix) and feature_name not in filters["features"],
                    df.columns
                )))
            else:
                drop_features(df, filters["features"])

        # Replace missing value codes by NaN.
        missing_value_codes = json_object["missing_value_codes"]
        for key, values in missing_value_codes.items():
            df[key].replace(values, np.nan, inplace=True)

    # Special case for PROTEO: Replace missing values with NaN.
    for key in df.columns[[feature_name.startswith("PROTEO") for feature_name in df.columns]]:
        df[key].replace([-777777, 0], np.nan, inplace=True)

    # Special case for PROTEO: Replace LLDL and GHDL with 0 and 999999999, respectively.
    df.replace(-999999, 0, inplace=True)
    df.replace(-888888, 999999999, inplace=True)

    return df

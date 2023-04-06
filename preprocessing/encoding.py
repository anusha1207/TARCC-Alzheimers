"""
Provides functions for encoding the dataframe.
"""
import json

import pandas as pd


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the input dataframe using dummy encoding. The dataframe must be cleaned before
    calling this function. This function encodes all variables in "encodings" array in config/data_codes.json file.
    Users can edit this configuration file to add or remove features to be encoded.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A cleaned and encoded TARCC dataframe.
    """

    with open("../config/data_codes.json") as data_codes:
        encodings = json.load(data_codes)["encodings"]
        return pd.get_dummies(df, dummy_na=True, columns=encodings, drop_first=True)

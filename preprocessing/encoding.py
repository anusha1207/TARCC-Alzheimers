"""
Provides functions for encoding the dataframe.
"""
import pandas as pd


def encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the demographics (A1) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the A1 categorical variables encoded.
    """
    return pd.get_dummies(df, dummy_na=True, columns=['A1_HISPANIC', 'A1_MARISTAT', 'A1_RACE', 'A1_SEX'])


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the input dataframe using dummy encoding. The dataframe must be cleaned before
    calling this function.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A cleaned and encoded TARCC dataframe.
    """

    df = encode_demographics(df)

    # Call all encode_* functions here.

    return df

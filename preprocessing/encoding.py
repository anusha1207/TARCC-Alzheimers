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


def encode_medical_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the medical history (A5) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the A1 categorical variables encoded.
    """

    return pd.get_dummies(df, dummy_na=True, columns=["A5_CVHATT","A5_CVAFIB","A5_CVANGIO","A5_CVBYPASS","A5_CVPACE",
                                                      "A5_CVCHF", "A5_CBTIA","A5_SEIZURES","A5_TRAUMBRF",
                                                      "A5_TRAUMEXT","A5_TRAUMCHR","A5_PD","A5_HYPERTEN","A5_HYPERCHO",
                                                      "A5_DIABETES","A5_B12DEF","A5_THYROID","A5_INCONTU","A5_INCONTF",
                                                      "A5_CANCER","A5_DEP2YRS","A5_ALCOHOL","A5_TOBAC30","A5_TOBAC100",
                                                      "A5_PACKSPER","A5_PSYCDIS","A5_IBD"])

def encode_D1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the input dataframe using dummy encoding. The dataframe must be cleaned before
    calling this function.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A cleaned and encoded TARCC dataframe.
    """
    cols = []
    for col in df.columns:
        if col.startswith('D1_') and col.endswith('IF'):
            cols.append(col)
    return pd.get_dummies(df, dummy_na=True, columns=cols)

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
    df = encode_medical_history(df)
    # Call all encode_* functions here.

    return df
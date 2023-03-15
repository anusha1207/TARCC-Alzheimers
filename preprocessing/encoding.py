"""
Provides functions for encoding the dataframe.
"""

import pandas as pd


def encode_compliance_committee_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the compliance committee review (CCR) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the CCR categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=list(filter(lambda feature_name: feature_name.startswith("CCR"), df.columns)),
        drop_first=True
    )


def encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the demographics (A1) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the A1 categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=["A1_HISPANIC", "A1_MARISTAT", "A1_RACE", "A1_SEX"],
        drop_first=True
    )


def encode_family_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the family history (A3) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the A3 categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=["A3_PROP_PARENTS_DEM"],
        drop_first=True
    )


def encode_medical_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the medical history (A5) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the A1 categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=[
            "A5_CVHATT",
            "A5_CVAFIB",
            "A5_CVANGIO",
            "A5_CVBYPASS",
            "A5_CVPACE",
            "A5_CVCHF",
            "A5_CBTIA",
            "A5_SEIZURES",
            "A5_TRAUMBRF",
            "A5_TRAUMEXT",
            "A5_TRAUMCHR",
            "A5_PD",
            "A5_HYPERTEN",
            "A5_HYPERCHO",
            "A5_DIABETES",
            "A5_B12DEF",
            "A5_THYROID",
            "A5_INCONTU",
            "A5_INCONTF",
            "A5_CANCER",
            "A5_DEP2YRS",
            "A5_ALCOHOL",
            "A5_TOBAC30",
            "A5_TOBAC100",
            "A5_PACKSPER",
            "A5_PSYCDIS",
            "A5_IBD",
            "A5_ARTHRITIC",
            "A5_AUTOIMM",
            "A5_CBSTROKE",
            "A5_CHRON_OTH",
            "A5_TOBACLSTYR",
        ],
        drop_first=True
    )


def encode_apolipoprotein_e(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the Apolipoprotein E (APOE) section of the input dataframe using dummy
    encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the APOE categorical variables encoded.
    """

    return pd.get_dummies(df, dummy_na=True, columns=["APOE_GENOTYPE"], drop_first=True)


def encode_body_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the body measurements (B1) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the B1 categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=[
            "B1_VISION",
            "B1_VISCORR",
            "B1_VISWCORR",
            "B1_HEARING",
            "B1_HEARAID",
            "B1_HEARWAID",
        ],
        drop_first=True
    )


def encode_npi_questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the NPI Questionnaire (B5) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the B5 categorical variables encoded.
    """

    return pd.get_dummies(
        df,
        dummy_na=True,
        columns=list(filter(lambda feature: feature.startswith("B5") and feature != "B5_NPIQINF", df.columns)),
        drop_first=True
    )


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
    return pd.get_dummies(df, dummy_na=True, columns=cols, drop_first=True)


def encode_rule_based_medicine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the Rule Based Medicine (RBM) section of the input dataframe using dummy
    encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the FBM categorical variables encoded.
    """

    return pd.get_dummies(df, dummy_na=True, columns=["RBM_Rule_Based_Medicine", "RBM_Batch"], drop_first=True)


def encode_quanterix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the Quanterix (Q1) section of the input dataframe using dummy encoding.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A dataframe with the Q1 categorical variables encoded.
    """

    return pd.get_dummies(df, dummy_na=True, columns=["Q1_Quanterix"], drop_first=True)


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical variables of the input dataframe using dummy encoding. The dataframe must be cleaned before
    calling this function.

    Args:
        df: The cleaned dataframe representing the TARCC dataset.

    Returns:
        A cleaned and encoded TARCC dataframe.
    """

    df = encode_compliance_committee_review(df)
    df = encode_demographics(df)
    df = encode_family_history(df)
    df = encode_medical_history(df)
    df = encode_apolipoprotein_e(df)
    df = encode_body_measurements(df)
    df = encode_npi_questionnaire(df)
    df = encode_D1(df)
    df = encode_rule_based_medicine(df)
    df = encode_quanterix(df)

    return df

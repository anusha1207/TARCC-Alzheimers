"""
A collection of utility functions.
"""
import pandas as pd


def log_features(df: pd.DataFrame, log="features") -> None:
    """
    Outputs a list of existing features and their datatypes to a log file.

    Args:
        df: The dataframe representing the TARCC dataset at any stage of the data science pipeline.
        log: The name of the log file.

    Returns:
        None
    """
    with open(f"{log}.log", "w") as f:
        for i in range(len(df.columns)):
            f.write(f"{df.columns[i]}\t{df.dtypes[i]}\n")


def split_csv(original_df: pd.DataFrame):
    """
    Takes in the original dataset and creates three new data frames: one for combined blood and clinical data, one for
    blood data only, and one for clinical data only.

    Args:
        original_df: The cleaned and encoded TARCC dataset.

    Returns: Three new data frames formatted as follows:
        combined - contains blood and clinical features, but only patients who have drawn blood.
        blood_only - contains only blood features and only patients who have drawn blood.
        clinical_only - contains only clinical features and all patients.
    """

    blood_feats = ["APOE", "PROTEO", "RBM", "Q1", "P1", "PATID"]

    filtered_feats = list(filter(lambda name: any(name.startswith(prefix) for prefix in blood_feats), original_df.columns))

    filtered_feats.remove("PATID")
    filtered_feats.remove("P1_PT_TYPE")

    combined = original_df.dropna(subset=["RBM_TARC_PID"])
    combined = combined[combined["P1_PT_TYPE"] != 4]

    blood_only = original_df.dropna(subset=["RBM_TARC_PID"])[["PATID", "P1_PT_TYPE"] + filtered_feats]
    blood_only = blood_only[blood_only["P1_PT_TYPE"] != 4]

    clinical_only = original_df.drop(filtered_feats, axis=1)

    blood_only.to_csv("Blood Data.csv", index=False)
    clinical_only.to_csv("Clinical Data.csv", index=False)

    return combined, blood_only, clinical_only


def get_features_label(cleaned_df):
    """
    Takes in the cleaned df and outputs the features df and label df

    Args:
        cleaned_df: cleaned TARCC dataframe

    Returns:
            Two data frames:
                label - labels of CN, MCI and AD for all the data
                features - feature set
    """
    label_df = cleaned_df["P1_PT_TYPE"]
    features_df = cleaned_df.drop("P1_PT_TYPE", axis=1)
    return label_df, features_df


def remove_bookkeeping_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all bookkeeping features from the input dataset. This function should be used before modeling or feature
    selection so that these features are not included in the models.

    Args:
        df: The cleaned and encoded TARCC dataframe.

    Returns:
        The input dataframe with PATID, STUDYID, VISIT, and RBM_TARC_PID removed.
    """

    if "PATID" in df.columns:
        df = df.drop("PATID", axis=1)
    if "STUDYID" in df.columns:
        df = df.drop("STUDYID", axis=1)
    if "VISIT" in df.columns:
        df = df.drop("VISIT", axis=1)
    if "RBM_TARC_PID" in df.columns:
        df = df.drop("RBM_TARC_PID", axis=1)

    return df

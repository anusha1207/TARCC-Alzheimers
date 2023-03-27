"""
Reads the CSV file and cleans the dataframe.
"""
import numpy as np
import pandas as pd


def clean_miscellaneous(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the miscellaneous (MISC) section.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(["MISC_TARC_PAT_VISIT", "MISC_SITEID"], axis=1, inplace=True)


def clean_compliance_committee_review(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the compliance committee review (CCR) section.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(
        list(filter(lambda name: name.startswith("CCR"), df.columns)),
        axis=1,
        inplace=True
    )


def clean_demographics(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the demographics (A1) section.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(
        [
            "A1_BIRTHYR",
            "A1_EVENTDATEX",
            "A1_HANDEDNESS",
            "A1_HISPANIC_TYPE",
            "A1_HISPORX",
            "A1_RACESEC",
            "A1_RESIDENC",
        ],
        axis=1,
        inplace=True
    )

    # Replace missing values with NaN.
    missing_value_codes = {
        "A1_HISPANIC": [9],
        "A1_MARISTAT": [8, 9],
        "A1_RACE": [50, 99],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)


def clean_family_history(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the family history (A3) section. We merge the dad or mom having
    dementia into one feature which is the proportion of parents with dementia.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    # Replace missing values with NaN.
    missing_value_codes = {
        "A3_MOMDEM": [9],
        "A3_DADDEM": [9],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)

    # Creating 'PROP_PARENTS_DEM' which is the proportion of parents with dementia
    df["A3_PROP_PARENTS_DEM"] = (df["A3_MOMDEM"] + df["A3_DADDEM"]) / 2

    df.drop(
        [
            "A3_MOMDEM",
            "A3_DADDEM"
        ],
        axis=1,
        inplace=True
    )


def clean_medicinal_history(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the medicinal history (A41-A44) section.

    We remove all features relating to prescriptions, non-prescriptions, anti-dementia drugs, systemic steroids, and
    drug trials, and we keep only the features relating to vitamin E.

    In the dataset, each drug is summarized by various fields such as name, route, strength, frequency, etc. Here, we
    consider only the "strength" and "strength units" field for each drug, which we summarize into one feature.
    Therefore, this function takes all A41-A44 features and replaces them with a single feature: the strength of vitamin
    E in milligrams.

    Notes:
        The TARCC codebook specifies four entries for vitamin E features (A-D), but only features for A are in the data.

        For vitamin E, 779 entries use IU, 105 entries use mg, 14 entries use µg, and 3 entries use mL, but we choose to
        keep our final feature in mg. To convert from IU to mg, we assume that all vitamin E used by patients is
        synthetic, implying a conversion rate of 0.45 mg / IU. And since very few entries use mL, we choose to replace
        these values by an average of all other non-zero vitamin E strengths.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

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

    # Drop all "A4" columns except for the vitamin E strengths.
    df.drop(
        list(filter(lambda name: name.startswith("A4") and name != "A42_VEAS", df.columns)),
        axis=1,
        inplace=True
    )


def clean_medical_history(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the medical history (A5) section. We drop all of the "Other"
    features (i.e., cardiovascular other and cerebrovascular other), and merge years of strokes/TIAs into
    "number of TIAs and number of strokes per patient".

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    # Creating 'NUM_STROKES' which is the number of strokes a patient has had
    df["A5_NUM_STROKES"] = np.sum(
        ~pd.isna(df[["A5_STROK1YR", "A5_STROK2YR", "A5_STROK3YR", "A5_STROK4YR", "A5_STROK5YR", "A5_STROK6YR"]]),
        axis=1
    )

    # Creating 'NUM_TIA' which is the number of TIAs a patient has had
    df["A5_NUM_TIA"] = np.sum(
        ~pd.isna(df[["A5_TIA1YR", "A5_TIA2YR", "A5_TIA3YR", "A5_TIA4YR", "A5_TIA5YR", "A5_TIA6YR"]]),
        axis=1
    )

    df.drop(
        [
            "A5_CVOTHR",
            "A5_CVOTHRX",
            "A5_CBOTHR",
            "A5_CHRON_OTHX",
            "A5_CBOTHRX",
            "A5_PDYR",
            "A5_PDOTHRYR",
            "A5_NCOTHR",
            "A5_NCOTHRX",
            "A5_ABUSOTHR",
            "A5_ABUSX",
            "A5_PSYCDISX",
            "A5_PDOTHR",
            "A5_DEPOTHR",
            "A5_STROK1YR",
            "A5_STROK2YR",
            "A5_STROK3YR",
            "A5_STROK4YR",
            "A5_STROK5YR",
            "A5_STROK6YR",
            "A5_TIA1YR",
            "A5_TIA2YR",
            "A5_TIA3YR",
            "A5_TIA4YR",
            "A5_TIA5YR",
            "A5_TIA6YR",
        ],
        axis=1,
        inplace=True
    )

    # Replace missing values with NaN.
    missing_value_codes = {
        "A5_CVHATT": [9],
        "A5_CVAFIB": [9],
        "A5_CVANGIO": [9],
        "A5_CVBYPASS": [9],
        "A5_CVPACE": [9],
        "A5_CVCHF": [9],
        "A5_CBSTROKE": [9],
        "A5_CBTIA": [9],
        "A5_SEIZURES": [9],
        "A5_TRAUMBRF": [9],
        "A5_TRAUMEXT": [9],
        "A5_TRAUMCHR": [9],
        "A5_PD": [9],
        "A5_HYPERTEN": [9],
        "A5_HYPERCHO": [9],
        "A5_DIABETES": [9],
        "A5_B12DEF": [9],
        "A5_THYROID": [9],
        "A5_INCONTU": [9],
        "A5_INCONTF": [9],
        "A5_CANCER": [9],
        "A5_DEP2YRS": [9],
        "A5_ALCOHOL": [9],
        "A5_TOBAC30": [9],
        "A5_TOBAC100": [9],
        "A5_PACKSPER": [9],
        "A5_PSYCDIS": [9],
        "A5_IBD": [9],
        "A5_ARTHRITIC": [9],
        "A5_AUTOIMM": [9],
        "A5_CHRON_OTH": [9],
        "A5_QUITSMOK": [99],
        "A5_TOBACLSTYR": [9],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)


def clean_apolipoprotein_e(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the Apolipoprotein E (APOE) section.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(["APOE_GENOTYPE_DIGITS", "APOE_E2_COUNT", "APOE_E3_COUNT", "APOE_E4_COUNT"], axis=1, inplace=True)


def clean_body_measurements(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the medicinal history (A41-A44) section.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    # Replace missing values with NaN.
    missing_value_codes = {
        "B1_VISION": [9],
        "B1_VISCORR": [9],
        "B1_VISWCORR": [9],
        "B1_HEARING": [9],
        "B1_HEARAID": [9],
        "B1_HEARWAID": [9],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)


def clean_npi_questionnaire(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean the NPI Questionnaire (B5) section. Most of the features in this section come
    in pairs: The first feature measures whether a patient has a given condition (using a 0/1 variable), and the second
    measures the severity of that feature (if the first feature was a 1) on a scale from 1 to 3 (inclusive). If the
    first feature is a 0, this function appends the score of 0 to the severity feature; otherwise, it just uses the
    severity score. This is done by simply removing all binary features, since the default value of each severity
    feature is already 0.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(["B5_NPIQINFX"], axis=1, inplace=True)

    # Drop the binary features.
    all_b5_features = list(filter(lambda feature: feature.startswith("B5"), df.columns))
    all_b5_features.remove("B5_NPIQINF")
    b5_binary_features = list(filter(lambda feature: not feature.endswith("SEV"), all_b5_features))
    df.drop(b5_binary_features, axis=1, inplace=True)


def clean_cognitive_tests(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the cognitive tests (C1) section. Since there are multiple tests and
    multiple scores for each test, we keep only the total scores of any sections or tests that test working memory.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    keep = [
        "C1_WAIS3_DIGTOT", "C1_WAISR_DIGTOT",
        "C1_MMSE", "C1_CDRSUM", "C1_CDRGLOB",
        "C1_SS_TRAILA", "C1_SS_TRAILB",
        "C1_CLOCK",
        "C1_WMS3_LMEM1", "C1_WMS3_LMEM2", "C1_WMS3_VRI", "C1_WMS3_VR2",
        "C1_WMSR_LMEM1", "C1_WMSR_LMEM2", "C1_WMSR_VRI", "C1_WMSR_VRII", "C1_WMSR_DIGTOT",
        "C1_GDS30",
    ]

    # Drop all irrelevant "C1" columns.
    df.drop(
        list(filter(lambda name: name.startswith("C1") and name not in keep, df.columns)),
        axis=1,
        inplace=True
    )

    # Replace missing values with NaN.
    missing_value_codes = {
        "C1_WAIS3_DIGTOT": [99],
        "C1_WAISR_DIGTOT": [99],
        "C1_MMSE": [99],
        "C1_CLOCK": [999],
        "C1_WMS3_LMEM1": [99],
        "C1_WMS3_LMEM2": [99],
        "C1_WMSR_LMEM1": [99],
        "C1_WMSR_LMEM2": [99],
        "C1_WMSR_VRI": [99],
        "C1_WMSR_VRII": [99],
        "C1_WMSR_DIGTOT": [99],
        "C1_GDS30": [99],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)


def clean_diagnoses(df: pd.DataFrame) -> None:
    """
    Eliminates the columns specifying diagnostic indicators for patients.

    Args:
        pandas dataframe of clinical/biomarker data

    Returns:
        None, alters the inputted dataframe
    """
    # list all columns starting with D1_ and end with IF (want to keep)
    filter_col = [col for col in df if col.startswith('D1_') and col.endswith('IF')]
    # list all columns starting with D1_ (don't want to keep all)
    filter_col_delete = [col for col in df if col.startswith('D1_')]
    # find difference between lists to show columns starting with D1_ and not ending in IF
    final_delete = [col for col in filter_col_delete if col not in filter_col]
    # dropping above columns
    df.drop(final_delete, axis=1, inplace=True)


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
    df['D1_total'] = window.sum(axis=1)


def clean_exit(df: pd.DataFrame) -> None:
    """
    Eliminates the columns specifying method of exit for patients.
    Also removes probable/possible AD features which are highly correlated with response.

    Args:
        pandas dataframe of clinical/biomarker data

    Returns:
        None, alters the inputted dataframe
    """
    unwanted_cols = []

    for col in df.columns:
        if col.startswith("E1"):
            unwanted_cols.append(col)
        if col.startswith("D1"):
            if col in ["D1_PROBAD", "D1_PROBADIF", "D1_POSSAD", "D1_POSSADIF", "D1_WHODIDDX"]:
                unwanted_cols.append(col)
    df.drop(unwanted_cols, axis=1, inplace=True)


def clean_extra_patient_info(df) -> None:
    """
    This function will modify the section of the dataset that contains information about the patients visit,
    their physical activities, and their designated informant.

    For a majority of these features, we have decided to remove them since they contain textual information
    or were deemed to not be relevant to the objective of this project. There are very few features that will be
    kept for these sections. For example, the section that describes the patient's physical activity does so by
    asking them to score different activities based on how often they do them. We decided to only keep the feature
    that contains the total score for these sections.

    This function modifies the data frame in place instead of producing a new one.

    Args:
        df: The original dataset

    Returns:
        None.
    """

    unwanted_feats = ["F1", "F2", "I1", "P1", "X1", "X2"]
    feats_to_keep = ["F1_PSMSTOTSCR", "F2_IADLTOTSCR", "P1_PT_TYPE"]

    for prefix in unwanted_feats:
        df.drop(
            list(filter(lambda name: name.startswith(prefix) and name not in feats_to_keep, df.columns)),
            axis=1,
            inplace=True
        )

    # Replace missing values with NaN.
    missing_value_codes = {
        "Q1_YKL_40": [-999, -888, -555, -333, -444],
        "Q1_GFAP": [-999, -888, -555, -333, -444],
        "Q1_NFL": [-999, -888, -555, -333, -444],
        "Q1_Total_tau": [-999, -888, -555, -333, -444],
        "Q1_UCHL1": [-999, -888, -555, -333, -444],
        "Q1_CD14": [-999, -888, -555, -333, -444],
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)


def clean_proteomics(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean the proteomics section. We replace missing values (-777777 and 0) with NaN and
    replace LLDL (-999999, "lower than low detection level") and GHDL (-888888, "greater than high detection level")
    with 0 and 999999999, respectively.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    # Replace missing values with NaN.
    for key in df.columns[[feature_name.startswith("PROTEO") for feature_name in df.columns]]:
        df[key].replace([-777777, 0], np.nan, inplace=True)

    # Replace LLDL and GHDL with 0 and 999999999, respectively.
    df.replace(-999999, 0, inplace=True)
    df.replace(-888888, 999999999, inplace=True)


def clean_other(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up features not covered by any of the other functions in this module.

    Args:
        df: The dataframe representing the TARCC dataset.

    Returns:
        None
    """

    df.drop(["PID", "GWAS"], axis=1, inplace=True)


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

    # Clean each section of the dataset.
    clean_miscellaneous(df)
    clean_compliance_committee_review(df)
    clean_demographics(df)
    clean_family_history(df)
    clean_medicinal_history(df)
    clean_medical_history(df)
    clean_apolipoprotein_e(df)
    clean_body_measurements(df)
    clean_npi_questionnaire(df)
    clean_cognitive_tests(df)
    clean_exit(df)
    clean_diagnoses(df)
    sum_D1(df)
    clean_extra_patient_info(df)
    clean_other(df)
    clean_proteomics(df)

    return df


def split_csv(original_df):
    """
    Takes in the original dataset and creates two new Data Frames, one with only clinical data
    and the other with only blood data.

    Args:
        original_df: The original dataset

    Returns:
        Two new data frames that contain only one type of data (either Blood or Clinical).
    """

    blood_feats = ["APOE", "PROTEO", "RBM", "Q1", "P1", "PATID"]

    filtered_feats = list(filter(lambda name: any(name.startswith(prefix) for prefix in blood_feats), original_df.columns))

    filtered_feats.remove("PATID")
    filtered_feats.remove("P1_PT_TYPE")

    new_df2 = original_df.drop(filtered_feats, axis=1)
    new_df1 = original_df.dropna(subset = ["RBM_TARC_PID"])

    new_df1.to_csv("Blood Data.csv", index=False)
    new_df2.to_csv("Clinical Data.csv", index=False)

    return new_df1, new_df2


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

"""
Reads the CSV file and cleans the dataframe.
"""

from logging import log, WARN

import numpy as np
import pandas as pd

# A mapping from feature keys to lists of missing values.
# Note: There are several discrepancies between the codebook and the actual dataset; these names are from the codebook.
#       Many of the features in the codebook do not appear in the dataset, so users should always check for a KeyError.
#       The codebook also capitalizes feature names differently. Users should always convert all feature names (except
#       "Q1_Total_tau") to all-caps.
# TODO: This should be split up into each of the "clean_*()" functions.
NON_PROTEO_MISSING_VALUES = dict(
    CCR_CL_IMPRESS_CODE=[10], A1_Hispanic=[9], A1_Hispanic_Type=[50, 99], A1_RACE=[50, 99], A1_RACESEC=[50, 88, 99],
    A1_RACETER=[50, 88, 99], A1_PRIMLANG=[8, 9], A1_EDUC=[99], A1_LIVSIT=[9], A1_INDEPEND=[9], A1_RESIDENC=[9],
    A1_MARISTAT=[8, 9], A1_Handedness=[9], A1_REFER=[8, 0], A3_MOMDEM=[9], A3_DADDEM=[9], A3_TWIN=[9], A3_SIBS=[99],
    A3_SIBSDEM=[9, 99], A3_KIDS=[99], A3_KIDSDEM=[99], A41_PMAStDA=[99], A41_PMAStMO=[99], A41_PMAStYR=[9999],
    A41_PMBStDA=[99], A41_PMBStMO=[99], A41_PMBStYR=[9999], A41_PMCStDA=[99], A41_PMCStMO=[99], A41_PMCStYR=[9999],
    A41_PMDStDA=[99], A41_PMDStMO=[99], A41_PMDStYR=[9999], A42_VEAStMO=[99], A42_VEAStDA=[99], A42_VEAStYR=[9999],
    A42_VEAEndMO=[99], A42_VEAEndDA=[99], A42_VEAEndYR=[9999], A42_VEBStMO=[99], A42_VEBStDA=[99], A42_VEBStYR=[9999],
    A42_VEBEndMO=[99], A42_VEBEndDA=[99], A42_VEBEndYR=[9999], A42_VECStMO=[99], A42_VECStDA=[99], A42_VECStYR=[9999],
    A42_VECEndMO=[99], A42_VECEndDA=[99], A42_VECEndYR=[9999], A42_VEDStMO=[99], A42_VEDStDA=[99], A42_VEDStYR=[9999],
    A42_VEDEndMO=[99], A42_VEDEndDA=[99], A42_VEDEndYR=[9999], A43_ADAStMO=[99], A43_ADAStDA=[99], A43_ADAStYR=[9999],
    A43_ADAEndMO=[99], A43_ADAEndDA=[99], A43_ADAEndYR=[9999], A43_ADBStMO=[99], A43_ADBStDA=[99], A43_ADBStYR=[9999],
    A43_ADBEndMO=[99], A43_ADBEndDA=[99], A43_ADBEndYR=[9999], A43_ADCStMO=[99], A43_ADCStDA=[99], A43_ADCStYR=[9999],
    A43_ADCEndMO=[99], A43_ADCEndDA=[99], A43_ADCEndYR=[9999], A43_ADDStMO=[99], A43_ADDStDA=[99], A43_ADDStYR=[9999],
    A43_ADDEndMO=[99], A43_ADDEndDA=[99], A43_ADDEndYR=[9999], A43_ADEStMO=[99], A43_ADEStDA=[99], A43_ADEStYR=[9999],
    A43_ADEEndMO=[99], A43_ADEEndDA=[99], A43_ADEEndYR=[9999], A43_ADFStMO=[99], A43_ADFStDA=[99], A43_ADFStYR=[9999],
    A43_ADFEndMO=[99], A43_ADFEndDA=[99], A43_ADFEndYR=[9999], A44_SSAStMO=[99], A44_SSAStDA=[99], A44_SSAStYR=[9999],
    A44_SSBStMO=[99], A44_SSBStDA=[99], A44_SSBStYR=[9999], A44_SSCStMO=[99], A44_SSCStDA=[99], A44_SSCStYR=[9999],
    A44_SSDStMO=[99], A44_SSDStDA=[99], A44_SSDStYR=[9999], A44_SSEStMO=[99], A44_SSEStDA=[99], A44_SSEStYR=[9999],
    A44_SSFStMO=[99], A44_SSFStDA=[99], A44_SSFStYR=[9999], A44_Drg_Trial=[9], A5_CVHATT=[9], A5_CVAFIB=[9],
    A5_CVANGIO=[9], A5_CVBYPASS=[9], A5_CVPACE=[9], A5_CVCHF=[9], A5_CVOTHR=[9], A5_CBSTROKE=[9], A5_STROK1YR=[9999],
    A5_STROK2YR=[9999], A5_STROK3YR=[9999], A5_STROK4YR=[9999], A5_STROK5YR=[9999], A5_STROK6YR=[9999], A5_CBTIA=[9],
    A5_TIA1YR=[9999], A5_TIA2YR=[9999], A5_TIA3YR=[9999], A5_TIA4YR=[9999], A5_TIA5YR=[9999], A5_TIA6YR=[9999],
    A5_CBOTHR=[9], A5_PD=[9], A5_PDYR=[9999], A5_PDOTHR=[9], A5_PDOTHRYR=[9999], A5_SEIZURES=[9], A5_TRAUMBRF=[9],
    A5_TRAUMEXT=[9], A5_TRAUMCHR=[9], A5_NCOTHR=[9], A5_HYPERTEN=[9], A5_HYPERCHO=[9], A5_DIABETES=[9], A5_B12DEF=[9],
    A5_THYROID=[9], A5_INCONTU=[9], A5_INCONTF=[9], A5_CANCER=[9], A5_DEP2YRS=[9], A5_DEPOTHR=[9], A5_ALCOHOL=[9],
    A5_TOBACLstYr=[9], A5_TOBAC30=[9], A5_TOBAC100=[9], A5_PACKSPER=[9], A5_ABUSOTHR=[9], A5_PSYCDIS=[9], A5_IBD=[9],
    A5_Arthritic=[9], A5_AutoImm=[9], A5_Chron_Oth=[9], B1_VISION=[9], B1_VISCORR=[9], B1_VISWCORR=[9], B1_HEARING=[9],
    B1_HEARAID=[9], B1_HEARWAID=[9], B5_DEPD=[9], B5_DISN=[9], C1_MMSE=[99], C1_WAIS3_DIGIF=[99], C1_WAISR_DIGIF=[99],
    C1_WMSR_DIGIF=[99], C1_WAIS3_DIGIB=[99], C1_WAISR_DIGIB=[99], C1_WMSR_DIGIB=[99], C1_WAIS3_DIGILF=[99],
    C1_WAISR_DIGILF=[99], C1_WMSR_DIGILF=[99], C1_WAIS3_DIGILB=[99], C1_WAISR_DIGILB=[99], C1_WMSR_DIGILB=[99],
    C1_WAIS3_DIGTot=[99], C1_WAISR_DIGTot=[99], C1_WMSR_DIGTot=[99], C1_TRAILA=[99], C1_TRAILAErr=[99], C1_TRAILB=[99],
    C1_TRAILBErr=[99], C1_CLOX1=[99], C1_CLOX2=[99], C1_CLOCK=[999], C1_WMS3_LMEM1=[99], C1_WMS3_LMEM2=[99],
    C1_WMSR_LMEM1=[99], C1_WMSR_LMEM2=[99], C1_BOSTON30=[99], C1_BOSTON60=[99], C1_FAS_F=[99], C1_FAS_A=[99],
    C1_FAS_S=[99], C1_Animal=[99], C1_AMNART=[99], C1_WAT=[99], C1_WMSR_VRI=[99], C1_WMSR_VRII=[99], C1_GDS15=[99],
    C1_GDS30=[99], C1_SS_WAIS3_DigTot=[-9], C1_SS_WAISR_DigTot=[-9], C1_SS_WMSR_DigTot=[-9], C1_SS_WMS3_LM_I=[-9],
    C1_SS_WMSR_LM_I=[-9], C1_SS_WMS3_LM_II=[-9], C1_SS_WMSR_LM_II=[-9], C1_SS_WMS3_VR_I=[-9], C1_SS_WMSR_VR_I=[-9],
    C1_SS_WMS3_VR_II=[-9], C1_SS_WMSR_VR_II=[-9], C1_TCS_LS=[99], C1_TCS_DK=[99], C1_TCS_PSV=[99], C1_TCS_OTH=[99],
    D1_WHODIDDX=[0], E1_Reside_care=[3], F2_IADL3=[0], F2_IADL4=[0], F2_IADL5=[0], F2_IADL6=[0], F2_IADL8=[0],
    I1_INHISPOR=[50, 99], I1_INRACE=[50, 99], I1_INRASEC=[50, 99], I1_INRATER=[50, 99], I1_INEDUC=[50, 99],
    I1_INRELTO=[7], P1_PtTypeDesc=["Other"], P1_WhyNotAll=[3], X1_DeclineAgePhy=[999], X1_DeclineAge=[999],
    X1_PhyEstDur=[99], X1_Hyperlip_Has=[9], X1_Hyper_Has=[9], X1_Obese_Is=[9], X1_Smoke_Ever=[9], X1_Smoke_Cur=[9],
    X1_Atrial_SR=[9], X1_Arrhy_SR=[9], X1_MI_SR=[9], X1_CHF_SR=[9], X1_Angina_SR=[9], X2_PhyEst=[99],
    X2_PhyEstFirm=[99], Q1_YKL_40=[-999, -888, -555, -333, -444], Q1_GFAP=[-999, -888, -555, -333, -444],
    Q1_NFL=[-999, -888, -555, -333, -444], Q1_Total_tau=[-999, -888, -555, -333, -444],
    Q1_UCHL1=[-999, -888, -555, -333, -444], Q1_CD14=[-999, -888, -555, -333, -444]
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
            "A1_RESIDENC"
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

    # Drop all "A4" columns except for the vitamin E strengths.
    df.drop(
        list(filter(lambda name: name.startswith("A4") and name != "A42_VEAS", df.columns)),
        axis=1,
        inplace=True
    )


def clean_family_history(df: pd.DataFrame) -> None:
    """
    Modifies the input dataframe to clean up the family history (A3) section. We merge the dad or mom having
    dementia into one feature which is the proportion of parents with dementia.
    Args:
        df: The dataframe representing the TARCC dataset.
    Returns:
        None
    """

    missing_value_codes = {
        "A3_MOMDEM": [9],
        "A3_DADDEM": [9]
    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)

    # Creating 'PROP_PARENTS_DEM' which is the proportion of parents with dementia
    df['PROP_PARENTS_DEM'] = (df['A3_MOMDEM'] + df['A3_DADDEM']) / 2

    df.drop(
        [
            "A3_MOMDEM",
            "A3_DADDEM"
        ],
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

    # "['A5_TOBACLstYr', 'A5_Arthritic', 'A5_AutoImm', 'A5_Chron_Oth', 'A5_Chron_OthX'] not in index"
    missing_value_codes = {
        "A5_CVHATT": [9],
        "A5_CVAFIB": [9],
        "A5_CVANGIO": [9],
        "A5_CVBYPASS": [9],
        "A5_CVPACE": [9],
        "A5_CVCHF": [9],
        "A5_CVOTHR": [9],
        "A5_CBSTROKE": [9],
        "A5_STROK1YR": [9999, ' '],
        "A5_STROK2YR": [9999, ' '],
        "A5_STROK3YR": [9999, ' '],
        "A5_STROK4YR": [9999, ' '],
        "A5_STROK5YR": [9999, ' '],
        "A5_STROK6YR": [9999, ' '],
        "A5_CBTIA": [9],
        "A5_TIA1YR": [9999, ' '],
        "A5_TIA2YR": [9999, ' '],
        "A5_TIA3YR": [9999, ' '],
        "A5_TIA4YR": [9999, ' '],
        "A5_TIA5YR": [9999, ' '],
        "A5_TIA6YR": [9999, ' '],
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

    }
    for feature_name in missing_value_codes:
        df[feature_name].replace(missing_value_codes[feature_name], np.nan, inplace=True)

    # Creating 'NUM_STROKES' which is the number of strokes a patient has had
    df['NUM_STROKES'] = np.sum(~pd.isna(df[['A5_STROK1YR','A5_STROK2YR','A5_STROK3YR','A5_STROK4YR', 'A5_STROK5YR', 'A5_STROK6YR']]), axis=1)

    # Creating 'NUM_TIA' which is the number of TIAs a patient has had
    df['NUM_TIA'] = np.sum(~pd.isna(df[['A5_TIA1YR','A5_TIA2YR','A5_TIA3YR','A5_TIA4YR', 'A5_TIA5YR', 'A5_TIA6YR']]), axis=1)

    df.drop(
        [
            "A5_CVOTHR",
            "A5_CVOTHRX",
            "A5_CBOTHR",
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

def clean_cognitive_tests(df: pd.DataFrame) -> None:

    keep = ["C1_WAIS3_DIGTOT", "C1_WAISR_DIGTOT",
            "C1_MMSE", "C1_CDRSUM", "C1_CDRGLOB",
            "C1_SS_TRAILA", "C1_SS_TRAILB",
            "C1_CLOCK",
            "C1_WMS3_LMEM1", "C1_WMS3_LMEM2", "C1_WMS3_VRI", "C1_WMS3_VR2",
            "C1_WMSR_LMEM1", "C1_WMSR_LMEM2", "C1_WMSR_VRI", "C1_WMSR_VRII", "C1_WMSR_DIGTOT",
            "C1_GDS30",
            "C1_AMNART"]

    # Drop all "C1" columns except for the vitamin E and anti-dementia drug strengths.
    df.drop(
        list(filter(lambda name: name.startswith("C1") and name not in keep, df.columns)),
        axis=1,
        inplace=True
    )

def clean_exit(df: pd.DataFrame) -> None:
    """
    Eliminates the columns specifying method of exit for patients.
    Also removes probable/possible AD features which are highly correlated with response
    Inputs: 
        pandas dataframe of clinical/biomarker data
    Returns:
        None, alters the inputted dataframe
    """
    unwanted_cols = []

    for col in df.columns:
        if col.startswith("E1"):
            unwanted_cols.append(col)
        if col.startswith("D1"):
            if col in ['D1_PROBAD','D1_PROBADIF','D1_POSSAD','D1_POSSADIF', 'D1_WHODIDDX']:
                unwanted_cols.append(col)
    df.drop(unwanted_cols, axis=1, inplace=True)
    
def clean_D1(df: pd.DataFrame) -> None:
    """
    Eliminates the columns specifying diagnostic indicators for patients
    Inputs: 
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
    df['D1_total'] = window.sum(axis = 1)

def get_cleaned_data() -> pd.DataFrame:
    """
    Reads the CSV file and returns the cleaned dataframe.
    Returns:
        The cleaned dataframe representing the TARCC data.
    """
    df = pd.read_csv("data/TARCC_data.csv")

    # Replace all empty strings with NaN, and convert all relevant columns to numeric (float).
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # TODO: The following lines convert all features to numeric and remove all textual columns.
    # TODO: This behavior should be deferred to each of the "clean_*()" functions.
    df = df.apply(pd.to_numeric, errors="ignore")  # TODO: This may be removed after cleaning each section.
    # # Drop all text columns.
    # df = df.drop(columns=df.select_dtypes("object"))

    # Clean each section of the dataset.
    clean_demographics(df)
    clean_medicinal_history(df)
    clean_family_history(df)
    clean_cognitive_tests(df)
    clean_exit(df)
    clean_D1(df)
    sum_D1(df)

    # Replace missing values with NaN.
    for key, value in NON_PROTEO_MISSING_VALUES.items():
        actual_key = key if key == "Q1_Total_tau" else key.upper()
        try:
            df[actual_key].replace(value, np.nan, inplace=True)
        except KeyError:
            log(WARN, f"Key {actual_key} is missing from the dataframe.")

    # For proteo features, replace missing values with NaN and replace LLDL and GHDL with 0 and 999999999, respectively.
    for key in df.columns[[column_name.startswith("PROTEO") for column_name in df.columns]]:
        df[key].replace([-777777, 0], np.nan, inplace=True)
    df = df.replace(-888888, 999999999)
    df = df.replace(-999999, 0)

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

    blood_feats = ["PROTEO", "RBM", "Q1", "X1", "X2", "P1", "PATID"]

    filtered_feats = list(filter(lambda name: any(name.startswith(prefix) for prefix in blood_feats), original_df.columns))

    new_df1 = original_df[filtered_feats]
    filtered_feats.remove("PATID")
    new_df2 = original_df.drop(filtered_feats, axis=1)

    new_df1.to_csv("Blood Only Data.csv", index=False)
    new_df2.to_csv("Clinical Only Data.csv", index=False)

    return new_df1, new_df2


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

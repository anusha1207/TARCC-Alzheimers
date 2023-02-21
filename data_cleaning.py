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


def get_cleaned_data() -> pd.DataFrame:
    """
    Reads the CSV file and returns the cleaned dataframe.
    Returns:
        A cleaned dataframe.
    """

    df = pd.read_csv("TARCC_data.csv")

    # Replace all empty strings with NaN, and convert all relevant columns to numeric (float).
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop all text columns.
    df = df.drop(columns=df.select_dtypes("object"))

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

df = get_cleaned_data()
# print(df.columns)
# df.to_csv("Cleaned data.csv", index=False)


def split_csv(original_df):
    """
    Takes in the original dataset and creates two new Data Frames, one with only clinical data
    and the other with only blood data.

    Args:
        original_df: The original dataset
    Returns:
        Two new data frames that contain only one type of data (eit"her Blood or Clinical).
    """

    blood_columns = []

    # Get the column names of those that include blood and bio-marker data
    for col in df.columns:
        if col.startswith("PROTEO"):
            blood_columns.append(col)
        elif col.startswith("RBM"):
            blood_columns.append(col)
        elif col.startswith("Q1"):
            blood_columns.append(col)
        elif col.startswith("X1") or col.startswith("X2"):
            blood_columns.append(col)
        elif col.startswith("P1"):
            blood_columns.append(col)
        elif col.startswith("PATID"):
            blood_columns.append(col)

    new_df1 = original_df[blood_columns]

    blood_columns.remove("PATID")

    new_df2 = original_df.drop(blood_columns, axis=1)

    new_df1.to_csv("Blood Only Data.csv", index=False)
    new_df2.to_csv("Clinical Only Data.csv", index=False)

    return new_df1, new_df2


def update_wanted_features(df):

    unwanted_cols = []

    for col in df.columns:

        if col.startswith("F1"):
            if col == "F1_PSMSTOTSCR":
                continue
            else:
                unwanted_cols.append(col)

        elif col.startswith("F2"):
            if col == "F2_IADLTOTSCR":
                continue
            else:
                unwanted_cols.append(col)

        elif col.startswith("I1"):
            unwanted_cols.append(col)

        elif col.startswith("P1"):
            if col == "P1_PT_TYPE":
                continue
            else:
                unwanted_cols.append(col)

        elif col.startswith("X1") or col.startswith("X2"):
            print(col)
            unwanted_cols.append(col)

        # Need to ask John about PID and GWAS

    updated_df = df.drop(unwanted_cols, axis=1)

    return updated_df

new_df = update_wanted_features(df)

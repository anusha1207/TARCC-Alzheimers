## Load libraries

import pandas as pd
import numpy as np
import regex as re




def preprocessing(data):

  #subset the data to the patient visits where blood samples are taken
  df_with_RBM = data[data['RBM_Rule_Based_Medicine']==1]
  #retain only relevant blood + protein + disorders features with biological factors like age, sex,etc.
  df = df_with_RBM[['PATID', 'AGE','A1_SEX', 'A3_DADDEM', 'A3_MOMDEM', 'A5_ALCOHOL', 'A5_ARTHRITIC', 'A5_AUTOIMM', 'A5_B12DEF',
 'A5_CANCER', 'A5_CBSTROKE', 'A5_CBTIA', 'A5_CVAFIB', 'A5_CVANGIO', 'A5_CVBYPASS', 'A5_CVCHF', 'A5_CVHATT', 'A5_CVPACE',
 'A5_DIABETES', 'A5_HYPERCHO','A5_HYPERTEN', 'A5_IBD', 'A5_INCONTF', 'A5_INCONTU', 'A5_PD', 'A5_PSYCDIS', 'A5_THYROID',
 'A5_TRAUMBRF', 'A5_TRAUMCHR', 'B1_BMI', 'B1_BPDIAS', 'B1_BPSYS', 'B5_DEL', 'B5_HALL', 'B5_AGIT', 'B5_DEPD', 'B5_ANX',
 'B5_ELAT', 'B5_APA', 'B5_DISN', 'B5_IRR', 'B5_MOT', 'B5_NITE', 'B5_APP', 'D1_BRNINJ', 'D1_CORT', 'D1_DEP', 'D1_DYSILL',
 'D1_HYCEPH', 'D1_NEOP', 'D1_PARK', 'D1_PPAPH', 'D1_STROKE', 'D1_VASC', 'X1_OBESE_AB', 'X1_OBESE_IS','P1_PT_TYPE',
 'PROTEO_ADIPONECTIN', 'PROTEO_BFGF', 'PROTEO_CLUSTERIN', 'PROTEO_CRP', 'PROTEO_EOTAXIN_HUMAN', 'PROTEO_FABP3', 'PROTEO_FACTOR_VII',
 'PROTEO_FLT_1', 'PROTEO_GLUCAGON', 'PROTEO_GM_CSF_HUMAN', 'PROTEO_IFN_Y_HUMAN', 'PROTEO_IL_10_HUMAN',
 'PROTEO_IL_12_P40_HUMAN','PROTEO_IL_15', 'PROTEO_IL_16', 'PROTEO_IL_1B_HUMAN', 'PROTEO_IL_2_HUMAN', 'PROTEO_IL_5_HUMAN', 'PROTEO_IL_6_HUMAN',
 'PROTEO_IL_7_HUMAN', 'PROTEO_IL_8_HUMAN', 'PROTEO_INSULIN', 'PROTEO_IP_10_HUMAN', 'PROTEO_LBP', 'PROTEO_LEPTIN', 'PROTEO_MCP_1_HUMAN',
 'PROTEO_MCP_4_HUMAN', 'PROTEO_MDC_HUMAN', 'PROTEO_MIP_1A_HUMAN', 'PROTEO_MIP_1B_HUMAN', 'PROTEO_MPO', 'PROTEO_NT_PROBNP', 'PROTEO_PIGF',
 'PROTEO_PYY', 'PROTEO_RESISTIN', 'PROTEO_SAA', 'PROTEO_SICAM_1', 'PROTEO_SVCAM_1', 'PROTEO_TARC_HUMAN', 'PROTEO_TIE_2', 'PROTEO_TNF_A_HUMAN',
 'PROTEO_TNF_B', 'PROTEO_TOTAL_GIP', 'PROTEO_VEGF_ANGIO_PLATE', 'PROTEO_VEGF_CYTO_PLATE', 'PROTEO_VEGF_C', 'PROTEO_VEGF_D', 'RBM_ACE_CD143',
 'RBM_Adiponectin', 'RBM_AgRP', 'RBM_Alpha_1', 'RBM_Alpha_2', 'RBM_Alpga_F', 'RBM_Amphiregulin', 'RBM_ANG_2', 'RBM_Angiotensinogen',
 'RBM_APO_A1', 'RBM_APO_CIII','RBM_APO_H', 'RBM_AXL', 'RBM_BLC', 'RBM_B2M','RBM_BTC', 'RBM_BDNF', 'RBM_CRP', 'RBM_CA_125', 'RBM_CA_19_9',
 'RBM_CEA', 'RBM_CD40', 'RBM_CD40L', 'RBM_CgA', 'RBM_Complement_3', 'RBM_Cortisol', 'RBM_CK_MB', 'RBM_CTGF', 'RBM_EGF','RBM_EGF_R', 'RBM_ENA_78',
 'RBM_EN_RAGE', 'RBM_Eotaxin', 'RBM_Eotaxin_3', 'RBM_Epiregulin', 'RBM_Factor_VII', 'RBM_FAS', 'RBM_FASL', 'RBM_FABP', 'RBM_Ferritin', 'RBM_Fibrinogen',
 'RBM_FSH', 'RBM_G_CSF', 'RBM_GSTs', 'RBM_GRO_alpha', 'RBM_GH', 'RBM_Haptoglobin', 'RBM_HB_EGF', 'RBM_HCC_4', 'RBM_HGF', 'RBM_I_309', 'RBM_ICAM_1',
 'RBM_IFNg', 'RBM_IgA', 'RBM_IgE', 'RBM_IGF_1', 'RBM_IGF_BP_2', 'RBM_IgM', 'RBM_IL_10', 'RBM_IL_12p40', 'RBM_IL_13', 'RBM_IL_15', 'RBM_IL_16',
 'RBM_IL_18', 'RBM_IL_1ra', 'RBM_IL_3', 'RBM_IL_5', 'RBM_IL_7', 'RBM_IL_8', 'RBM_Insulin', 'RBM_Leptin', 'RBM_LH', 'RBM_Lpa', 'RBM_MCP_1',
 'RBM_MDC', 'RBM_MIF', 'RBM_MIP_1a', 'RBM_MIP_1b', 'RBM_MMP_3', 'RBM_MPO', 'RBM_Myoglobin', 'RBM_PAI_1', 'RBM_PP', 'RBM_PDGF', 'RBM_Progesterone',
 'RBM_Prolactin', 'RBM_PAP', 'RBM_PARC', 'RBM_RANTES', 'RBM_Resistin', 'RBM_S100b', 'RBM_SAP', 'RBM_SGOT', 'RBM_SHBG', 'RBM_SOD', 'RBM_Sortilin',
 'RBM_sRAGE', 'RBM_SCF', 'RBM_Tenascin_C', 'RBM_Testosterone', 'RBM_TGF_alpha', 'RBM_THPO', 'RBM_THPO_1', 'RBM_TECK', 'RBM_TSH', 'RBM_TBG',
 'RBM_TIMP_1', 'RBM_TF', 'RBM_TNF_RII', 'RBM_TNF_alpha', 'RBM_TNF_beta', 'RBM_TRAIL_R3', 'RBM_VCAM_1', 'RBM_VEGF', 'RBM_VWF']]
  
  #replace empty strings with NaN values
  df = df.replace(r'^\s*$', np.nan, regex=True)
  #removing rows which contain missing vals in all RBM features
  all_cols = df.columns
  regex_rbm = re.compile("^RBM_.*$")
  rbm_cols = list(filter(regex_rbm.match, all_cols))
  df = df[df[rbm_cols].apply(pd.Series.nunique, axis=1) > 1]

  #Removing rows with missing vals in proteo features
  missing_rows = list(df[df == -777777].count()[df[df == -777777].count() > 0].index)
  indices = []
  for row in missing_rows:
    for ind in list(df[df[row] == -777777].index):
      indices.append(ind)

  indices = [*set(indices)]
  indices
  df = df.drop(indices, axis=0)

  #Tranforming extreme values in proteo features
  # -888888(GHDL) mapped to highest possible value and LLDL mapped to least value
  df = df.replace(-888888, 999999999)
  df = df.replace(-999999, 0)

  #convert categorical object columns to floats
  categoricalcols=list(df.dtypes[df.dtypes==object].index)
  df[categoricalcols]=df[categoricalcols].astype(float)

  #removing patient record which has 2 visits
  df = df.drop(3217, axis=0)

  #removing diagnosis variables that are 3 or 4
  df = df[df['P1_PT_TYPE'].isin([1,2])]
  return df

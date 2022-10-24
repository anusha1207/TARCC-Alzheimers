## Load libraries

import pandas as pd
import numpy as np




def preprocessing(df):
  """"
  This function drops columns with more than 60% NAs or no relation to alzheimers.
  Input: raw dataframe
  Output: cleaned dataframe
  """"
  mod_df = df.replace(r'^\s*$', np.nan, regex=True)
  min_count =  int((40/100)*mod_df.shape[0] + 1) #60% is NA
  mod_df = mod_df.dropna(axis=1, thresh=min_count)
  
  #Study ID, PatID, no of visits, misc id, ccr, misc site id and birth year
  df2 = mod_df.drop(mod_df.iloc[:, 0:7],axis = 1) 
  
  #Event date, handedness, hispanic, marital status
  df2 = df2.drop(df2.iloc[:, 1:5],axis = 1)
  

  df2=df2.drop([], axis=1)
  #df2 = df2.drop(df2.iloc[:, 31:54],axis = 1)
  #ALL THE COLUMNS WITH OTHER are removed, and so is information about date, vision and hearing since they have no correlation
  #other columns such as boolean non disease related boolean values are also dropped like :
  #was sample collected or not, dna test type (buccal/blood etc) etc
  #columns discontinued on or before 2010 are also removed (hyper, hyperlip etc)
  #cerad colum individual trials removed, only total is retained (C1_CERAD_LL_TOT), same with cerad no and yes since cerad wr is there
  df2=df2.drop(['A5_CBOTHR','A5_CHRON_OTH','A5_CVOTHR','A5_DEPOTHR','A5_NCOTHR','A5_PDOTHR', 
                'A5_PDOTHRYR', 'A5_PDYR', 'B1_HEARAID', 'B1_HEARING','B1_HEARWAID', 'B1_VISCORR',
                'B1_VISION','B1_VISWCORR','B5_NPIQINF','C1_DATEX','C1_CDRCA','C1_CDRGLOB','C1_CDRHOB',
                'C1_CDRJU','C1_CDRMEM','C1_CDROR','C1_CDRPER', 'C1_CERAD_LL_1','C1_CERAD_LL_2',
                'C1_CERAD_LL_3','C1_CERAD_WR_NO',
                'C1_CERAD_WR_YES','C1_GDS15', 'P1_WHYNOTALL',
                'P1_WHOLEBLOOD',  'P1_SERUM','P1_SHARE_AGREE','P1_TIMEDRAWX','P1_TIMEFOODX','P1_BIOSERUM',
                'P1_BUFFY','P1_CSF','P1_DATEDRAWX','P1_DATEFOODX','P1_HRSAFTFOOT', 'P1_PLASMA', 
                'E1_WDOTHREAS','E1_WITHDREW', 'RBM_Rule_Based_Medicine','RBM_Rule_Based_Medicine_tp',
                'X2_PHYEST',  'X2_IDURM','X2_IDURY','X2_MRDURM','X2_MRDURY','X1_HYPER_HAS','X1_HYPERLIP_HAS',
                'X1_HYPERLIP_SR', 'P1_INBIOMARKER','P1_INGENETICS', 'D1_WHODIDDX','P1_DNACOLLECTED',
                'P1_DNATYPE'], axis=1)
  #'A41_PMBF' is mostly 0 aka NO. AND REMOVING FREQUENCY (FU- Daily monthly weekly since most are 0)
  #'A42_VEASU',A43_ADAF 'A43_ADAS','A43_ADASU'  'A43_ADBF' 'A44_DRG_TRIAL' since mostly 0
  df2=df2.drop(['A41_PMBF','A43_ADFFU','A43_ADEFU','A43_ADAFU','A41_PMBFU',
                'A41_PMBPFU','A42_VEAFU','A42_VEASU','A43_ADAF',
                'A43_ADAS','A43_ADASU','A43_ADBENDDA','A43_ADBENDMO','A43_ADBENDYR',
                'A43_ADBFU', 'A43_ADBF', 'A43_ADBS','A43_ADBSU','A43_ADCF','A43_ADCFU',
                'A43_ADCS','A43_ADCSU','A43_ADDF','A43_ADDFU','A43_ADDS','A43_ADDSU',
                'A43_ADEF','A43_ADEFU','A43_ADES','A43_ADESU',
                'A43_ADFF','A43_ADFFU','A43_ADFS','A43_ADFSU','A44_DRG_TRIAL', 'A5_INCONTF', 'A5_INCONTU',
                'A5_DEP2YRS','A5_PACKSPER'],axis=1)

  df2=df2.drop(['A44_SSAF','A44_SSAFU','A44_SSAPREV','A44_SSAS','A44_SSBF','A44_SSBFU',
 'A44_SSBPREV','A44_SSBS','A44_SSBSU','A44_SSCF','A44_SSCFU','A44_SSCPREV','A44_SSCS',
 'A44_SSCSU','A44_SSDF','A44_SSDFU','A44_SSDPREV','A44_SSDS','A44_SSDSU','A44_SSEF',
 'A44_SSEFU','A44_SSEPREV','A44_SSES','A44_SSESU','A44_SSFF','A44_SSFFU','A44_SSFPREV',
 'A44_SSFS','A44_SSFSU'],axis=1)
  
  #remove ALL INFORMANT VARIABLES since it does not influence risk of Alzheimer's at all
  df2=df2.drop(['I1_INBIRYR','I1_INCALLS','I1_INDATECONTX','I1_INEDUC','I1_INHISP','I1_INHOWCONTACT',
                'I1_INLIVWTH','I1_INRACE','I1_INRASEC','I1_INRATER','I1_INRELTO','I1_INRELY','I1_INSEX',
                'I1_INVISITS','I1_ISNEWINFORM'],axis=1)
  
  #removing severity of mental health as >12k have missing values
  df2=df2.drop(['B5_DELSEV','B5_HALLSEV','B5_AGITSEV', 'B5_DEPDSEV', 'B5_ANXSEV','B5_ELATSEV',
 'B5_APASEV','B5_DISNSEV','B5_IRRSEV','B5_MOTSEV','B5_NITESEV','B5_APPSEV'], axis=1)
  #C1_LITPROB-whether patient has literacy problem, C1_TOOIMPAIRED-patient impaired or not to test,
  #'C1_TRAILA','C1_TRAILAERR','C1_TRAILB','C1_TRAILBERR' - trials A and B- not significant,
  # D1_ALCDEMIF- 
  df2=df2.drop(['C1_LITPROB','C1_TOOIMPAIRED', 'C1_TRAILA',
 'C1_TRAILAERR','C1_TRAILB','C1_TRAILBERR','D1_ALCDEMIF','D1_BRNINJIF','D1_COGOTHIF','D1_CORTIF',
 'D1_DEMUNIF','D1_DEPIF','D1_DLBIF','D1_DOWNSIF','D1_DYSILLIF','D1_FTDIF','D1_HUNTIF','D1_HYCEPHIF', 'D1_IMPNOMCI',
 'D1_MCIAMEM', 'D1_MCIAPATT','D1_MCIAPEX','D1_MCIAPLAN','D1_MCIAPLUS','D1_MCIAPVIS','D1_MCIN1ATT','D1_MCIN1EX',
 'D1_MCIN1LAN','D1_MCIN1VIS','D1_MCIN2ATT','D1_MCIN2EX','D1_MCIN2LAN','D1_MCIN2VIS','D1_MCINON1','D1_MCINON2',
 'D1_MEDSIF','D1_NEOPIF','D1_OTHPSY','D1_OTHPSYIF','D1_PARK','D1_PARKIF','D1_POS_VASC','D1_POS_VASCIF',
 'D1_POSSADIF','D1_PPAPHIF','D1_PRIONIF','D1_PROBADIF','D1_PSPIF','D1_STROKE','D1_STROKEIF','D1_VASCIF',
 'D1_VASC','D1_PSP','D1_PRION','D1_NEOP','D1_PPAPH','X1_ANGINA_SR','X1_ARRHY_SR','X1_ATRIAL_SR','X1_BMIGT30',
 'X1_SMOKE_CUR','X2_PHYESTFIRM','F1_PSMS1','F1_PSMS2','F1_PSMS3','F1_PSMS4','F1_PSMS5','F1_PSMS6','F2_IADL1',
 'F2_IADL2','F2_IADL3','F2_IADL4','F2_IADL5','F2_IADL6','F2_IADL7','F2_IADL8'], axis=1)
  
  #ALL PROTEO BLOCKS HAVE THE SAME VALUE OF High (9999) so there is no variance
  df2=df2.drop(['PROTEO_ADIPONECTIN',
 'PROTEO_BFGF',
 'PROTEO_CLUSTERIN',
 'PROTEO_CRP',
 'PROTEO_EOTAXIN_HUMAN',
 'PROTEO_EOTAXIN_3_HUMAN',
 'PROTEO_FABP3',
 'PROTEO_FACTOR_VII',
 'PROTEO_FLT_1',
 'PROTEO_GLUCAGON',
 'PROTEO_GM_CSF_HUMAN',
 'PROTEO_IFN_Y_HUMAN',
 'PROTEO_IL_10_HUMAN',
 'PROTEO_IL_12_P40_HUMAN',
 'PROTEO_IL_12_P70_HUMAN',
 'PROTEO_IL_13_HUMAN',
 'PROTEO_IL_15',
 'PROTEO_IL_16',
 'PROTEO_IL_17A',
 'PROTEO_IL_1A',
 'PROTEO_IL_1B_HUMAN',
 'PROTEO_IL_2_HUMAN',
 'PROTEO_IL_4_HUMAN',
 'PROTEO_IL_5_HUMAN',
 'PROTEO_IL_6_HUMAN',
 'PROTEO_IL_7_HUMAN',
 'PROTEO_IL_8_HUMAN',
 'PROTEO_IL_8_HA',
 'PROTEO_INSULIN',
 'PROTEO_IP_10_HUMAN',
 'PROTEO_LBP',
 'PROTEO_LEPTIN',
 'PROTEO_MCP_1_HUMAN',
 'PROTEO_MCP_4_HUMAN',
 'PROTEO_MDC_HUMAN',
 'PROTEO_MIP_1A_HUMAN',
 'PROTEO_MIP_1B_HUMAN',
 'PROTEO_MPO',
 'PROTEO_NT_PROBNP',
 'PROTEO_PIGF',
 'PROTEO_PYY',
 'PROTEO_RESISTIN',
 'PROTEO_SAA',
 'PROTEO_SICAM_1',
 'PROTEO_SVCAM_1',
 'PROTEO_TARC_HUMAN',
 'PROTEO_TIE_2',
 'PROTEO_TNF_A_HUMAN',
 'PROTEO_TNF_B',
 'PROTEO_TOTAL_GIP',
 'PROTEO_VEGF_ANGIO_PLATE',
 'PROTEO_VEGF_CYTO_PLATE',
 'PROTEO_VEGF_C',
 'PROTEO_VEGF_D',
 'Q1_Quanterix',
 'Q1_Quanterix_tp',
 'Q1_YKL_40',
 'Q1_GFAP',
 'Q1_NFL',
 'Q1_Total_tau',
 'Q1_UCHL1',
 'Q1_CD14',
 'P1_PTTYPEDESC'], axis=1)
  df2 = df2.dropna(subset=['D1_NORMCOG'])
  #AGE categories
  # category=pd.cut(df2.AGE,bins=[49,60,70,80,90,100,110],labels=['Fifties', 'Sixties', 'Seventies','Eighties','Nineties','100+'])
  # df2.insert(7, 'AGE_GROUP', category)
  # df2=df2.drop(['AGE'])

  #fill na vals as -9
  nullvals=list(df2.isna().sum()[df2.isna().sum()>0].index)
  df2[nullvals]=df2[nullvals].fillna(-9)

  #convert categorical object columns to floats
  categoricalcols=list(df2.dtypes[df2.dtypes==object].index)
  df2[categoricalcols]=df2[categoricalcols].astype(float)

#ACCORDING TO PUBLISHED PAPERS, A WAIS DIGIT SCORE OF >5 MEANS HEALTHY INDIVIDUAL (0)
#Below 5 indicates cognitive impairment. (1)
  df2["C1_WAIS3_DIGIF"] = pd.to_numeric(df2["C1_WAIS3_DIGIF"])
  df2.loc[(df2["C1_WAIS3_DIGIF"] < 5), "C1_WAIS3_DIGIF"] = 1
  df2.loc[(df2["C1_WAIS3_DIGIF"] >= 5), "C1_WAIS3_DIGIF"] = 0
# GERIATRIC DEPRESSION SCALE
#Scores of 0 - 9 are considered normal (0)
# 10 - 19 indicate mild depression  (1)
#20 - 30 indicate severe depression. (2)
  df2.loc[(df2["C1_GDS30"] <= 9), "C1_GDS30"] = 0
  df2.loc[(df2["C1_GDS30"] > 9) & (df2["C1_GDS30"] <= 19), "C1_GDS30"] = 1
  df2.loc[(df2["C1_GDS30"] >= 20), "C1_GDS30"] = 2

  return(df2)

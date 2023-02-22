from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

codebook = pd.read_excel("20220602-TARCC-Codebook.xlsx")

# ensure codebook variable names are strings, and in all caps
codebook["Variable Name"] = codebook["Variable Name"].astype(str)
codebook["Variable Name"] = codebook["Variable Name"].apply(str.upper)

names_to_codebook_proteo = {"PROTEO_ADIPONECTIN" : "Proteomics_Adiponectin", "PROTEO_BFGF" : "Proteomics_bFGF", "PROTEO_CLUSTERIN" : "Proteomics_Clusterin",
"PROTEO_CRP" : "Proteomics_CRP", "PROTEO_EOTAXIN_HUMAN" : "Proteomics_Eotaxin (Human)", "PROTEO_EOTAXIN_3_HUMAN" : "Proteomics_Eotaxin-3 (Human)", "PROTEO_FABP3" : "Proteomics_FABP3",
"PROTEO_FACTOR_VII" : "Proteomics_Factor VII", "PROTEO_FLT_1" : "Proteomics_Flt-1", "PROTEO_GLUCAGON" : "Proteomics_Glucagon", "PROTEO_GM_CSF_HUMAN" : "Proteomics_GM-CSF (Human)",
"PROTEO_IFN_Y_HUMAN" : "Proteomics_IFN-y (Human)", "PROTEO_IL_10_HUMAN" : "Proteomics_IL-10 (Human)", "PROTEO_IL_12_P40_HUMAN" : "Proteomics_IL-12 p40 (Human)", 
"PROTEO_IL_12_P70_HUMAN" : "Proteomics_IL-12 p70 (Human)", "PROTEO_IL_13_HUMAN" : "Proteomics_IL-13 (Human)", "PROTEO_IL_15" : "Proteomics_IL-15", "PROTEO_IL_16" : "Proteomics_IL-16",
"PROTEO_IL_17A" : "Proteomics_IL-17A", "PROTEO_IL_1A" : "Proteomics_IL-1a", "PROTEO_IL_1B_HUMAN" : "Proteomics_IL-1B (Human)", "PROTEO_IL_2_HUMAN" : "Proteomics_IL-2 (Human)",
"PROTEO_IL_4_HUMAN" : "Proteomics_IL-4 (Human)", "PROTEO_IL_5_HUMAN" : "Proteomics_IL-5 (Human)", "PROTEO_IL_6_HUMAN" : "Proteomics_IL-6 (Human)", "PROTEO_IL_7_HUMAN" : "Proteomics_IL-7 (Human)", 
"PROTEO_IL_8_HUMAN" : "Proteomics_IL-8 (Human)", "PROTEO_IL_8_HA" : "Proteomics_IL-8 HA", "PROTEO_INSULIN" : "Proteomics_Insulin", "PROTEO_IP_10_HUMAN" : "Proteomics_IP-10 (Human)",
"PROTEO_LBP" : "Proteomics_LBP", "PROTEO_LEPTIN" : "Proteomics_Leptin", "PROTEO_MCP_1_HUMAN" : "Proteomics_MCP-1 (Human)", "PROTEO_MCP_4_HUMAN" : "Proteomics_MCP-4 (Human)",
"PROTEO_MDC_HUMAN" : "Proteomics_MDC (Human)", "PROTEO_MIP_1A_HUMAN" : "Proteomics_MIP-1a (Human)", "PROTEO_MIP_1B_HUMAN" : "Proteomics_MIP-1B (Human)", "PROTEO_MPO" : "Proteomics_MPO", 
"PROTEO_NT_PROBNP" : "Proteomics_NT-proBNP", "PROTEO_PIGF" : "Proteomics_PIGF", "PROTEO_PYY" : "Proteomics_PYY", "PROTEO_RESISTIN" : "Proteomics_Resistin", "PROTEO_SAA" : "Proteomics_SAA",
"PROTEO_SICAM_1" : "Proteomics_sICAM-1", "PROTEO_SVCAM_1" : "Proteomics_sVCAM-1", "PROTEO_TARC_HUMAN" : "Proteomics_TARC (Human)", "PROTEO_TIE_2" : "Proteomics_TIE-2", 
"PROTEO_TNF_A_HUMAN" : "Proteomics_TNF-a (Human)", "PROTEO_TNF_B" : "Proteomics_TNF-B", "PROTEO_TOTAL_GIP" : "Proteomics_Total GIP", "PROTEO_VEGF_ANGIO_PLATE" : "Proteomics_VEGF- angio plate", 
"PROTEO_VEGF_CYTO_PLATE" : "Proteomics_VEGF- cyto plate", "PROTEO_VEGF_C" : "Proteomics_VEGF-C", "PROTEO_VEGF_D" : "Proteomics_VEGF-D"}

def get_desc_table(cols):
    descs = []
    for col in cols:
        entry = {}
        if col in names_to_codebook_proteo:
            col = names_to_codebook_proteo[col]
        col = str(col).upper()
        row = codebook[codebook["Variable Name"] == col]
        entry["Variable Name"] = col
        entry["Description"] = row["Description"].values
        descs.append(entry)
    for entry in descs:
        if len(entry["Description"]) != 0:
            entry["Description"] = entry["Description"][0]
        else:
            entry["Description"] = "Incompatible"
    return descs
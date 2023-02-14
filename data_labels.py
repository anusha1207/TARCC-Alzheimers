from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

codebook = pd.read_excel("20220602-TARCC-Codebook.xlsx")

def get_desc_table(cols) -> dict:
    descs = {}
    for col in cols:
        row = codebook.query(f"`Variable Name` == '{col}'")
        descs[col] = row["Description"][0]
    return descs
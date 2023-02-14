from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

codebook = pd.read_excel("20220602-TARCC-Codebook.xlsx")

def get_desc(col) -> str:
    q = f"`Variable Name` == '{col}'"
    row = codebook.query(q)
    return row["Description"][0]

def get_desc_table(cols) -> dict:
    descs = {}
    for col in cols:
        descs[col] = get_desc(col)
    return descs
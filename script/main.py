###
#This main module is to run all dependent modules
###

from script import modeling as m
import pandas as pd
from config import config

non_genetic_df = pd.read_csv(config.DATA, low_memory=False)

m.model_main(non_genetic_df)


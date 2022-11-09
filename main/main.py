###
#This main module is to run all dependent modules
###

from Modeling import modeling as m
import pandas as pd
from Configurations import config

non_genetic_df = pd.read_csv('20220916 updated TARCC Data for Dr Broussard.csv', low_memory=False)
non_genetic_df = pd.read_csv(config.DATA, low_memory=False)

m.model_main(non_genetic_df, dataset='blood')


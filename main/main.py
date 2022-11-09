###
#This main module is to run all dependent modules
###

from Modeling import modeling as m
import pandas as pd
from Configurations import config

non_genetic_df = pd.read_csv(config.DATA, low_memory=False)

m.model_main(non_genetic_df, dataset='blood')


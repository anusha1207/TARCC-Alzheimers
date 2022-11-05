###
#This main module is to run all dependent modules
###

import modeling as m
import pandas as pd
import config

non_genetic_df = pd.read_csv(config.DATA, low_memory=False)

m.model_main(non_genetic_df, dataset='blood')

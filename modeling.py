import preprocessing as pp

### Load data
non_genetic_data = pd.read_csv("20220916 updated TARCC Data for Dr Broussard.csv")

df = pp.preprocessing(non_genetic_data)
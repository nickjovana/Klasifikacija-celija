import pandas as pd
import numpy as np
import gc

fajl1=pd.read_csv("007_HCC_cells_csv.csv", index_col = False)
fajl2=pd.read_csv("008_HCC_cells_csv.csv", index_col = False)
fajl3=pd.read_csv("009_HCC_cells_csv.csv", index_col = False)

fajl1 = fajl1.T
kolone1 = fajl1.iloc[0, :]
fajl1 = fajl1.iloc[1:, :]
fajl1.columns = kolone1
gc.collect()

fajl2 = fajl2.T
kolone2 = fajl2.iloc[0, :]
fajl2 = fajl2.iloc[1:, :]
fajl2.columns = kolone2
gc.collect()

fajl3 = fajl3.T
kolone3 = fajl3.iloc[0, :]
fajl3 = fajl3.iloc[1:, :]
fajl3.columns = kolone3
gc.collect()

fajl1['class'] = 0
fajl2['class'] = 1
fajl3['class'] = 2
gc.collect()

spojeni = pd.concat([fajl1,fajl2,fajl3], axis = 0, ignore_index = True)
spojeni.loc[:, (spojeni != 0).any(axis=0)]
spojeni.dropna() 
spojeni.to_csv('./spojeni1.csv', index = False)
  

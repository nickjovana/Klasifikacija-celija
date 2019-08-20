#importujemo potrebne biblioteke
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#ucitavanje podataka
tabela1=pd.read_csv("009_HCC_cells_csv.csv")

#broj redova i kolona grafici--------
redovi = tabela1.shape[0]  
kolone = tabela1.shape[1]
print(redovi,kolone)  
#brisanje nula i autlejera

tabela1.drop(tabela1.columns[[0]], axis=1, inplace=True)
tabela10 = tabela1[(tabela1.T != 0).any()]

 
#autlejeri--------
i=1
j=0
brojac=0
for i in range(1, redovi):
	if brojac==1:
		tabela10.drop(tabela10.index[i])
	for j in range(0,kolone):
		if tabela10.iloc[i,j]!=0:
			brojac=brojac+1
		if brojac>1:
			brojac=0
			break			
redovi = tabela10.shape[0]  
kolone = tabela10.shape[1]
print(redovi,kolone)  
#broj redova i kolona nova grafici-----

#normalizacija
x = tabela10.values
standard_scaler = StandardScaler()
x_standardized = standard_scaler.fit_transform(x)
tabela10=pd.DataFrame(x_standardized, index=tabela10.index.values)

#pca
pca = PCA(3)
x = tabela10.values
x = pca.fit_transform(x)
tabela10=pd.DataFrame(x, index=tabela10.index.values)
tabela10.to_csv("fajl3.csv", encoding='utf-8', index=False)

#grafik podataka novih---------------
y = tabela10.values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[:, 0], y[:, 1], y[:, 2])
plt.show()

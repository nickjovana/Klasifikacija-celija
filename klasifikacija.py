import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
    
#citanje podataka
df = pd.read_csv('./tabela.csv')
#X skup na osnovu koga klasifikujemp
X = df.loc[:, df.columns != 'class']
#y skup vrednosti koji oznacavaju klasu
y = df[['class']]

#podela na test i trening skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#clf menjamo u zavisnosti od metode koju koristimo
#u nastavku rada su navedeni svi pozivi funkcija koje smo koristili
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
clf.fit(X_train, y_train.values.ravel())
    
#vrsimo predikciju
y_test_predicted = clf.predict(X_test)
y_train_predicted = clf.predict(X_train)
    
#racunamo rezultate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
train_conf = sklearn.metrics.confusion_matrix(y_train, y_train_predicted)
test_conf = sklearn.metrics.confusion_matrix(y_test, y_test_predicted)  
    
#stampanje
print('Preciznost trening skupa: {}'.format(train_acc))
print("Matrica konfuzije:\n{}".format(train_conf))
print('Preciznost test skupa: {}'.format(test_acc))
print("Matrica konfuzije:\n{}".format(test_conf))

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:14:19 2019

@author: ertasnihan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
veriler = pd.read_csv('maaslar.csv')

#dataframe dilimleme(slice)
x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

#Numpy array dönüşümü
X= x.values
Y= y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y, color='red')
plt.plot(x,lin_reg.predict(x), color='blue')
plt.show()

#polinomal regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=2) # degree 4 olunca daha iyi tahmin etti bu veri için böyle
x_poly =poly_reg.fit_transform(X)
print(x_poly)
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly, y)

#görselleştirme
plt.scatter(X, Y, color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue') #tahmin değerinin önce polynomial yap
plt.show()

#tahminler
print(lin_reg.predict(np.array([[11]])))
print(lin_reg.predict(np.array([[6.6]])))
 
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[11]]))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[6.6]]))))


#öznitelik ölçekleme(standartlaşma)
from sklearn.preprocessing import StandardScaler
sc1= StandardScaler()
x_olcekli= sc1.fit_transform(X)

sc2= StandardScaler()
y_olcekli= sc2.fit_transform(Y)


#SVR regression
from sklearn.svm import SVR
svr_reg =SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)


plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue') #tahmin değerinin önce polynomial yap
plt.show()

print(svr_reg.predict(np.array([[11]])))







    
    










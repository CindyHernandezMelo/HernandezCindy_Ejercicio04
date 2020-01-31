import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 

%matplotlib inline
data = pd.read_csv('Cars93.csv')
Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.3)
from itertools import combinations 
plt.figure()
regresion = sklearn.linear_model.LinearRegression()
R_2 = np.zeros([np.shape(X)[1],462])
for i in range(1,np.shape(X)[1]):
    comb = np.asarray(list(combinations(np.arange(np.shape(X)[1]), i)))
    for j in range(np.shape(comb)[0]):
        regresion.fit(X_train[:,comb[j,:]], Y_train)
        R_2[i,j] = regresion.score(X_test[:,comb[j,:]], Y_test)
        plt.scatter(i,regresion.score(X_test[:,comb[j,:]], Y_test))
        
ind = np.unravel_index(np.argmax(R_2, axis=None), R_2.shape) 
    
plt.xlabel('Numero de variables')
plt.ylabel('R$^2$')
comb = np.asarray(list(combinations(np.arange(np.shape(X)[1]), ind[0])))
ind_betas = comb[162]
for k in ind_betas:
    print(columns[k])
    
N = 30
alphas = np.logspace(0.1,1,N)
MSE = np.zeros(N)
from sklearn.metrics import mean_squared_error
for a in range(30):
    lasso = sklearn.linear_model.Lasso(alpha=alphas[a])
    lasso.fit(X_train, Y_train)
    MSE
    
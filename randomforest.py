import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('Salaries.csv')
print(data)
df=pd.DataFrame(data)
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print("X=",X)
print("y=",y)
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X, y)
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values


X_grid = np.arange(min(X), max(X), 0.01) 
  
# reshape for reshaping the data into a len(X_grid)*1 array, 
# i.e. to make a column out of the X_grid value                  
X_grid = X_grid.reshape((len(X_grid), 1))
  
# Scatter plot for original data
plt.scatter(X, y, color = 'blue')  
  
# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid), 
         color = 'green') 
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

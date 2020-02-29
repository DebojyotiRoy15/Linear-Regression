import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

data = pd.read_excel('slr12 (1).xls')

X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, Y_train)

plt.scatter(X_train, Y_train)
plt.plot(X_train, LR.predict(X_train), color='red')
plt.xlabel('Annual franchise fee $(1000)')
plt.ylabel('Startup cost $(1000) for a pizza franchise')
plt.title('Pizza franchise')

plt.scatter(X_test, Y_test)
plt.plot(X_train, LR.predict(X_train), color='red')
plt.xlabel('Annual franchise fee $(1000)')
plt.ylabel('Startup cost $(1000) for a pizza franchise')
plt.title('Pizza franchise')

LR.intercept_ //537.5545749806037
LR.coef_ // array([0.65024136])

//The equation is y = 0.65024136*x + 537.5545749806037

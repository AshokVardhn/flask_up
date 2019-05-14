import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,r2_score, mean_squared_error

x= np.random.randn(10000)
# y = np.power(np.sin(x),2) + 2*np.power(x,2)+5
y = np.power(np.sin(x),2) + 2*x+5
print(x,y)

plt.scatter(y,x,color='r')
X_train, X_test, y_train, y_test  = train_test_split(x.reshape(-1,1),y.reshape(-1,1),test_size=0.3,random_state=4)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
plt.scatter(y_pred,X_test,color='b')

print("r2 score: ",r2_score(y_pred,y_test))
print("MSE : ",mean_squared_error(y_pred,y_test))
# print("Confusion Matrix",confusion_matrix(y_test,y_pred))
print("Coefficients & Intercepts are :",lr.coef_,lr.intercept_)

plt.show()

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin


# rs = np.random.RandomState(1)
x = 10*np.random.rand(50).reshape(-1,1)
y = 2*x + 10 + np.random.rand(50).reshape(-1,1)

#Representing Data
# plt.scatter(x,y,color='r')

linear = LinearRegression()
x_test = np.random.randn(50).reshape(-1,1)
linear.fit(x,y)
linear_ypred = linear.predict(x_test)
# plt.scatter(x_test,linear_ypred,color='b')



'''Polynomial Features'''

# poly_x = np.array([2,3,4])
# poly = PolynomialFeatures(3,include_bias=False)
# # poly = PolynomialFeatures(4,include_bias=False)
# print(poly.fit_transform(poly_x.reshape((-1,1))))
# # print(poly.fit_transform(poly_x[:,None]))

'''Pipelining'''
rng = np.random.RandomState(1)
poly_x = 10 * rng.rand(100).reshape((-1,1))
poly_y = np.sin(poly_x) + 0.1 * rng.randn(100).reshape(-1,1)

pipeline = make_pipeline(PolynomialFeatures(7),LinearRegression())
pipeline.fit(poly_x,poly_y)
pipe_ypred = pipeline.predict(poly_x)
plt.scatter(poly_x,poly_y,color='b')
plt.scatter(poly_x,pipe_ypred,color='g')


#creating a custom Gaussian Basis function

# from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


np.random.seed(10)
gauss_x = 10*np.random.rand(50)
# gauss_y = np.sin(gauss_x) + 0.1 *np.power(gauss_x,15) + np.power(gauss_x,14)
gauss_y = 5*np.sin(gauss_x) + 0.1 *np.random.rand(50)
gauss_x = gauss_x.reshape(-1,1)
# gauss_y = gauss_y.reshape(-1,1)

gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(gauss_x, gauss_y)
gauss_xtest = 10*np.random.rand(50).reshape(-1,1)
gauss_ypred = gauss_model.predict(gauss_xtest)

# plt.scatter(gauss_x, gauss_y,color='black')
# plt.scatter(gauss_xtest, gauss_ypred,color='orange')
# plt.plot(gauss_xtest, gauss_ypred)
# plt.xlim(0, 10);
# plt.ylim(-5, 5);
# print(gauss_model.steps)
# print(gauss_model.steps[0])
# print(gauss_model.steps[0][1].centers_)
# print(gauss_model.steps[1][1].coef_)


# def basis_plot(model, title=None):
#     fig, ax = plt.subplots(2, sharex=True)
#     model.fit(x[:, np.newaxis], y)
#     ax[0].scatter(x, y)
#     ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
#     ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
#
#     if title:
#         ax[0].set_title(title)
#
#     ax[1].plot(model.steps[0][1].centers_,
#                model.steps[1][1].coef_)
#     ax[1].set(xlabel='basis location',
#               ylabel='coefficient',
#               xlim=(0, 10))
#
#
# # basis_plot(gauss_model)



# from sklearn.linear_model import Ridge
rng = np.random.RandomState(1)
ridge_x = 10 * rng.rand(100).reshape((-1,1))
ridge_y  = np.sin(ridge_x) + 0.1 * rng.randn(100).reshape(-1,1)

ridge_model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=0.1))
ridge_model.fit(ridge_x,ridge_y)
ridge_xtest = 10*np.random.rand(100).reshape(-1,1)
ridge_ypred = ridge_model.predict(ridge_xtest)

plt.scatter(ridge_x, ridge_y,color='yellow')
plt.scatter(ridge_xtest, ridge_ypred,color='red')

'''Extended Plotting using subplots'''
plots = np.logspace(-6,6,4)
fig,ax = plt.subplots(len(plots),sharex=True)
for index,i in enumerate(plots):
    ridge_x = 10 * rng.rand(100).reshape((-1, 1))
    ridge_y = np.sin(ridge_x) + 0.1 * rng.randn(100).reshape(-1, 1)
    ridge_model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=i))
    ridge_model.fit(ridge_x, ridge_y)
    ridge_xtest = 10 * np.random.rand(100).reshape(-1, 1)
    ridge_ypred = ridge_model.predict(ridge_xtest)

    ax[index].scatter(ridge_x,ridge_y)
    ax[index].scatter(ridge_xtest,ridge_ypred)
    ax[index].set(xlabel='x',ylabel='y',title=i)
    print("Polynomial With Ridge Regression: with alpha = ",i," : ", r2_score(ridge_y, ridge_ypred),
          mean_squared_error(ridge_y, ridge_ypred))

print("Model slope      Model Intercept     R2 Score        MeanSqared Error")
print(linear.coef_,linear.intercept_,r2_score(x_test,linear_ypred),mean_squared_error(x_test,linear_ypred))
print("Gaussian with Linear Regression  :",r2_score(gauss_y,gauss_ypred),mean_squared_error(gauss_y,gauss_ypred))
print("Polynomial With Linear Regression:    ",r2_score(poly_y,pipe_ypred),mean_squared_error(poly_y,pipe_ypred))
print("Polynomial With Ridge Regression:    ",r2_score(ridge_y,ridge_ypred),mean_squared_error(ridge_y,ridge_ypred))






plt.show()
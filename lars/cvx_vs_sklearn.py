import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()

x = diabetes.data[:, np.newaxis, 2]
y = diabetes.target.reshape(-1, 1)

reg = linear_model.LinearRegression()
reg.fit(x, y)

import cvxpy as cvx

b = cvx.Variable(1,1)
obj = cvx.Minimize(cvx.sum_squares(b*x-y))
cvx.Problem(obj).solve()

print("Sklearn: ", reg.coef_[0, 0])
print("cvxpy: ", b.value)
print("Same? ", reg.coef_[0, 0] == b.value)

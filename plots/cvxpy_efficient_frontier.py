import numpy as np
from cvxpy import *
from matplotlib import pyplot as plt

def markowitz_optimizer(data):
	# w^T \Sigma w - \gamma \bar{r}^T w
	n, d = data.shape
	w = Variable(n)
	gamma = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	ret = rbar*w
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	prob = Problem(Minimize(risk - gamma * ret), [sum_entries(w) == 1, w >= 0])
	SAMPLES = 100
	risk_data = np.zeros(SAMPLES)
	ret_data = np.zeros(SAMPLES)
	gamma_vals = np.logspace(-2, 3, num=SAMPLES)
	for i in range(SAMPLES):
		gamma.value = gamma_vals[i]
		prob.solve()
		risk_data[i] = sqrt(risk).value
		ret_data[i] = ret.value
	return risk_data, ret_data

def lasso_optimizer(data, tau):
	# ?
	# w^T \Sigma w - \gamma \bar{r}^T w + tau||w||_1
	n, d = data.shape
	w = Variable(n)
	gamma = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	ret = rbar*w
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	prob = Problem(Minimize(risk - gamma * ret + tau), [sum_entries(w) == 1, w >= 0])
	SAMPLES = 100
	risk_data = np.zeros(SAMPLES)
	ret_data = np.zeros(SAMPLES)
	gamma_vals = np.logspace(-2, 3, num=SAMPLES)
	for i in range(SAMPLES):
		gamma.value = gamma_vals[i]
		prob.solve()
		risk_data[i] = sqrt(risk).value
		ret_data[i] = ret.value
	return risk_data, ret_data

data = np.load('example_returns.ndarray')

# Markowitz
risk_data, ret_data = markowitz_optimizer(data)
plt.plot(risk_data, ret_data)

# lasso
risk_lasso, ret_lasso = lasso_optimizer(data, 2)
plt.plot(risk_lasso, ret_lasso)

plt.show()
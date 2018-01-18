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

def markowitz_optimizer_no_lagrange(data):
	# w^T \Sigma w - \gamma \bar{r}^T w
	n, d = data.shape
	w = Variable(n)
	mu0 = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	objective = Minimize(risk)
	constraints = [rbar*w == mu0, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	SAMPLES = 100
	risk_data = np.zeros(SAMPLES)
	ret_data = np.zeros(SAMPLES)
	portfolios = np.zeros((SAMPLES, n))
	mu0_values = np.linspace(0.013, 0.05, 100)
	for i in range(SAMPLES):
		mu0.value = mu0_values[i]
		prob.solve()
		risk_data[i] = sqrt(risk).value
		ret_data[i] = rbar*w.value
		portfolios[i] = np.array(w.value).flatten()
	return risk_data, ret_data, portfolios

def lasso_optimizer(data, tau):
	# ?
	# w^T \Sigma w - \gamma \bar{r}^T w + tau||w||_1
	n, d = data.shape
	w = Variable(n)
	mu0 = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	objective = Minimize(risk + tau*norm(w,1))
	constraints = [rbar*w == mu0, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	SAMPLES = 100
	risk_data = np.zeros(SAMPLES)
	ret_data = np.zeros(SAMPLES)
	portfolios = np.zeros((SAMPLES, n))
	mu0_values = np.linspace(0.013, 0.05, 100)
	for i in range(SAMPLES):
		mu0.value = mu0_values[i]
		prob.solve()
		risk_data[i] = sqrt(risk).value
		ret_data[i] = rbar*w.value
		portfolios[i] = np.array(w.value).flatten()
	return risk_data, ret_data, portfolios


from build_example_data import build_example_ml_return_data
data, test = build_example_ml_return_data()
#data = np.load('example_returns.ndarray')

# Markowitz
risk_data, ret_data = markowitz_optimizer(data)
plt.plot(risk_data, ret_data, 'o', markersize=3, markeredgecolor='black')

# lasso
#risk_lasso, ret_lasso = lasso_optimizer(data, 2)
#plt.plot(risk_lasso, ret_lasso)

plt.show()
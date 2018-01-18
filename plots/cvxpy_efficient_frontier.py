import numpy as np
from cvxpy import *
from matplotlib import pyplot as plt

def random_weights(n):
	w = np.random.rand(n)
	return w / sum(w)

def random_portfolios(N, n):
	portfolios = []
	for i in range(N):
		portfolios.append(random_weights(n))
	return portfolios
	
def markowitz_optimizer_lagrange(data):
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

def markowitz_optimizer(data):
	n, d = data.shape
	w = Variable(n)
	mu0 = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	objective = Minimize(risk)
	constraints = [rbar*w == mu0, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	portfolios = []
	i = 0
	while i < 0.2:
		mu0.value = i
		prob.solve()
		if type(w.value) != type(None):
			portfolios.append(np.array(w.value).flatten())
		i += 0.0001
	return portfolios

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

def fit(data, portfolios):
	rbar = np.mean(data, axis=1)
	Sigma = np.cov(data)
	SAMPLES = len(portfolios)
	risks = np.zeros(SAMPLES)
	returns = np.zeros(SAMPLES)
	for i in range(SAMPLES):
		w = portfolios[i]
		returns[i] = np.dot(rbar, w)
		risks[i] = np.sqrt(np.dot(w, np.dot(Sigma, w)))
	return risks, returns

def plot_ml(data, test):
	###########################################################################################################################
	# figure init
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Trained frontier vs actual portfolio performance curve')
	ax.set_xlabel('Risk (standard deviation)')
	ax.set_ylabel('Return')
	#
	###########################################################################################################################
	# Random portfolios
	n, d = data.shape
	
	random_portfolios_ = random_portfolios(1000, n)
	random_train_risk, random_train_return = fit(data, random_portfolios_)
	random_test_risk, random_test_return = fit(test, random_portfolios_)
	
	ax.plot(random_train_risk, random_train_return, 'o', markersize=3, markeredgecolor='black', label='Random train portfolios')
	ax.plot(random_test_risk, random_test_return, 'o', markersize=3, markeredgecolor='black', label='Random test portfolios')
	#
	###########################################################################################################################
	# Naive portfolios 
	naive_portfolio = np.repeat(1/n, n)
	naive_test_risk, naive_test_return = fit(test, [naive_portfolio])
	naive_train_risk, naive_train_return = fit(data, [naive_portfolio])
	
	ax.plot(naive_train_risk, naive_train_return, 'o', markersize=5, markeredgecolor='black', label='Naive train portfolio')
	ax.plot(naive_test_risk, naive_test_return, 'o', markersize=5, markeredgecolor='black', label='Naive test portfolio')
	#
	###########################################################################################################################	
	# efficient frontiers
	portfolios = markowitz_optimizer(data)
	portfolios_test = markowitz_optimizer(test)
	
	risks, returns = fit(data, portfolios)
	risks_test, returns_test = fit(test, portfolios)
	risks_test_, returns_test_ = fit(test, portfolios_test)
	
	ax.plot(risks, returns, '-', markersize=3, markeredgecolor='black', label='Trained efficient frontier')
	ax.plot(risks_test_, returns_test_, '-', markersize=3, markeredgecolor='black', label='Test efficitent frontier')
	ax.plot(risks_test, returns_test, '-', markersize=3, markeredgecolor='black', label='Actual portfolio performance curve')
	#
	########################################################################################################################
	# Display and store image
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()
	#
	###########################################################################################################################

from build_example_data import build_example_ml_return_data
from build_example_data import build_example_return_data

# Markowitz
#risk_data, ret_data, portfolios = markowitz_optimizer_no_lagrange(data)
#print(portfolios)
#plt.plot(risk_data, ret_data, 'o', markersize=3, markeredgecolor='black')

# lasso
#risk_lasso, ret_lasso = lasso_optimizer(data, 2)
#plt.plot(risk_lasso, ret_lasso)

data, test = build_example_ml_return_data()
#data = build_example_return_data()
plot_ml(data, test)

plt.show()
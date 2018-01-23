import numpy as np
from cvxpy import *
from matplotlib import pyplot as plt
from matplotlib import colors

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

def plot_efficient(data, ax, text, color=None):
	portfolios = markowitz_optimizer(data)
	risks, returns = fit(data, portfolios)
	if color:
		ax.plot(risks, returns, color, label=text)
	else:
		ax.plot(risks, returns, color, label=text)
	return portfolios

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
	portfolios = plot_efficient(data, ax, 'Trained efficient frontier')
	test_portfolios = plot_efficient(test, ax, 'Test efficitent frontier')
	risks_test, returns_test = fit(test, portfolios)
	ax.plot(risks_test, returns_test, '-', markersize=3, markeredgecolor='black', label='Actual portfolio performance curve')
	#
	########################################################################################################################
	# Display and store image
	save_image(plt, fig, ax)
	#
	###########################################################################################################################

def split_data(data, B):
	return [data[b] for b in B]

def portfolio_risk_return(data, B):
	n = len(B)
	w = np.repeat(1/n, n)
	d = split_data(data, B)
	rbar = np.mean(d, axis=1)
	sigma = np.sqrt(np.dot(w, np.dot(np.cov(d), w)))
	mu = np.dot(rbar, w)
	return sigma, mu

def depth_first_indexes(data, k, minret, maxrisk):
	A = []
	N, D = data.shape
	S = [[i] for i in range(N)]
	while S:
		B = S.pop(0)
		sigma, mu = portfolio_risk_return(data, B)
		print(B, sigma, mu)
		if mu >= minret and sigma <= maxrisk:
			A.append(B)
		if(len(B) < k):
			i = B[-1]
			for j in range(N-1, i, -1):
				C = B.copy()
				C.append(j)
				S.insert(0, C)
	return A

def plot_boundaries(ax, mu0, s0):
	# Boundaries
	ax.axhline(y=mu0, color='#1f77b4', label=r'$\mu_0$')
	ax.axvline(x=s0, color='#ff7f0e', label=r'$\sigma_0$')

def plot_k_portfolios(data, k, minret, maxrisk):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('K portfolios')
	ax.set_xlabel('Risk (standard deviation)')
	ax.set_ylabel('Return')


	portfolios = plot_efficient(data, ax, 'Trained efficient frontier', 'k')
	plot_boundaries(ax, minret, maxrisk)

	indexes = depth_first_indexes(data, k, minret, maxrisk)
	colors = get_colors(k)
	for i in indexes:
		subdata = split_data(data, i)
		n = len(subdata)
		w = np.repeat(1/n, n)
		risk, ret = fit(subdata, [w])
		label = 'Naive ' + str(n) + '-portfolios'
		ax.plot(risk, ret, color=colors[n-1], linestyle=' ', marker='o', markeredgecolor='black', markersize=4, label=label)
		
	n, d = data.shape
	w = np.repeat(1/n, n)
	sigma, mu = fit(data, [w])
	ax.plot(sigma, mu, 'y*', markeredgecolor='black', markersize=6, label='Naive portfolio')

	save_image(plt, fig, ax)

def get_colors(n):
	c = colors.get_named_colors_mapping()
	c = list(c.values())
	c = c[:-8]
	return np.random.choice(c, n)

def save_image(plt, fig, ax):
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()

#from build_example_data import build_example_ml_return_data
#from build_example_data import build_example_return_data
from build_example_data import *

#data, test = build_example_ml_return_data()
#data = build_example_return_data()
data = np.load('DJ30.ndarray')


# PLOTS
#plot_ml(data, test)
#plot_k_portfolios(data, 3, .0001, .2)
#plot_k_portfolios(data, 5, .00001, .09)
plot_k_portfolios(data, 10, .0004, .02)
#plot_k_portfolios(data, 5, .00043, .018)


plt.show()
import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from plot_helpers import create_fig
from plot_helpers import save_image

from build_example_data import sp_data_remove_outliers

from randomized_k_portfolios import uniform
from randomized_k_portfolios import return_weighted
from randomized_k_portfolios import markowitz_randomized

from solvers import cvxopt_solve
from solvers import cvxopt_solve_max_risk
from solvers import cvxopt_fit
from solvers import cvxpy_fit

def eff_front(data, k, N, max_lambd, mean, cov, ax):	
	portfolios = cvxopt_solve_max_risk(mean, cov, N, max_lambd)
	returns, risks = cvxopt_fit(mean, cov, portfolios)

	ax.plot(risks, returns, label='Efficient frontier')


def plot_uniform(data, k, precision, max_lambd):
	mean = np.mean(data, axis=1)
	cov = np.cov(data)
	title = 'Uniformly random ' + str(k) + '-portfolios'
	fig, ax = create_fig(title, 'Risk', 'Return')

	eff_front(data, k, precision, max_lambd, mean, cov, ax)

	n = len(data)
	for i in range(1000):
		uni_port = uniform(n, k)
		ri, ret = cvxpy_fit(mean, cov, [uni_port])
		ax.plot(ri, ret, 'ro', markersize=3, markeredgecolor='k', label='Randomized portfolios')
	save_image(fig, ax)

def plot_uniform_and_naive(data, k, precision, max_lambd):
	mean = np.mean(data, axis=1)
	cov = np.cov(data)
	title = 'Efficient frontier and randomized portfolios'
	fig, ax = create_fig(title, 'Standard deviation', 'Expectation')

	eff_front(data, k, precision, max_lambd, mean, cov, ax)

	n, d = data.shape
	for i in range(1000):
		uni_port = uniform(n, k)
		ri, ret = cvxpy_fit(mean, cov, uni_port)
		ax.plot(ri, ret, 'ro', markersize=3, markeredgecolor='k', label='Randomized ' + str(k) + '-portfolios')
	naive_port = np.repeat(1/n, n)
	naive_ri, naive_ret = cvxpy_fit(mean, cov, [naive_port])
	ax.plot(naive_ri, naive_ret, 'bo', markersize=3, markeredgecolor='k', label='Naive portfolio')
	save_image(fig, ax)


def plot_return_weighted(data, k, precision, max_lambd):
	mean = np.mean(data, axis=1)
	cov = np.cov(data)
	title = 'Random return weighted ' + str(k) + '-portfolios'
	fig, ax = create_fig(title, 'Risk', 'Return')

	eff_front(data, k, precision, max_lambd, mean, cov, ax)

	n = len(data)
	for i in range(1000):
		re_port = return_weighted(n, k, mean)
		ri, ret = cvxpy_fit(mean, cov, [re_port])
		ax.plot(ri, ret, 'ro', markersize=3, markeredgecolor='k', label='Randomized portfolios')
	save_image(fig, ax)

def plot_markowitz_randomized(data, k, size, precision, max_lambd, mu0, sigma0):
	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	title = 'Markowitz randomized ' + str(k) + '-portfolios ' + r'$\sigma_0 = $' + str(sigma0) + r' $\mu_0 = $' + str(mu0)
	fig, ax = create_fig(title, 'Risk', 'Return')

	eff_front(data, k, precision, max_lambd, mean, cov, ax)
	ax.axvline(x=sigma0, color='green', label=r'$\sigma_0$')
	ax.axhline(y=mu0, color='brown', label=r'$\mu_0$')

	n = len(data)
	markowitz, mw_port = markowitz_randomized(n, k, size, mean, cov, max_lambd, mu0, sigma0)
	mri, mret = cvxpy_fit(mean, cov, [markowitz])
	ax.plot(mri, mret, 'ko', markersize=3, label='Markowitz portfolio distribution')
	for w in mw_port:
		ri, ret = cvxpy_fit(mean, cov, [w])
		if ri <= sigma0 and ret >= mu0:
			ax.plot(ri, ret, 'ro', markersize=3, markeredgecolor='k', label='Randomized portfolios')

	save_image(fig, ax)


def plot_new_window(risks, returns, s0, m0, alpha):
	inside_risks = []
	inside_returns = []
	outside_returns = []
	outside_risks = []
	for ri, re in zip(risks, returns):
		if s0 > ri and m0 < re:
			inside_risks.append(ri)
			inside_returns.append(re)
		else:
			outside_returns.append(re)
			outside_risks.append(ri)		
	plt.plot(inside_risks, inside_returns, 'bo')
	plt.plot(outside_risks, outside_returns, 'ro')
	plt.axhline(m0)
	plt.axvline(s0)
	plt.show()


data = np.load('sp332.ndarray')
#data = sp_data_remove_outliers()
#plot_uniform(data, 50, 100, .2)
#plot_return_weighted(data, 50, 100, .2)
#plot_markowitz_randomized(data, 10, 1000, 10, 0.2, .001, .018)
plot_uniform_and_naive(data, 50, 100, .2)

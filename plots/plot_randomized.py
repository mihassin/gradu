import numpy as np

from matplotlib import pyplot as plt

from plot_helpers import create_fig
from plot_helpers import save_image

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


data = np.load('sp332.ndarray')
#plot_uniform(data, 50, 100, 10)
#plot_return_weighted(data, 50, 100, 10)
plot_markowitz_randomized(data, 50, 1000, 10, 0.2, .00125, .014)
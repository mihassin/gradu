import numpy as np

from matplotlib import pyplot as plt

from plot_helpers import create_fig
from plot_helpers import save_image

# solvers
from solvers import cvxopt_solve
from solvers import cvxopt_solve_max_risk
from solvers import cvxopt_fit
from solvers import cvxpy_fit
from solvers import lasso_attempt_1
from solvers import normalized_unconstrained_lasso

# data
from build_example_data import iid_n01_data
from build_example_data import generate_multinormal
from build_example_data import sample_multinormal
from build_example_data import build_example_ml_return_data


def plot_efficient_frontier(data, precision):
	# Efficient frontier
	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	fig, ax =  create_fig('Efficient frontier', 'Risk', 'Return')

	portfolios = cvxopt_solve(mean, cov, precision)
	returns, risks = cvxopt_fit(mean, cov, portfolios)

	ax.plot(risks, returns, label='Efficient frontier')

	save_image(fig, ax)

def plot_efficient_frontier_maxrisk(data, precision, maxrisk):
	# Efficient frontier
	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	fig, ax = create_fig('Efficient frontier', 'Risk', 'Return')

	portfolios = cvxopt_solve_max_risk(mean, cov, precision, maxrisk)
	returns, risks = cvxopt_fit(mean, cov, portfolios)

	ax.plot(risks, returns, label='Efficient frontier')

	save_image(fig, ax)

def test_lasso(data, precision, maxrisk, tau):
	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	fig, ax = create_fig('Lasso ' + r'$\tau$ = ' + str(tau), 'Risk', 'Return')

	portfolios = cvxopt_solve_max_risk(mean, cov, precision, maxrisk)
	returns, risks = cvxopt_fit(mean, cov, portfolios)

	ax.plot(risks, returns, label='Efficient frontier')

	p_test = lasso_attempt_1(mean, cov, precision, maxrisk, tau)
	r_test, ri_test = cvxopt_fit(mean, cov, p_test)
	ax.plot(ri_test, r_test, label='Lasso frontier')
	save_image(fig, ax)

def plot_normalized_unconstrained_lasso(data, lambd):
	cov = np.cov(data)
	mean = np.mean(data, axis=1)

	fig, ax = create_fig('Mean-standard deviation', 'Risk', 'Return')

	portfolios = cvxopt_solve(mean, cov, 100)
	returns, risks = cvxopt_fit(mean, cov, portfolios)
	ax.plot(risks, returns, label='Efficient frontier')

	alphas, norm_coefs = normalized_unconstrained_lasso(data, lambd)
	risks, returns = cvxpy_fit(mean, cov, norm_coefs)
	for i in range(len(risks)):
		ax.plot(risks[i], returns[i], 'o', label=r'$\tau$ = ' + str(np.round(np.sort(alphas)[i], decimals=8)))
	save_image(fig, ax, legend=1)


data = np.load('DJ30.ndarray')
#data = np.load('sp332.ndarray')
#plot_efficient_frontier(data, 100)
#plot_efficient_frontier_maxrisk(data, 100, 0.5)
#test_lasso(data, 100, 0.5, 100)
plot_normalized_unconstrained_lasso(data, -100)
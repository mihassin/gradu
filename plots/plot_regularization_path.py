import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors

from plot_helpers import create_fig
from plot_helpers import save_image

# solvers
from solvers import lasso_single_attempt_1
from solvers import lasso_single_attempt_2
from solvers import cvxpy_fit
from solvers import cvxopt_solve
from solvers import cvxopt_fit
from solvers import normalized_unconstrained_lasso

# data
from build_example_data import iid_n01_data
from build_example_data import generate_multinormal
from build_example_data import sample_multinormal
from build_example_data import build_example_ml_return_data

def plot_regularization_path(data, rho):
	fig, ax = create_fig('Regularization path ' + r'$\rho = $' + str(rho), r'$\tau$', 'Weight')

	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	taus = np.arange(0,11000, step=1000)
	portfolios = np.array([[p for p in lasso_single_attempt_1(mean, cov, rho, int(taus[0]))]])
	for tau in taus[1:]:
		portfolios = np.append(portfolios, [[p for p in lasso_single_attempt_1(mean, cov, rho, int(tau))]], axis=0)
	for p in portfolios.T:
		ax.plot(taus, p)
	save_image(fig, ax, legend=False)

def plot_normalized_unconstrained_lasso_path(a, n, mu0, fn):
	fig, ax = create_fig('Regularization path ' + r'$\mu_0 = $' + str(mu0), r'$\tau$', 'Weight')
	ax.plot(a, n)
	save_image(fig, ax, plt_name=fn, legend=False)

data = np.load('DJ30.ndarray')
#data = np.load('sp332.ndarray')
#plot_regularization_path(data, 0.0006)
lambd = -100
alphas, norm_coefs = normalized_unconstrained_lasso(data, lambd)
plot_normalized_unconstrained_lasso_path(alphas, norm_coefs, lambd, 'norm_reg_path_1.png')

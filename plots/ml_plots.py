import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors

# solvers
from solvers import solve_cvxopt
from solvers import fit_cvxopt

# data
from build_example_data import iid_n01_data
from build_example_data import generate_multinormal
from build_example_data import sample_multinormal
from build_example_data import build_example_ml_return_data

def ml_iid_plot(n, train_t, test_t):
	# Data
	train = iid_n01_data(n, train_t)
	test =  iid_n01_data(n, test_t)
	ml_plot(train, test)

def ml_multinormal(n, train_t, test_t):
	mu, sigma = generate_multinormal(n)
	train = sample_multinormal(mu, sigma, train_t)
	test = sample_multinormal(mu, sigma, test_t)
	ml_plot(train, test)

def ml_plot(train, test):
	# stats
	mean_train = np.mean(train, axis=1)
	cov_train = np.cov(train)
	mean_test = np.mean(test, axis=1)
	cov_test = np.cov(test)
	# solve
	portfolios = solve_cvxopt(mean_train, cov_train, train.shape[1])
	# fit
	returns_train, risks_train = fit_cvxopt(mean_train, cov_train, portfolios)
	returns_test, risks_test = fit_cvxopt(mean_test, cov_test, portfolios)
	# building the image
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Train frontier vs test frontier')
	ax.set_xlabel('Risk')
	ax.set_ylabel('Return')
	ax.plot(risks_train, returns_train, '-', markersize=3, markeredgecolor='black', label='Trained efficient frontier')
	ax.plot(risks_test, returns_test, '-', markersize=3, markeredgecolor='black', label='Actual portfolio performance curve')
	# presentation
	save_image(plt, fig, ax)

def ml_actual_vs_estimates(n, t, test_samples):
	mu, sigma = generate_multinormal(n)
	portfolios = solve_cvxopt(mu, sigma, t)
	returns_actual, risks_actual = fit_cvxopt(mu, sigma, portfolios)
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Estimates n = ' + str(test_samples))
	ax.set_xlabel('Risk')
	ax.set_ylabel('Return')
	rets = []
	rsks = []
	for i in range(test_samples):
		test = sample_multinormal(mu, sigma, t)
		mean_test = np.mean(test, axis=1)
		cov_test = np.cov(test)
		returns_test, risks_test = fit_cvxopt(mean_test, cov_test, portfolios)
		rets.append(returns_test)
		rsks.append(risks_test)
		ax.plot(risks_test, returns_test, 'r-', markersize=3, markeredgecolor='black', label='Estimate frontiers')
	rets = np.array(rets)
	rsks = np.array(rsks)
	ax.plot(np.mean(rsks, axis=0), np.mean(rets, axis=0), 'k-', markersize=5, markeredgecolor='black', label='Average frontier')
	ax.plot(risks_actual, returns_actual, 'g-', markersize=5, markeredgecolor='black', label='Actual frontier')
	save_image(plt, fig, ax)

def save_image(plt, fig, ax):
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()

# DATA
#ml_iid_plot(4, 1000, 1000)
#ml_multinormal(4, 1000, 1000)
ml_actual_vs_estimates(4, 1000, 1000)
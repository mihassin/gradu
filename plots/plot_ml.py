import numpy as np

from matplotlib import pyplot as plt

# solvers
from solvers import cvxopt_solve
from solvers import cvxopt_fit

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
	portfolios = cvxopt_solve(mean_train, cov_train, train.shape[1])
	# fit
	returns_train, risks_train = cvxopt_fit(mean_train, cov_train, portfolios)
	returns_test, risks_test = cvxopt_fit(mean_test, cov_test, portfolios)
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
	portfolios = cvxopt_solve(mu, sigma, 100)
	returns_actual, risks_actual = cvxopt_fit(mu, sigma, portfolios)
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
		returns_test, risks_test = cvxopt_fit(mean_test, cov_test, portfolios)
		rets.append(returns_test)
		rsks.append(risks_test)
		ax.plot(risks_test, returns_test, 'r-', markersize=3, markeredgecolor='black', label='Estimate frontiers')
	rets = np.array(rets)
	rsks = np.array(rsks)
	ax.plot(np.mean(rsks, axis=0), np.mean(rets, axis=0), 'k-', markersize=5, markeredgecolor='black', label='Average frontier')
	ax.plot(risks_actual, returns_actual, 'g-', markersize=5, markeredgecolor='black', label='Actual frontier')
	save_image(plt, fig, ax)

def ml_train_vs_test_helper(mu, sigma, n, t, test_samples):
	train = sample_multinormal(mu, sigma, t)
	mean = np.mean(train, axis=1)
	cov = np.cov(train)
	portfolios = cvxopt_solve(mean, cov, 100)
	returns_train, risks_train = cvxopt_fit(mean, cov, portfolios)
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Test curves n = ' + str(test_samples))
	ax.set_xlabel('Risk')
	ax.set_ylabel('Return')
	rets = []
	rsks = []
	for i in range(test_samples):
		test = sample_multinormal(mu, sigma, t)
		mean_test = np.mean(test, axis=1)
		cov_test = np.cov(test)
		returns_test, risks_test = cvxopt_fit(mean_test, cov_test, portfolios)
		rets.append(returns_test)
		rsks.append(risks_test)
		ax.plot(risks_test, returns_test, 'r-', markersize=3, markeredgecolor='black', label='Test frontiers')
	rets = np.array(rets)
	rsks = np.array(rsks)
	ax.plot(np.mean(rsks, axis=0), np.mean(rets, axis=0), 'k-', markersize=5, markeredgecolor='black', label='Average test frontier')
	ax.plot(risks_train, returns_train, 'g-', markersize=5, markeredgecolor='black', label='Train frontier')
	return fig, ax

def ml_train_vs_test(n, t, test_samples):
	mu, sigma = generate_multinormal(n)
	fig, ax = ml_train_vs_test_helper(mu, sigma, n, t, test_samples)
	save_image(fig, ax)

def ml_train_vs_test_vs_actual(n, t, test_samples):
	mu, sigma = generate_multinormal(n)
	fig, ax = ml_train_vs_test_helper(mu, sigma, n, t, test_samples)
	portfolios = cvxopt_solve(mu, sigma, 100)
	returns_actual, risks_actual = cvxopt_fit(mu, sigma, portfolios)
	ax.plot(risks_actual, returns_actual, 'b-', markersize=5, markeredgecolor='black', label='Actual frontier')
	save_image(fig, ax)

def save_image(fig, ax):
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()

# DATA
#ml_iid_plot(4, 1000, 1000)
#ml_multinormal(4, 1000, 1000)
#ml_actual_vs_estimates(300, 5000, 10)
#ml_train_vs_test(100, 1000, 1000)
ml_train_vs_test_vs_actual(100, 1000, 1000)
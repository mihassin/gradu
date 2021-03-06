import numpy as np
from matplotlib import pyplot as plt
import cvxopt as opt
from cvxopt import solvers, blas
from itertools import combinations
	
def random_weights(n):
	w = np.random.rand(n)
	return w / sum(w)

def random_portfolios(returns):
	p = np.asmatrix(np.mean(returns, axis=1))
	w = np.asmatrix(random_weights(returns.shape[0]))
	C = np.asmatrix(np.cov(returns))
	mu = w * p.T
	sigma = np.sqrt(w*C*w.T)
	return mu, sigma

def naive_portfolio(returns):
	n, d = returns.shape
	p = np.asmatrix(np.mean(returns, axis=1))
	w = np.asmatrix(np.repeat(1/n, n))
	C = np.asmatrix(np.cov(returns))
	mu = w * p.T
	sigma = np.sqrt(w*C*w.T)
	return mu, sigma

def solve(returns):
	n = returns.shape[0]
	returns = np.asmatrix(returns)
	N = returns.shape[1]
	mus = [10**(5 * t/N - 1) for t in range(N)]
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	portfolios = [solvers.qp(S, -mu*pbar, G, h, A, b)['x'] for mu in mus]
	return portfolios

def fit(returns, portfolios):
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	return returns, risks

def optimal_portfolios(returns):
	portfolios = solve(returns)
	returns, risks = fit(returns, portfolios)
	return returns, risks

def plot_random_portfolios(returns, ax):
	# Random portfolios
	means, stds = np.column_stack([random_portfolios(returns) for _ in range(500)])
	ax.plot(stds, means, 'bo', markersize=5, markeredgecolor='black', label='Random portfolios')	

def plot_naive_max_portfolio(returns, ax):
	# Naïve max assets
	m, s = naive_portfolio(returns)
	ax.plot(s, m, 'ro', markersize=5, markeredgecolor='black', label='Naive portfolio')
	return m, s

def plot_efficient_frontier(returns, ax):
	# Max assets Markowitz optimal
	weights, rets, risks = optimal_portfolios(returns)
	ax.plot(risks, rets, 'r-', label='Efficient frontier')

def plot_boundaries(returns, ax, mu0, s0):
	# Boundaries
	ax.axhline(y=mu0, color='#1f77b4', label=r'$\mu_0$')
	ax.axvline(x=s0, color='#ff7f0e', label=r'$\sigma_0$')

def plot_legend(ax):
	# legend
	return ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

def plot_2_port(returns, ax):
	n, d = returns.shape
	# 2-portfolios naïve and opt
	for i in combinations(list(range(n)), 2):
		rr = np.array([returns[i[0]], returns[i[1]]])
		#kw, kr, ks = optimal_portfolios(rr)
		#ax.plot(ks, kr, 'g-')	
		kkr, kks = naive_portfolio(rr)
		ax.plot(kks, kkr, 'go', marker="D", markersize=5, markeredgecolor='black', label='Naive 2-portfolios')

def plot_3_port(returns, ax):
	n, d = returns.shape
	# 3-portfolios naïve and opt
	for i in combinations(list(range(n)), 3):
		rr = np.array([returns[i[0]], returns[i[1]], returns[i[2]]])
		#kw, kr, ks = optimal_portfolios(rr)
		#ax.plot(ks, kr, 'y-')	
		kkr, kks = naive_portfolio(rr)
		ax.plot(kks, kkr, 'yo', marker="s", markersize=5, markeredgecolor='black', label='Naive 3-portfolios')

def plot_k_portfolios(returns, ax):
	plot_2_port(returns, ax)
	plot_3_port(returns, ax)

def skeleton(returns, mu0, s0):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('k-portfolios')
	ax.set_xlabel('Risk (standard deviation)')
	ax.set_ylabel('Return')

	plot_random_portfolios(returns, ax)
	m, s = plot_naive_max_portfolio(returns, ax)
	
	#plot_k_portfolios(returns, ax)
	plot_efficient_frontier(returns, ax)
	#plot_boundaries(returns, ax, mu0, s0)

	lgd = plot_legend(ax)
	# presentation
	fig.savefig('image_output.png', format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.show()

def iid_n01_data():
	return np.random.randn(4, 1000)

def generate_multinormal():
	mean = [.01, .05, .08, .035]
	cov = [[.03 , .15 , .04 , .3], [.15 , .2 , .45 , .06], [.04 , .45 , .4 , .2], [.3 , .06 , .2 , .02]]
	return np.random.multivariate_normal(mean, cov, 1000).T
	

from build_example_data import build_example_ml_return_data
from build_example_data import build_example_return_data
def ml_plot():
	#data, test = build_example_ml_return_data()
	data = np.random.randn(4, 10000)
	test = np.random.randn(4, 1000)
	#data = generate_multinormal()
	#test = generate_multinormal()
	portfolios = solve(data)
	# efficient training frontier
	returns, risks = fit(data, portfolios)
	# test curve
	returns_test, risks_test = fit(test, portfolios)
	#portfolios_test = solve(test)
	#returns_test_, risks_test_ = fit(test, portfolios_test)
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Trained frontier vs actual portfolio performance curve')
	ax.set_xlabel('Risk (standard deviation)')
	ax.set_ylabel('Return')
	ax.plot(risks, returns, '-', markersize=3, markeredgecolor='black', label='Trained efficient frontier')
	ax.plot(risks_test, returns_test, '-', markersize=3, markeredgecolor='black', label='Actual portfolio performance curve')
	#ax.plot(risks_test_, returns_test_, 'o-', markersize=3, markeredgecolor='black', label='Test data efficitent frontier')
	#lgd = plot_legend(ax)
	# presentation
	#fig.savefig('ml_image_output.png', format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	plot_naive_max_portfolio(test, ax)
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()
	train_mu = np.mean(data, axis=1)
	test_mu = np.mean(test, axis=1)
	train_cov = np.cov(data)
	test_cov = np.cov(test)
	return train_mu, test_mu, train_cov, test_cov

def main():
	#returns = iid_n01_data()
	#returns = other_data()
	#returns = build_example_return_data()
	returns = np.load('sp333.ndarray')
	(mu0, s0) = (0.02, 0.058)
	skeleton(returns, mu0, s0)
	#print(ml_plot())

if __name__ == "__main__":
	main()
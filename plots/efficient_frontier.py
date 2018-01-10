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

def optimal_portfolios(returns):
	n = len(returns)
	returns = np.asmatrix(returns)
	N = returns.shape[1]
	mus = [10**(5 * t/N - 1) for t in range(N)]
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	m1 = np.polyfit(returns, risks, 2)
	x1 = np.sqrt(m1[2] / m1[0])
	wt = solvers.qp(opt.matrix(x1*S), -pbar, G, h, A, b)['x']
	return np.asarray(wt), returns, risks 

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

	#plot_random_portfolios(returns, ax)
	m, s = plot_naive_max_portfolio(returns, ax)
	
	plot_k_portfolios(returns, ax)
	plot_efficient_frontier(returns, ax)
	plot_boundaries(returns, ax, mu0, s0)

	lgd = plot_legend(ax)
	# presentation
	fig.savefig('image_output.png', format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.show()

def iid_n01_data():
	return np.random.randn(4, 1000)

def other_data():
	return np.array([
		1 * np.random.randn(1000) - 2,
		1 * np.random.randn(1000) + 5,
		1.5 * np.random.randn(1000) + 3,
		10 * np.random.randn(1000) + 10
		])

def main():
	#returns = iid_n01_data()
	#returns = other_data()
	returns = np.load('example_returns.ndarray')
	(mu0, s0) = (0.02, 0.058)
	skeleton(returns, mu0, s0)

if __name__ == "__main__":
	main()
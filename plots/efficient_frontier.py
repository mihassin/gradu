import numpy as np
from matplotlib import pyplot as plt
import cvxopt as opt
from cvxopt import solvers, blas

returns = np.random.randn(4, 1000)

def random_weights(n):
	w = np.random.rand(n)
	return w / sum(w)

def random_portfolios(returns):
	p = np.asmatrix(np.mean(returns, axis=1))
	w = np.asmatrix(random_weights(returns.shape[0]))
	C = np.asmatrix(np.cov(returns))
	mu = w * p.T
	sigma = np.sqrt(w*C*w.T)
	if sigma > 2:
		return random_portfolios(returns)
	return mu, sigma

means, stds = np.column_stack([random_portfolios(returns) for _ in range(500)])

def optimal_portfolios(returns):
	n = len(returns)
	returns = np.asmatrix(returns)
	N = 1000
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

weights, rets, risks = optimal_portfolios(returns)

plt.plot(stds, means, 'bo', markersize=5, markeredgecolor='black')
plt.xlabel('Risk (standard deviation)')
plt.ylabel('Return')
plt.title('The Efficient Frontier')
plt.plot(risks, rets, 'g-')
plt.show()
import numpy as np
import cvxopt as opt
from cvxopt import solvers, blas

def cvxopt_solve_single(mean, cov):
	return cvxopt_solve(mean, cov, 1)

def cvxopt_solve(mean, cov, N):
	n = cov.shape[0]
	mus = [10**(5 * t/N - 1) for t in range(N)]
	S = opt.matrix(cov)
	pbar = opt.matrix(mean)
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	portfolios = [solvers.qp(S, -mu*pbar, G, h, A, b)['x'] for mu in mus]
	return portfolios

def cvxopt_solve_max_risk(mean, cov, N, maxrisk):
	n = cov.shape[0]
	precision = 1/N
	mus = [maxrisk*precision*i for i in range(1, N + 1)]
	S = opt.matrix(cov)
	pbar = opt.matrix(mean)
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	portfolios = [solvers.qp(S, -mu*pbar, G, h, A, b)['x'] for mu in mus]
	return portfolios


def cvxopt_fit(mean, cov, portfolios):
	pbar = opt.matrix(mean)
	S = opt.matrix(cov)
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	return returns, risks

def cvxopt_naive_fit(mean, cov):
	n = mean.shape[0]
	w = np.repeat(1/n, n)
	mu = np.dot(w, mean)
	sigma = np.sqrt(np.dot(np.dot(w, cov), w))
	return mu, sigma

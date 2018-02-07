import numpy as np

import cvxopt as opt
from cvxopt import solvers, blas

from cvxpy import *

def solve_cvxopt(mean, cov, N):
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

def fit_cvxopt(mean, cov, portfolios):
	pbar = opt.matrix(mean)
	S = opt.matrix(cov)
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	return returns, risks
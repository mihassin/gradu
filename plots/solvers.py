import numpy as np

from cvxpy import *

import cvxopt as opt
from cvxopt import solvers, blas

from sklearn import linear_model

# CVXPY
def cvxpy_lasso_single(data, tau, mu):
	t, n = data.shape
	w = Variable(n)
	mu0 = Parameter(t, sign='positive')
	lambd = Parameter(sign='positive')
	rbar = np.mean(data, axis=0)
	objective = Minimize(1/t * sum_squares(mu0 - (R*w)) + lambd*norm(w, 1))
	constraints = [rbar*w == mu, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	mu0.value = np.repeat(mu, t)
	lambd.value = tau
	prob.solve()
	return np.array(w.value).flatten()

def cvxpy_markowitz_leastsquares_single(data, mu):
	t, n = data.shape
	w = Variable(n)
	mu0 = Parameter(t, sign='positive')
	rbar = np.mean(data, axis=0)
	objective = Minimize(1/t * sum_squares(mu0 - (R*w)))
	constraints = [rbar*w == mu, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	mu0.value = np.repeat(mu, t)
	prob.solve()
	return np.array(w.value).flatten()

def cvxpy_markowitz_single(mean, cov, mu):
	return cvxpy_markowitz(mean, cov, [mu])

def cvxpy_markowitz(mean, cov, mus):
	n = cov.shape[0]
	w = Variable(n)
	mu0 = Parameter(sign='positive')
	rbar = np.mean(data, axis=1)
	Sigma = np.cov(data)
	risk = quad_form(w, Sigma)
	objective = Minimize(risk)
	constraints = [rbar*w == mu0, sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	portfolios = np.zeros((len(mus), n))
	for i in range(len(mus)):
		mu0.value = mus[i]
		prob.solve('CVXOPT')
		portfolios[i] = np.array(w.value).flatten()
	return portfolios

def solve_portfolios(data, w, mu0, prob, frequency = 0.0001, end_cond = 0.2):
	portfolios = []
	i = 0
	while i < end_cond:
		d = mu0.size[1]
		mu0.value = np.repeat(i, d)
		prob.solve()
		if type(w.value) != type(None):
			portfolios.append(np.array(w.value).flatten())
		i += frequency
	return portfolios

def cvxpy_fit(mean, cov, portfolios):
	SAMPLES = len(portfolios)
	risks = np.zeros(SAMPLES)
	returns = np.zeros(SAMPLES)
	for i in range(SAMPLES):
		w = portfolios[i]
		returns[i] = np.dot(mean, w)
		risks[i] = np.sqrt(np.dot(w, np.dot(cov, w)))
	return risks, returns

# CVXOPT
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

def lasso_attempt_1(mean, cov, N, maxrisk, tau):
	n = cov.shape[0]
	precision = 1/N
	mus = [maxrisk*precision*i for i in range(1, N + 1)]
	Q = opt.matrix(cov)
	pbar = opt.matrix(mean)
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	penalty = tau*opt.matrix(1.0, (n, 1))
	portfolios = [solvers.qp(Q, penalty-mu*pbar, G, h, A, b)['x'] for mu in mus]
	return portfolios

def lasso_single_attempt_1(mean, cov, mu, tau):
	n = cov.shape[0]
	Q = opt.matrix(cov)
	pbar = opt.matrix(mean)
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	penalty = tau*opt.matrix(1.0, (n, 1))
	portfolios = solvers.qp(Q, penalty-mu*pbar, G, h, A, b)['x']
	return portfolios

def lasso_single_attempt_2(mean, cov, mu, maxrisk, tau):
	n = cov.shape[0]
	precision = 1/N
	mus = [maxrisk*precision*i for i in range(1, N + 1)]
	Q = opt.matrix(cov)
	G = -opt.matrix(np.eye(n))
	h = opt.matrix(0.0, (n, 1))
	A = opt.matrix([np.ones(n), mean])
	b = opt.matrix([1.0, mu])
	penalty = tau*opt.matrix(1.0, (n, 1))
	portfolio = solvers.qp(Q, penalty, G, h, A, b)['x']
	return portfolio

# SKLEARN
def normalized_unconstrained_lasso(data, mu0):
	X = data.T
	y = np.repeat(mu0, X.shape[0])
	alphas, _, coefs = linear_model.lars_path(X, y, method='lasso')
	n, p = coefs.shape
	norm_coefs = np.zeros((n, p))
	sums = np.sum(coefs, axis=0)
	for i in range(n):
		for j in range(p):
			if not sums[j] == 0:
				c = 1/sums[j]
				norm_coefs[i, j] = c*coefs[i,j]
	# First value is alpha where every coef is zero
	# We require at least one coef to be non-zero
	return alphas[1:], norm_coefs.T[1:]
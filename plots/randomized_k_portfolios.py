import numpy as np

from solvers import cvxopt_solve_max_risk
from solvers import cvxopt_fit
from solvers import cvxpy_fit

def randomized_k_portfolios(n, k, p, size=1):
	'''result = np.random.choice(range(n), k, replace=0, p=p)
	portfolio = np.array([0 if x not in result else 1/k*1**x for x in range(n)])
	return portfolio'''	
	portfolios = np.array([])
	for i in range(size):
		result = np.random.choice(range(n), k, replace=0, p=p)
		portfolio = np.array([0 if x not in result else 1/k*1**x for x in range(n)])
		if portfolios.shape[0] == 0:
			portfolios = np.array([portfolio])
		else:
			portfolios = np.append(portfolios, [portfolio], axis=0)
	return portfolios

# 1 uniform
def uniform(n, k, size=1):
	return randomized_k_portfolios(n, k, np.repeat(1/n, n), size=size)

# 2 return weighted
def return_weighted(n, k, r):
	r_sum = np.sum(r)
	p = np.array([ri / r_sum for ri in r])
	return randomized_k_portfolios(n, k, p)

# 3 markowitz randomized
def markowitz_randomized(n, k, size, mean, cov, lambd, mu0, sigma0):
	N = 100
	portfolios = cvxopt_solve_max_risk(mean, cov, N, lambd)
	returns, risks = cvxopt_fit(mean, cov, portfolios)
	options = []
	for i in range(N):
		w = portfolios[i]
		risk = risks[i]
		ret = returns[i]
		if risk <= sigma0 and ret >= mu0:
			options.append(np.array(w).flatten())
	markowitz = options[np.random.choice(len(options))]
	return markowitz, np.array([randomized_k_portfolios(n, k, markowitz, size) for i in range(size)])

def asset_update(k, p, mean, cov, mu0, sigma0):
	n = mean.shape[0]
	RP = randomized_k_portfolios(n, k, p, size=1000)
	risks, returns = cvxpy_fit(mean, cov, RP)
	RPIR = np.array([])
	for i in range(len(RP)):
		if returns[i] > mu0 and risks[i] < sigma0:
			if RPIR.shape[0] == 0:
				RPIR = np.array([RP[i]])
			else:
				RPIR = np.append(RPIR, [RP[i]], axis=0)
	p = np.zeros(n)
	for i in range(n):
		p[i] = RPIR[RPIR[:, i] > 0].shape[0] / RPIR.shape[0]
	return RPIR, p
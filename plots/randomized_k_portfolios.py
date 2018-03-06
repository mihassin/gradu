import numpy as np

from solvers import cvxopt_solve_max_risk
from solvers import cvxopt_fit

def randomized_k_portfolios(n, k, p, size=1):
	result = np.random.choice(range(n), k, replace=0, p=p)
	return np.array([0 if x not in result else 1/k*1**x for x in range(n)])

# 1 uniform
def uniform(n, k):
	return randomized_k_portfolios(n, k, np.repeat(1/n, n))

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
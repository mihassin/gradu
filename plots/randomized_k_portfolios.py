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
	mu1, sigma1 = update_return_risk(RP, mu0, sigma0)

def asset_update_old(k, p, mean, cov, mu0, sigma0):
	# Number of assets
	n = mean.shape[0]
	# Size amout of random portfolios drawn with the distribution of p 
	RP = randomized_k_portfolios(n, k, p, size=1000)
	# Risks and returns of random portfolios RP
	risks, returns = cvxpy_fit(mean, cov, RP)
	# Initialization of intersection of RP and interesting region IR {(risk, return) : risk < sigma0 and return > mu0}
	RPIR = np.array([])
	# Method to find the set objects of RPIR
	for i in range(len(RP)):
		if returns[i] > mu0 and risks[i] < sigma0:
			if RPIR.shape[0] == 0:
				RPIR = np.array([RP[i]])
			else:
				RPIR = np.append(RPIR, [RP[i]], axis=0)
	# New asset distribution p_new initialization
	p_new = np.zeros(n)
	# Cardinality of RPIR
	m = RPIR.shape[0]
	# if statement to evade zero division error within this clause
	if m > 0:
		# p_new update method
		for i in range(n):
			z = 0
			for r in RPIR:
				if r[i] > 0:
					z += 1
		#p[i] = RPIR[RPIR[:, i] > 0].shape[0] / m
			p_new[i] = z / m
	# Normalization of p_new					
	if not p_new.sum() == 0: 
		p_new /= p_new.sum()
	else:
		p_new = p
	return RPIR, p_new

def narrow_IR(risks, returns, alpha=0.01, gamma=0.8):
	a = 0
	a += alpha
	i = 1
	n = returns.shape[0]
	max_risk = risks.max()
	min_return = returns.min()
	while(i > gamma):
		i = 0
		s0 = max_risk*(1-a)
		m0 = min_return*(1+a)	
		for ri, re in zip(risks, returns):
			if s0 > ri and m0 < re:
				i += 1
		i /= n
		if(i > gamma):
			a += alpha
		else:
			a -= alpha
	sigma = max_risk*(1-a)
	mu = min_return*(1+a)
	return sigma, mu, a



'''
j = 0
r1 = []
r2 = []
for i in range(100):
	print(i)
	RPIR1, p1 = asset_update(10, p0, mean, cov, 0.001, 0.018) 
	RPIR2, p2 = asset_update(10, p1, mean, cov, 0.001, 0.018) 
	i1 = RPIR1.shape[0] / 1000
	i2 = RPIR2.shape[0] / 1000
	r1.append(i1)
	r2.append(i2)
	if i2 > i1:
		j += 1
j / 100
'''

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


def asset_update(k, mean, cov, mu0, sigma0, alpha=0.1, e=1e-7):
	n = mean.shape[0]
	p = np.repeat(1/n, n)
	RP = randomized_k_portfolios(n, k, p, size=1000)
	risks, returns = cvxpy_fit(mean, cov, RP)
	s0, m0 = initial_borders(risks, returns, gamma=.5)
	RPIR = np.array([])
	x = np.array([])
	y = np.array([])
	j = 0
	while(s0 > sigma0 or m0 < mu0):
		ir = np.array([])
		ir00 = np.array([])
		# IR = {(sigma, mu) : sigma < s0 and mu > m0}
		for i in range(len(RP)):
			if returns[i] > m0 and risks[i] < s0:
				if ir.shape[0] == 0:
					ir = np.array([RP[i]])
				else:
					ir = np.append(ir, [RP[i]], axis=0)
			if returns[i] > mu0 and risks[i] < sigma0:
				if ir00.shape[0] == 0:
					ir00 = np.array([RP[i]])
				else:
					ir00 = np.append(ir00, [RP[i]], axis=0)
		p = np.zeros(n)
		m = ir.shape[0]
		j += 1
		x = np.append(x, j)
		y = np.append(y, ir00.shape[0])
		print(x)
		print(y)
		# If IR contains points
		if m > 0:
			for i in range(n):
				z = 0
				for r in ir:
					if r[i] > 0:
						z += 1
				p[i] = z / m
			RPIR = np.copy(ir)
			ri, re  = cvxpy_fit(mean,cov,ir00)
			#plt.plot(ri, re, 'o')
		# else report previous best
		else:
			print('Interesting region was empty. Reporting previous iteration')
			plt.plot(x, y)
			return RPIR, s0, m0
		# normalization
		p /= p.sum()
		# create RP for next iteration
		RP = randomized_k_portfolios(n, k, p, size=1000)
		risks, returns = cvxpy_fit(mean, cov, RP)
		# decrease s0 by the factor alpha of the difference of s0 and sigma0
		if s0 > sigma0:
			s0 = s0 - (s0 - sigma0) * alpha
		# increase m0 by the factor alpha of the difference of mu0 and m0
		if m0 < mu0:
			m0 = m0 + (mu0 - m0) * alpha
		# adding zeros
		if (s0 - sigma0)*alpha < e and (mu0 - m0)*alpha < e:
			plt.plot(x, y)
			return RPIR, s0, m0
		print('s: ' + str(s0) + ', m: ' + str(m0))
	plt.plot(x, y)
	return RPIR, s0, m0

a,s,m=asset_update(10,mean,cov,0.0012,0.014,alpha=0.3)

def initial_borders(risks, returns, alpha=0.01, gamma=0.8):
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
	return sigma, mu

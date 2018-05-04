import numpy as np
from src.solvers import cvxpy_fit
from src.solvers import normalized_unconstrained_lasso
from src.randomized_k_portfolios import asset_update

def results_ipu(bin_i, bin_ii, k):
	N = bin_i.shape[0]
	w_naive = np.repeat(1/N, N)
	mu_i = bin_i.mean(axis=1)
	mu_ii = bin_ii.mean(axis=1)
	cov_i = np.cov(bin_i)
	cov_ii = np.cov(bin_ii)
	sigma_w, mu_w = cvxpy_fit(mu_i, cov_i, [w_naive])
	a, w = normalized_unconstrained_lasso(bin_i, mu_w)
	w_lasso = np.array([])
	for w_ in w:
		w_i = np.array([1 for _ in w_ if not _ == 0])
		if np.sum(w_i) == k:
			w_lasso = w_
			break
	print(w_lasso.shape)
	std0, mu0 = cvxpy_fit(mu_i, cov_i, [w_lasso])
	w_ipu, s0, m0 = asset_update(k, mu_i, cov_i, mu0, std0)
	print(w_ipu)
	ris, rets = cvxpy_fit(mu_ii, cov_ii, w_ipu)
	bin_ret = rets.mean()
	bin_ri = ris.mean()
	bin_S = bin_ret / bin_ri
	return np.array([bin_ret, bin_ri, bin_S])

import numpy as np
from src.solvers import normalized_unconstrained_lasso
from src.solvers import cvxpy_fit

def results_lasso(bin_i, bin_ii, indexes):
	N = bin_i.shape[0]
	w_naive = np.repeat(1/N, N)
	sigma_w, mu_w = cvxpy_fit(bin_i.mean(axis=1), np.cov(bin_i), [w_naive])
	alph, w = normalized_unconstrained_lasso(bin_i, mu_w)
	for i in indexes:
		w_i = np.array([1 for _ in w[i] if not _ == 0])
		print(np.sum(w_i))
	bin_w = np.array([w[i] for i in indexes])
	bin_ri, bin_ret = cvxpy_fit(bin_ii.mean(axis=1), np.cov(bin_ii), bin_w)
	bin_S = bin_ret / bin_ri
	return np.array([bin_ret, bin_ri, bin_S]), mu_w

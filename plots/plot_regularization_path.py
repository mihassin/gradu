import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors

# solvers
from solvers import lasso_single
from solvers import cvxpy_fit
from solvers import cvxopt_solve
from solvers import cvxopt_fit

# data
from build_example_data import iid_n01_data
from build_example_data import generate_multinormal
from build_example_data import sample_multinormal
from build_example_data import build_example_ml_return_data

def plot_regularization_path(data, mu):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Regularization path ' + r'$\mu_0 = $' + str(mu))
	ax.set_xlabel(r'$\tau$')
	ax.set_ylabel('Weight')
	
	mean = np.mean(data, axis=1)
	cov = np.cov(data)

	taus = np.arange(0,11000, step=1000)
	portfolios = np.array([[p for p in lasso_single(mean, cov, mu, int(taus[0]))]])
	for tau in taus[1:]:
		portfolios = np.append(portfolios, [[p for p in lasso_single(mean, cov, mu, int(tau))]], axis=0)
	for p in portfolios.T:
		ax.plot(taus, p)
	print(portfolios)	
	save_image(plt, fig, ax)


def get_colors(n):
	c = colors.get_named_colors_mapping()
	c = list(c.values())
	c = c[:-8]
	return np.random.choice(c, n)

def save_image(plt, fig, ax):
	ax.legend()
	fig.savefig('reg_path.png', format='png')
	plt.show()

#data = np.load('DJ30.ndarray')
#data = np.load('sp332.ndarray')
#plot_regularization_path(data, 0.0006)

from sklearn import linear_model
def normalized_unconstrained_lasso(data, mu0):
	X = data.T
	y = np.repeat(mu0, X.shape[0])
	alphas, _, coefs = linear_model.lars_path(X, y, method='lasso')
	def f(coefs):
		n, p = coefs.shape
		norm_coefs = np.zeros((n, p))
		sums = np.sum(coefs, axis=0)
		for i in range(n):
			for j in range(p):
				if not sums[j] == 0:
					c = 1/sums[j]
					norm_coefs[i, j] = c*coefs[i,j]
		return norm_coefs
	norm_coefs = f(coefs)
	def plot_result(a, n, mu0, fn):
			fig = plt.figure()
			ax = plt.subplot(111)
			ax.set_title('Regularization path ' + r'$\mu_0 = $' + str(mu0))
			ax.set_xlabel(r'$\tau$')
			ax.set_ylabel('Weight')
			ax.plot(a, n)
			fig.savefig(fn, format='png')
			plt.show()
	#plot_result(alphas, norm_coefs.T, mu0, 'norm_reg_path_1.png')		
	#plot_result(alphas[1:], norm_coefs.T[1:], mu0, 'norm_reg_path_2.png')
	return alphas[1:], norm_coefs.T[1:]

data = np.load('DJ30.ndarray')
cov = np.cov(data)
mean = np.mean(data, axis=1)

fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Mean-standard deviation')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')

portfolios = cvxopt_solve(mean, cov, 100)
returns, risks = cvxopt_fit(mean, cov, portfolios)
ax.plot(risks, returns, label='Efficient frontier')

alphas, norm_coefs = normalized_unconstrained_lasso(data, 0.006)
risks, returns = cvxpy_fit(mean, cov, norm_coefs)
for i in range(len(risks)):
	ax.plot(risks[i], returns[i], 'o', label=r'$\tau$ = ' + str(np.round(np.sort(alphas)[i], decimals=8)))
ax.legend()
fig.savefig('Efficient.png', format='png')
plt.show()
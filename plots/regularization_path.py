import numpy as np
from cvxpy import *
from matplotlib import pyplot as plt
from matplotlib import colors

def lasso_solve_single(data, tau, mu):
	t, n = data.shape
	w = Variable(n)
	mu0 = Parameter(t, sign='positive')
	lambd = Parameter(sign='positive')
	rbar = np.mean(data, axis=0)
	objective = Minimize(1/t * sum_squares(mu0 - (R*w)) + lambd*norm(w,1))
	constraints = [rbar*w == mu, w>=0]#sum_entries(w) == 1, w >= 0]
	prob = Problem(objective, constraints)
	mu0.value = np.repeat(mu, t)
	lambd.value = tau
	prob.solve()
	return np.array(w.value).flatten()

def markowitz_solve_single(data, mu):
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


def plot_regularization_path(data, mu0):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Regularization path ' + r'$\mu_0 = $' + str(mu0))
	ax.set_xlabel(r'$\tau$')
	ax.set_ylabel('Weight')
	
	mu = 0.0006
	#taus = [10**i for i in range(8)]
	taus = np.arange(1, 11)
	#taus[0] = 0
	print(taus)
	portfolios = np.array([lasso_solve_single(data, 0, mu0)])
	for tau in taus[1:]:
		portfolios = np.append(portfolios, [lasso_solve_single(data, tau, mu)], axis=0)
	for p in portfolios.T:
		#ax.plot(range(len(taus)), p)
		ax.plot(taus, p)
	#taus.insert(0, 0)
	#ax.set_xticklabels(taus)
	save_image(plt, fig, ax)

def get_colors(n):
	c = colors.get_named_colors_mapping()
	c = list(c.values())
	c = c[:-8]
	return np.random.choice(c, n)

def save_image(plt, fig, ax):
	ax.legend()
	fig.savefig('ml_image_output.png', format='png')
	plt.show()

data = np.load('DJ30.ndarray')
R = data.T
plot_regularization_path(R, 0.0006)


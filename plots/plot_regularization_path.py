import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors

# solvers
from solvers import lasso_single

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

	taus = np.arange(100,1100, step=100)
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
data = np.load('sp332.ndarray')
plot_regularization_path(data, 0.002)
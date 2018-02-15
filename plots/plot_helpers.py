from matplotlib import pyplot as plt

def create_fig(title, x, y):
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_title(title)
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	return fig, ax 

def save_image(fig, ax, plt_name='output.png' ,legend=True):
	if legend:
		ax.legend()
	fig.savefig(plt_name, format='png')
	plt.show()

def get_colors(n):
	c = colors.get_named_colors_mapping()
	c = list(c.values())
	c = c[:-8]
	return np.random.choice(c, n)
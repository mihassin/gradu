import numpy as np

def build_example_return_data():
	data = np.array([])
	files = ['FB.csv', 'IBM.csv', 'INTC.csv', 'NVDA.csv']
	for file in files:
		tmp = np.genfromtxt(file, delimiter=',')
		tmp = tmp[2:]
		tmp = tmp.T[1]
		r = np.array([(tmp[i+1] - tmp[i]) / tmp[i] for i in range(len(tmp) - 1)])
		if not data.size:
			data = np.array([r])
		else:
			data = np.append(data, [r], axis=0)
	return data

def main():
	data = build_example_return_data()
	data.dump('example_returns.ndarray')

if __name__ == '__main__':
	main()
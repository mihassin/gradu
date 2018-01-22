import numpy as np

def build_example_return_data():
	data = np.array([])
	files = ['FB.csv', 'IBM.csv', 'INTC.csv', 'NVDA.csv']
	files = ['stocks/' + file for file in files]
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

def build_example_ml_return_data():
	data = np.array([])
	files = ['AMD.csv', 'APC.csv', 'MCD.csv', 'WMT.csv']
	files = ['stocks/' + file for file in files]
	for file in files:
		tmp = np.genfromtxt(file, delimiter=',')
		tmp = tmp[2:]
		tmp = tmp.T[1]
		r = np.array([(tmp[i+1] - tmp[i]) / tmp[i] for i in range(len(tmp) - 1)])
		if not data.size:
			data = np.array([r])
		else:
			data = np.append(data, [r], axis=0)
	test = data[:,-12:]
	return data[:,:-12], test

def build_DJ30():
	dtype = ['i8', 'datetime64', 'f8', 'f8', 'f8', 'f8', 'i8', 'f8', 'S5']
	file = 'DJ30-1985-2003.csv'
	data = np.genfromtxt(file, dtype=dtype, delimiter=';')

def correct_form_DJ30():
	# must be exicuted in the same directory where this file is located
	file = 'DJ30-1985-2003.csv'
	with open(file, 'r+') as f:
		lines = f.readlines()
		f.seek(0)
		f.truncate()
		for line in lines:
			comas = 0
			i = 0
			j = 0
			for c in line:
				if comas < 1:
					i += 1
				if comas < 2:
					j += 1
				if comas == 2:
					break
				if c == ';':
					comas += 1
			date = line[i+1:j-2]
			stuff = date.split('-')
			year = stuff[2]
			if year == '00' or year == '01' or year == '02' or year == '03':
				year = '20'+year
			else:
				year = '19'+year
			month = stuff[1]
			month = str(strptime(month, '%b').tm_mon)
			if len(month) == 1:
				month = '0' + month
			day = stuff[0]
			if len(day) == 1:
				day = '0' + day
			new_date= year + '-' + month + '-' + day
			line = line.replace(date, new_date)
			line = line.replace(',', '.')
			line = line.replace('"', '')
			f.write(line)

def main():
	data = build_example_return_data()
	data.dump('example_returns.ndarray')

if __name__ == '__main__':
	main()
import numpy as np
from sklearn import linear_model

filename = 'Average_Value_Weighted_Returns.csv'
data = np.genfromtxt(filename, delimiter=',')

# start from 1976/07
data = data[601:]
tdata = np.transpose(data)

# exclude months
tdata = tdata[1:]

# missing values
for i in range(len(tdata)):
	for j in range(len(tdata[i])):
		if tdata[i][j] == -99.99:
			tdata[i][j] = False

# remove vectors with missing values
tmp = np.array([tdata[0]])
for i in tdata[1:]:
	if i.all():
		tmp = np.append(tmp, [i], axis=0)

train = np.transpose(tmp)
T, N = train.shape

# target construction
portfolio_expected_return = 1.1
target = np.repeat(portfolio_expected_return, T)



# model
tau = 1/T
reg = linear_model.LassoLars(alpha = tau, fit_intercept=False, positive=True)
reg.fit(train, target)
print(reg.coef_, np.sum(reg.coef_))

import numpy as np

filename = 'Average_Value_Weighted_Returns.csv'
data = np.genfromtxt(filename, delimiter=',')

# data 1976/07 - 2006/06 
data = data[601:961]
tdata = np.transpose(data)

A = np.transpose(tdata[1:])
print(A.mean())
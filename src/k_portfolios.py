from itertools import combinations as subsets
import numpy as np

# Breadth first travelsal of D
def breadth_first(D, k, minret, maxrisk):
	B = list()
	for i in range(k):
		Ci = subsets(D, i)
		for C in Ci:
			if np.mean(C) >= minret and np.var(C) <= maxrisk:
				B.append(C)
	return B

# Depth first travelsal of D
def depth_first(D, k, minret, maxrisk):
	B = list()
	S = D.copy()
	while S:
		L = list(S.pop(0))
		if np.mean(L) >= minret and np.var(L) <= maxrisk:
			B.append(L)
		if(len(L) < k):
			v = L[-1]
			i = D.index(v)
			C = D[i + 1:]	
			for c in reversed(C):
				L_ = L.copy()
				L_.extend(c)
				S.append(L_)
	return B

# Correct solution, with reverse
def test_depth_first(D, k):
	B = list()
	S = D.copy()
	while S:
		L = list(S.pop(0))
		B.append(L)
		if(len(L) < k):
			# tail element of L
			v = L[-1]
			i = D.index(v)
			# tail of the original D starting from index i+1
			C = D[i+1:]
			# Candidate generation, get rid of reverse
			for c in reversed(C):
				L_ = L.copy()
				L_.extend(c)
				S.insert(0, L_)
	return B

# Correct solution, however subsets not ordered (might not matter)
def test_depth_first_no_reverse(D, k):
	B = list()
	S = D.copy()
	while S:
		L = list(S.pop(0))
		B.append(L)
		if(len(L) < k):
			# tail element of L
			v = L[-1]
			i = D.index(v)
			# tail of the original D starting from index i+1
			C = D[i+1:]
			# Candidate generation
			for c in C:
				L_ = L.copy()
				L_.extend(c)
				S.insert(0, L_)
	return B

def test_depth_first_indecies(data, k):
	A = []
	N, D = data.shape
	S = [[i] for i in range(D)]
	while S:
		B = S.pop(0)
		A.append(B)
		if(len(B) < k):
			i = B[-1]
			for j in range(D-1, i, -1):
				C = B.copy()
				C.append(j)
				S.insert(0, C)
	return A

data = list('abcd')
print(test_depth_first(data, 3))
print(test_depth_first_no_reverse(data, 3))

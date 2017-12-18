
import numpy as np
from time import perf_counter as pc
from cvxpy import *



filename = 'Average_Value_Weighted_Returns.csv'
data = np.genfromtxt(filename, delimiter=',')

# data 1976/07 - 2006/06 
data = data[601:961]
tdata = np.transpose(data)

A = np.transpose(tdata[1:])
T, N = A.shape
portfolio_expected_return = 0.066
y = np.repeat(portfolio_expected_return, T)

alpha = 0.1

def obj(x):  # following sklearn's definition from user-guide!
    return (1. / (2*A.shape[0])) * np.square(np.linalg.norm(A.dot(x) - y, 2)) + alpha * np.linalg.norm(x, 1)


""" accelerated pg  -> sum x == 1 """
def solve_pg(A, b, momentum=0.9, maxiter=1000):
    """ remarks:
            algorithm: accelerated projected gradient
            projection: proj on probability-simplex
                -> naive and slow using cvxpy + ecos
            line-search: armijo-rule along projection-arc (Bertsekas book)
                -> suffers from slow projection
            stopping-criterion: naive
            gradient-calculation: precomputes AtA
                -> not needed and not recommended for huge sparse data!
    """

    M, N = A.shape
    x = np.zeros(N)

    AtA = A.T.dot(A)
    Atb = A.T.dot(b)

    stop_count = 0

    # projection helper
    x_ = Variable(N)
    v_ = Parameter(N)
    objective_ =  Minimize(0.5 * square(norm(x_ - v_, 2)))
    constraints_ = [sum(x_) == 1]
    problem_ = Problem(objective_, constraints_)

    def gradient(x):
        return AtA.dot(x) - Atb

    def obj(x):
        return 0.5 * np.linalg.norm(A.dot(x) - b)**2

    it = 0
    while True:
        grad = gradient(x)

        # line search
        alpha = 1
        beta = 0.5
        sigma=1e-2
        old_obj = obj(x)
        while True:
            new_x = x - alpha * grad
            new_obj = obj(new_x)
            if old_obj - new_obj >= sigma * grad.dot(x - new_x):
                break
            else:
                alpha *= beta

        x_old = x[:]
        x = x - alpha*grad

        # projection
        v_.value = x
        problem_.solve()
        x = np.array(x_.value.flat)

        y = x + momentum * (x - x_old)

        if np.abs(old_obj - obj(x)) < 1e-2:
            stop_count += 1
        else:
            stop_count = 0

        if stop_count == 3:
            print('early-stopping @ it: ', it)
            return x

        it += 1

        if it == maxiter:
            return x


print('\n acc pg')
t0 = pc()
x = solve_pg(A, y)
print('used (secs): ', pc() - t0)
print(obj(x))
print('sum x: ', np.sum(x))
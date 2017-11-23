import numpy as np

def fitness(x):
	return sum(x)
	
	
N = 10
x = [0]*N

f = fitness(x)
sigma = 1./N

for i in xrange(10):

	x_ = [(bit+1)%2 if np.random.random()<sigma else bit for bit in x]
	f_ = fitness(x_)
	
	if f_ > f:
		f = f_
		x = x_

	print x
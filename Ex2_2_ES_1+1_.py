# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:40:16 2017

@author: Philip
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark_functions import sphere, rosenbrock, rastrigin, schwefel, get_parameter_range


fitness_fun = sphere

N_dim = 2
N_generations = 1000

mut_sigma = 1.0
mut_tau = 1.2

rechenberg_G = 5
rechenberg_succes_counter = 0

class Chromosome():
    
    def __init__(self, x):
        self.x = x
        self.fitness = self.__fitness()
    
    def __fitness(self):
        return fitness_fun(self.x)



def get_parent():
    x = ( xi_high - xi_low ) * np.random.random( N_dim ) + xi_low
    return Chromosome(x)


def muatation(child):
    x_mut = child.x + mut_sigma * np.random.normal(size=N_dim)
    for xi in x_mut:
        if xi > xi_high:
            xi = xi_high
        elif xi < xi_low:
            xi = xi_low
    return Chromosome(x_mut)
    

(xi_low, xi_high) = get_parameter_range( fitness_fun.__name__ )
parent = get_parent()

evo_hist = [parent]


for i in range(N_generations):
    
    child = parent
    child = muatation(child)
    
#    print(parent.fitness, child.fitness)
    
    if child.fitness < parent.fitness:
        rechenberg_succes_counter += 1
        parent = child
    
    evo_hist.append(parent)
    
    if (i+1) % rechenberg_G == 0:
        if rechenberg_succes_counter / rechenberg_G > 1/5:
            mut_sigma *= mut_tau
        elif rechenberg_succes_counter / rechenberg_G < 1/5:
            mut_sigma /= mut_tau
        rechenberg_succes_counter = 0
    


evo_fitness = [ x.fitness for x in evo_hist ]
evo_x = [ x.x[0] for x in evo_hist ]
evo_y = [ x.x[1] for x in evo_hist ]


fig, axs = plt.subplots(1, 2)
#fig.tight_layout()
axs[0].plot(np.log10(evo_fitness))

#fig, ax0 = plt.subplots()
#fig.tight_layout()

x = y = np.linspace(start=xi_low, stop=xi_high, num=200)
X, Y = np.meshgrid(x, y)
Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
Z = np.reshape(Zs, X.shape)

im = axs[1].pcolormesh(X, Y, Z, cmap='RdYlBu_r')
axs[1].set_aspect('equal','box')
fig.colorbar(im, ax=axs[1])
axs[1].plot(evo_x,evo_y,color='black')
    
plt.show()


















    






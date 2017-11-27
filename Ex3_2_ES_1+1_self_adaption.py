#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:43:07 2017

@author: pmirbach
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark_functions import sphere, rosenbrock, rastrigin, schwefel, get_parameter_range
from plot_scripts import plot_f_logf_sigma


fitness_fun = rosenbrock
N_dim = 2
N_generations = 1000


flag_show2Dplot = 1
if N_dim != 2:
    flag_show2Dplot = 0



mut_sigma_0 = 1.0
mut_tau = 0.95


class Chromosome():
    
    def __init__(self, x, sigma):
        self.x = x
        self.sigma = sigma
        self.fitness = self.__fitness()
    
    def __fitness(self):
        return fitness_fun(self.x)



def get_parent():
    x = np.array( ( xi_high - xi_low ) * np.random.random( N_dim ) + xi_low )
    return Chromosome(x, mut_sigma_0)


def muatation_log_normal(child):
    sigma_mutated = child.sigma * np.exp( mut_tau * np.random.normal() )
    
    x_mutated = child.x + sigma_mutated * np.random.normal( size=N_dim )
    x_mutated = np.where( x_mutated > xi_high, xi_high, x_mutated )
    x_mutated = np.where( x_mutated < xi_low, xi_low, x_mutated )
    return Chromosome( x_mutated, sigma_mutated )
    

(xi_low, xi_high) = get_parameter_range( fitness_fun.__name__ )
parent = get_parent()

evo_hist = [parent]


for i in range(N_generations):
    
    child = parent
    child_mutated = muatation_log_normal(child)
    
    if child_mutated.fitness < parent.fitness:
        parent = child_mutated
        
    evo_hist.append(parent)
    




# Get data from evolution history to plot
evo_fitness = [ x.fitness for x in evo_hist ]
evo_mut_sigma = [ x.sigma for x in evo_hist ]


# Plotting
if flag_show2Dplot:
    fig, axs = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1,3]})
    fig.tight_layout()
    ax_fit = plot_f_logf_sigma(axs[0], evo_fitness, evo_mut_sigma)
else:
    fig, axs = plt.subplots()
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    ax_fit = plot_f_logf_sigma(axs, evo_fitness, evo_mut_sigma)
mng = plt.get_current_fig_manager()
if hasattr(mng, 'window'):
    mng.window.showMaximized()


fig.subplots_adjust(top=0.93)
fig.suptitle('(1+1)-ES with Gaußmutation and Rechenberg, {} dimensions, {} generations'.format(N_dim, N_generations)
             ,fontsize=16)


if flag_show2Dplot:
    evo_x = [ x.x[0] for x in evo_hist ]
    evo_y = [ x.x[1] for x in evo_hist ]

    # Create Z data of benchmark function
    x = y = np.linspace(start=xi_low, stop=xi_high, num=200)
    X, Y = np.meshgrid(x, y)
    Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    Z = np.reshape(Zs, X.shape)

    im = axs[1].pcolormesh(X, Y, Z, cmap='RdYlBu_r')
    axs[1].set_aspect('equal','box')
    fig.colorbar(im, ax=axs[1])
    axs[1].plot(evo_x,evo_y,color='black',marker='x')
    
plt.show()























    






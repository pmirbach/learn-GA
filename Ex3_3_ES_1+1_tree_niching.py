#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:01:14 2017

@author: pmirbach
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark_functions import schwefel, get_parameter_range
from plot_scripts import plot_f_logf_sigma


fitness_fun = schwefel
N_dim = 2
N_generations = 10000


flag_show2Dplot = 1
if N_dim != 2:
    flag_show2Dplot = 0


mut_sigma = 50.0
mut_tau = 1.2

niching_generations = 50
niching_probability = 0.1
niching_mutation_rate = 5.0


class Chromosome():
    
    def __init__(self, x):
        self.x = x
        self.fitness = self.__fitness()
    
    def __fitness(self):
        return fitness_fun(self.x)



def get_parent():
    x = np.array( ( xi_high - xi_low ) * np.random.random( N_dim ) + xi_low )
    return Chromosome( x )


def mutation(child, mutation_rate):
    x_mutated = child.x + mutation_rate * np.random.normal( size=N_dim )
    x_mutated = np.where( x_mutated > xi_high, xi_high, x_mutated )
    x_mutated = np.where( x_mutated < xi_low, xi_low, x_mutated )
    return Chromosome( x_mutated )


def GA( parent, mutation_rate, N_generations, niching=False ):
    
    evo_hist = [parent]
    
    for i in range(N_generations):      
        child = mutation(parent, mutation_rate)
        
        if child.fitness >= parent.fitness:    
            
            if niching and np.random.random() < niching_probability:
                
                evo_hist_niche = GA( child, niching_mutation_rate, niching_generations )
                
                if evo_hist_niche[-1].fitness < parent.fitness:
                    parent = evo_hist_niche[-1]
        else:          
            parent = child
        
        evo_hist.append(parent)   
    return evo_hist




(xi_low, xi_high) = get_parameter_range( fitness_fun.__name__ )
parent = get_parent()


evo_hist_prae = GA(parent, niching_mutation_rate, niching_generations, niching=False )
evo_hist = GA(evo_hist_prae[-1], mut_sigma, N_generations, niching=True )

evo_hist = evo_hist_prae + evo_hist



# Get data from evolution history to plot
evo_fitness = [ x.fitness for x in evo_hist ]
#evo_mut_sigma = [ x.sigma for x in evo_hist ]
evo_mut_sigma = mut_sigma * np.ones(N_generations)


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
fig.suptitle('(1+1)-ES with Gaußmutation and Log-Normal Mutation, {} dimensions, {} generations'.format(N_dim, N_generations)
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
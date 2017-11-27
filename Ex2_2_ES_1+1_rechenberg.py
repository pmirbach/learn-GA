# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:40:16 2017

@author: Philip
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark_functions import sphere, rosenbrock, rastrigin, schwefel, get_parameter_range
from plot_scripts import plot_f_logf_sigma


fitness_fun = rastrigin
N_dim = 2
N_generations = 100


flag_show2Dplot = 1
if N_dim != 2:
    flag_show2Dplot = 0



mut_sigma = 1.0
mut_tau = 1.5

rechenberg_G = 5
rechenberg_succes_counter = 0

class Chromosome():
    
    def __init__(self, x):
        self.x = x
        self.fitness = self.__fitness()
    
    def __fitness(self):
        return fitness_fun(self.x)



def get_parent():
    x = np.array( ( xi_high - xi_low ) * np.random.random( N_dim ) + xi_low )
    return Chromosome(x)


def muatation(child):
    x_mut = child.x + mut_sigma * np.random.normal(size=N_dim)
    x_mut = np.where( x_mut > xi_high, xi_high, x_mut )
    x_mut = np.where( x_mut < xi_low, xi_low, x_mut )
    return Chromosome(x_mut)
    

(xi_low, xi_high) = get_parameter_range( fitness_fun.__name__ )
parent = get_parent()


evo_hist = [(parent, mut_sigma)]


for i in range(N_generations):
    
    child = parent
    child = muatation(child)
    
    if child.fitness < parent.fitness:
        rechenberg_succes_counter += 1
        parent = child
    
    if (i+1) % rechenberg_G == 0:
        if rechenberg_succes_counter / rechenberg_G > 1/5:
            mut_sigma *= mut_tau
        elif rechenberg_succes_counter / rechenberg_G < 1/5:
            mut_sigma /= mut_tau
        rechenberg_succes_counter = 0
    
    evo_hist.append((parent, mut_sigma))
    

# Get data from evolution history to plot
evo_fitness = [ x[0].fitness for x in evo_hist ]
evo_mut_sigma = [ x[1] for x in evo_hist ]

if flag_show2Dplot:
    evo_x = [ x[0].x[0] for x in evo_hist ]
    evo_y = [ x[0].x[1] for x in evo_hist ]

    # Create Z data of benchmark function
    x = y = np.linspace(start=xi_low, stop=xi_high, num=200)
    X, Y = np.meshgrid(x, y)
    Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    Z = np.reshape(Zs, X.shape)


# Plotting
if flag_show2Dplot:
    fig, axs = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1,3]})
else:
    fig, axs = plt.subplots()
mng = plt.get_current_fig_manager()
if hasattr(mng, 'window'):
    mng.window.showMaximized()
    
fig.tight_layout()
fig.subplots_adjust(top=0.93)
fig.suptitle('(1+1)-ES with Gaußmutation and Rechenberg, {} dimensions, {} generations'.format(N_dim,N_generations),fontsize=16)


#def plot_f_logf_sigma(ax, f, sigma):
#    axes = [ax, ax.twinx(), ax.twinx()]
#    axes[-1].spines['right'].set_position(('axes',1.2))
#    axes[-1].set_frame_on(True)
#    axes[-1].patch.set_visible(False)
#    
#    plot_data = [f, np.log10(f), sigma]
#    ylabels = ['Fitness', '$\log_{10}$ Fitness', 'Muationrate $\sigma$']
#    linestyles = [':','-','--']
#    colors = ['Blue', 'Red', 'Green']
#    
#    for ax, data, ylabel, linestyle, color in zip(axes, plot_data, ylabels, linestyles, colors):
#        ax.plot(data, linestyle=linestyle, color=color)
#        ax.set_ylabel(ylabel, color=color,fontsize=14)
#        ax.tick_params(axis='y', colors=color)
#        
#    return axes

if flag_show2Dplot:
    ax_fit = plot_f_logf_sigma(axs[0], evo_fitness, evo_mut_sigma)
else:
    fig.subplots_adjust(right=0.75)
    ax_fit = plot_f_logf_sigma(axs, evo_fitness, evo_mut_sigma)

if flag_show2Dplot:
    im = axs[1].pcolormesh(X, Y, Z, cmap='RdYlBu_r')
    axs[1].set_aspect('equal','box')
    fig.colorbar(im, ax=axs[1])
    axs[1].plot(evo_x,evo_y,color='black',marker='x')
    
plt.show()























    






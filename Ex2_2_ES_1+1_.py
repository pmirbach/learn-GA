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
N_generations = 100

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

evo_hist = [(parent, mut_sigma)]


for i in range(N_generations):
    
    child = parent
    child = muatation(child)
    
#    print(parent.fitness, child.fitness)
    
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
evo_x = [ x[0].x[0] for x in evo_hist ]
evo_y = [ x[0].x[1] for x in evo_hist ]
evo_mut_sigma = [ x[1] for x in evo_hist ]


# Create Z data of benchmark function
x = y = np.linspace(start=xi_low, stop=xi_high, num=200)
X, Y = np.meshgrid(x, y)
Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
Z = np.reshape(Zs, X.shape)



fig, axs = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1,3]})
mng = plt.get_current_fig_manager()
if hasattr(mng, 'window'):
    mng.window.showMaximized()
fig.tight_layout()

def plot_f_logf_sigma(ax, f, sigma):
    axes = [ax, ax.twinx(), ax.twinx()]
    axes[-1].spines['right'].set_position(('axes',1.2))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    
    plot_data = [f, np.log10(f), sigma]
    ylabels = ['Fitness', '$\log_{10}$ Fitness', 'Muationrate $\sigma$']
    linestyles = [':','-','--']
    colors = ['Blue', 'Red', 'Green']
    
    for ax, data, ylabel, linestyle, color in zip(axes, plot_data, ylabels, linestyles, colors):
        ax.plot(data, linestyle=linestyle, color=color)
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', colors=color)
        
    return axes

ax_fit = plot_f_logf_sigma(axs[0], evo_fitness, evo_mut_sigma)


im = axs[1].pcolormesh(X, Y, Z, cmap='RdYlBu_r')
axs[1].set_aspect('equal','box')
fig.colorbar(im, ax=axs[1])
axs[1].plot(evo_x,evo_y,color='black',marker='x')
    
plt.show()























    






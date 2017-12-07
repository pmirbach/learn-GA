#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:24:29 2017

@author: pmirbach
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark_functions import sphere, get_parameter_range
from plot_scripts import plot_f_logf_sigma, get_restrain_cmap



N_dim = 2
N_generations = 1000


flag_show2Dplot = 1
if N_dim != 2:
    flag_show2Dplot = 0


mut_sigma_0 = 1.0
mut_tau = 1.2


rechenberg_G = 5

penalty_G = 5
penalty_alpha_0 = 0.1


# # # Fitness and penalty function # # #
fitness_fun = sphere

def penalty_fun(x):
    g = np.sum(x) - N_dim
    return - np.min([g, 0])


# # # Classes for Chromosomes # # #
class Base_Chromosome(object):
    def __init__(self, x, *args, **kwargs):
        self.x = x
        self.fitness = fitness_fun(x)

class Chromosome_restrain(Base_Chromosome):
    def __init__(self, x, alpha, *args, **kwargs):
        super(Chromosome_restrain, self).__init__(x, *args, **kwargs)
        self.penalty = penalty_fun(x)
        self.infeasible = True if self.penalty else False
        self.alpha = alpha
        self.fitness_eff = self.fitness + alpha * self.penalty
    
    def set_alpha(self, alpha):
        self.alpha = alpha
        self.fitness_eff = self.fitness + alpha * self.penalty

class Chromosome_adaptive(Base_Chromosome):
    def __init__(self, x, sigma, *args, **kwargs):
        super(Chromosome_adaptive, self).__init__(x, *args, **kwargs)
        self.sigma = sigma

class Chromosome(Chromosome_restrain, Chromosome_adaptive):
    def __init__(self, x, *args, **kwargs):
        super(Chromosome, self).__init__(x, *args, **kwargs)




def get_parent():
    infeasible = True
    while infeasible:
        x = np.array((xi_high - xi_low) * np.random.random(N_dim) + xi_low)
        Candidate = Chromosome(x, sigma=mut_sigma_0, alpha=penalty_alpha_0)
        if Candidate.penalty == 0:
            infeasible = False
    return Candidate


def mutation(child):
    x_mutated = child.x + child.sigma * np.random.normal(size=N_dim)
#    x_mutated = np.where(x_mutated > xi_high, xi_high, x_mutated)
#    x_mutated = np.where(x_mutated < xi_low, xi_low, x_mutated)
    return Chromosome(x_mutated, sigma=child.sigma, alpha=child.alpha)



#def selection(old, new):
#    if new.fitness_eff < old.fitness_eff:
        

def GA_death(parent, N_generations, method='death', rechenberg=False):
    
    if method == 'death':
        alpha_vec = [1e9] * N_generations          # effective death penalty
    elif method == 'det':
        alpha_vec = np.linspace(start=2.0, stop=100, num=N_generations)
    rechenberg_succes_counter = 0

    evo_hist = [parent]
    for i in range(N_generations):
        child = mutation(parent)
        if child.fitness_eff < parent.fitness_eff:
            parent = child
            rechenberg_succes_counter += 1
        if rechenberg:
            if (i+1) % rechenberg_G == 0:
                if rechenberg_succes_counter / rechenberg_G > 1/5:
                    parent.sigma *= mut_tau
                elif rechenberg_succes_counter / rechenberg_G < 1/5:
                    parent.sigma /= mut_tau
                rechenberg_succes_counter = 0
        evo_hist.append(parent)
        parent.set_alpha(alpha_vec[i])
    return evo_hist



def GA_adaptive(parent, N_generations):

    penalty_counter = 0
    rechenberg_succes_counter = 0
    
    rechenberg_G = 50

    evo_hist = [parent]

    for i in range(N_generations):

#        pen_alpha = pen_alpha_vec[i]

        child = mutation(parent)
        if child.infeasible:
            penalty_counter += 1

        if child.fitness_eff < parent.fitness_eff:
            rechenberg_succes_counter += 1
            parent = child

        if (i+1) % rechenberg_G == 0:
            if rechenberg_succes_counter / rechenberg_G > 1/5:
                parent.sigma *= mut_tau
            elif rechenberg_succes_counter / rechenberg_G < 1/5:
                parent.sigma /= mut_tau
            rechenberg_succes_counter = 0

        if (i+1) % penalty_G == 0:
            if penalty_counter / penalty_G > 1/5:
                parent.set_alpha(parent.alpha * 2.0)
            elif penalty_counter / penalty_G < 1/5:
                parent.set_alpha(parent.alpha / 1.2)
            penalty_counter = 0

        evo_hist.append(parent)
    return evo_hist


(xi_low, xi_high) = get_parameter_range(fitness_fun.__name__)

parent = get_parent()

#evo_hist = GA_death(parent, 10000, method='det', rechenberg=True)
evo_hist = GA_adaptive(parent, 1000)




# Get data from evolution history to plot
evo_fitness = [ x.fitness for x in evo_hist ]
evo_mut_sigma = [ x.sigma for x in evo_hist ]
evo_alpha = [ x.alpha for x in evo_hist ]


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
titlestr = '(1+1)-ES, Gaußmutation, tangent-restrain, N:{0}, gens:{1}, solution: {2}'
fig.suptitle(titlestr.format(N_dim, N_generations, evo_hist[-1].x), fontsize=16)



if flag_show2Dplot:
    evo_x = [ x.x[0] for x in evo_hist ]
    evo_y = [ x.x[1] for x in evo_hist ]

    # Create Z data of benchmark function
    x = y = np.linspace(start=xi_low, stop=xi_high, num=200)
    X, Y = np.meshgrid(x, y)
    Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    Z = np.reshape(Zs, X.shape)
    Penaltys = np.array( [ penalty_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    Penalty = np.reshape(Penaltys, X.shape)
    
    cm = get_restrain_cmap()
    
    xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
    extent = xmin, xmax, ymin, ymax

    im = axs[1].imshow(Z, vmax=np.amax(Z)/2, cmap='ocean_r',interpolation='bilinear',
            extent=extent,origin="lower")
    axs[1].set_aspect('equal','box')
    fig.colorbar(im, ax=axs[1])
    im2 = axs[1].imshow(Penalty, alpha=0.5, cmap=cm,interpolation='bilinear', 
             extent=extent,origin="lower")
    
    penalty_border = [(-8,10), (10,-8)]
    axs[1].plot(penalty_border[0], penalty_border[1], alpha=0.2, color='red')
    
    axs[1].plot(evo_x,evo_y,color='black',marker='x')

plt.show()

#fig2, ax2 = plt.subplots()
#ax2.plot(evo_alpha)


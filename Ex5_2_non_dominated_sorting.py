#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:25:53 2017

@author: pmirbach
"""

import numpy as np
from copy import deepcopy



N_dim = 30              # Number of dimensions
N_generations = 100     # Number of generations


mut_sigma = 0.05        # Mutation step-size


flg_parents_nr_fixed = 0
N_parents = 10          # Number of parents in GA
N_childs = 100          # Number of childs in GA
GA_selection_method = 'plus'    # plus or comma selection
GA_save_hist = 1        # Save history of evolution

plot_update_ylim = 1


xi_low, xi_high = 0, 1  # ZDT1 defined for 0<xi<1



class Chromosome():
    
    def __init__(self, x):
        
        self.x = x
        self.f1, self.f2 = self.__fitness(x)
    
    
    def __fitness(self, x):
        
        f1 = x[0]
        g = 1 + 9 / (len(x) - 1) * np.sum(x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return (f1, f2)


    def __repr__(self):
        a = (np.round(self.x, 2), np.round(self.f1, 3), np.round(self.f2, 3))
        return repr(a)
    



class Population():
    
    def __init__(self, N_parents, N_childs, flg_selection='plus'):
        
        self.N_parents = N_parents
        self.N_childs = N_childs
        self.flg_selection = flg_selection
        
        self.parent = self.__get_init_population()


    def __get_init_population(self):
        
        initial_population = []
        for _ in range(self.N_parents):
            x = np.array((xi_high - xi_low) * np.random.random(N_dim) + xi_low)
            initial_population.append(Chromosome(x))
        return initial_population
    
    
    def crossover(self):

        self.childs = []
        while len(self.childs) < self.N_childs:
            parent_1, parent_2 = np.random.choice(self.parent, size=2)
            splitting_point = np.random.randint(low=1, high=N_dim)
            child_x = [*parent_1.x[:splitting_point], *parent_2.x[splitting_point:]]
            self.childs.append(Chromosome(child_x))
    
    
    def mutation(self):
        
        childs_mutated = []
        for child in self.childs:
            child_x_mutated = child.x + mut_sigma * np.random.normal(size=N_dim)
            child_x_mutated = np.where(child_x_mutated > xi_high, xi_high, child_x_mutated)
            child_x_mutated = np.where(child_x_mutated < xi_low, xi_low, child_x_mutated)
            childs_mutated.append(Chromosome(child_x_mutated))
        self.childs = childs_mutated
            
    
    def selection(self):
        
        if self.flg_selection == 'plus':
            self.childs.extend(self.parent)
            
        rank1 = self.__get_childs_best_rank()
        
        if flg_parents_nr_fixed:
            if len(rank1) > self.N_parents:
                self.survivors = [rank1[1], *np.random.choice(rank1[1:-1], N_parents-2), rank1[-1]]
            elif len(rank1) < self.N_parents:
                N_missing_childs = self.N_parents - len(rank1)
                rank2 = self.__get_childs_best_rank()
                self.survivors = [*rank1, *np.random.choice(rank2, N_missing_childs)]
            else:
                self.survivors = rank1
        else:
            self.survivors = rank1
            
        self.parent = self.survivors
    
    
    def __get_childs_best_rank(self):
                
        childs_sorted_f1 = sorted(self.childs, key=lambda Chromosome: (Chromosome.f1, Chromosome.f2))
        childs_best_rank = [childs_sorted_f1.pop(0)]
        best_f2 = childs_best_rank[0].f2
        self.childs.remove(childs_best_rank[0])
        
        while childs_sorted_f1:
            candidate = childs_sorted_f1.pop(0)
            if candidate.f2 < best_f2:
                childs_best_rank.append(candidate)
                best_f2 = candidate.f2
                self.childs.remove(candidate)
        return childs_best_rank
        
    

def GA(N_generations, GA_save_hist=1):
    
    population = Population(N_parents, N_childs, flg_selection=GA_selection_method)
    
    if GA_save_hist:
        population_hist = []
    
    for _ in range(N_generations):
        population.crossover()
        population.mutation()
        population.selection()
        if GA_save_hist:
            population_hist.append(deepcopy(population))
    
    if GA_save_hist:
        return population_hist
    else:
        return population



if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from plot_scripts import plot_font_size
    
    plot_font_size()
    
    def plotter(ax, population):
        all_childs_f1 = [child.f1 for child in population.childs]
        all_childs_f2 = [child.f2 for child in population.childs]
        
        best_childs_f1 = [survivor.f1 for survivor in population.survivors]
        best_childs_f2 = [survivor.f2 for survivor in population.survivors]
        
        ax.plot(all_childs_f1, all_childs_f2, marker='x', linestyle='None', color='black')
        ax.plot(best_childs_f1, best_childs_f2, marker='o', linestyle='None', color='red')
        
        max_f1 = np.max([*all_childs_f1, *best_childs_f1])
        max_f2 = np.max([*all_childs_f2, *best_childs_f2])
        
        return (max_f1, max_f2)
        
    
    fig, ax = plt.subplots()
    ax.set(title='Non-dominated Sorting', xlabel='$f_1$', ylabel='$f_2$')
    
    
    if GA_save_hist:
        population_hist = GA(N_generations)
        
        gen_counter = 1 #N_generations
        
        plotter(ax, population_hist[gen_counter-1])
        
        fig.subplots_adjust(bottom=0.2)
        ax_gen = plt.axes([0.25, 0.05, 0.65, 0.03])
        slider_gen = Slider(ax_gen, 'Generation', 1, N_generations, valfmt='%0.0f', valinit=gen_counter)
        
        def update(val):
            global gen_counter
            new_counter = int(round(slider_gen.val))
    
            if new_counter != gen_counter:
                ax.lines = []
                (max_f1, max_f2) = plotter(ax, population_hist[new_counter-1])
                gen_counter = new_counter
                if plot_update_ylim:
                    ax.set(ylim=[0,max_f2+0.5])
                fig.canvas.draw_idle()
        slider_gen.on_changed(update)

        
        
    else:
        population = GA(N_generations, GA_save_hist=0)
        plotter(ax, population)
               
    plt.show()



























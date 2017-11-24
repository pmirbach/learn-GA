# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:07:59 2017

@author: Philip
"""

import numpy as np
import benchmark_functions


def fitness_ext(x):
    return benchmark_functions.sphere(x)



class chromosome():
    
    def __init__(self, x):       
        self.x = x
        self.fitness = self.__fitness()
      
    def __fitness(self):    
        return fitness_ext(self.x)



#def get_parent_population(N):
#    population = []
#    while len(population) < N:
#        x = ( xi_high - xi_low ) * np.random.random( N_dim ) + xi_low
#        population.append( Chromosome(x) )
#    return population
#
#
#def muatation(childs):
#    childs_mutated = []
#    for i in range(len(childs)):
#        x_new = childs[i].x + mut_sigma * np.random.normal(size=N_dim)
#        childs_mutated.append( Chromosome(x_new) )
#    return childs_mutated


cnt_pop = 20
pop = []

while len(pop) < cnt_pop:
    x = np.random.randint(low=0,high=10,size=3)
    pop.append(chromosome(x))


print(pop[0].x,pop[0].fitness)

print(1/5)
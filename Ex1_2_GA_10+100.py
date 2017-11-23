#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:43:51 2017

@author: pmirbach
"""

import numpy as np


bitstring_len = 10

cnt_parents = 10
cnt_childs = 100
cnt_generations = 100

mut_sigma = 1. / bitstring_len

stat_iters = 20


def _fitness(childs, final):
    
    fit = np.sum(childs, axis=1)
    
    if final:
        print('OneMaxFitness:\n{}'.format(fit))
        print('Mittelwert:\n{}'.format(np.mean(fit)))
        print('Standardabweichung:\n{}'.format(np.std(fit)))
        print('Median:\n{}'.format(np.median(fit)))
        print('Min OneMax:\n{}'.format(np.min(fit)))
        print('Max OneMax:\n{}'.format(np.max(fit)))
    
    return fit


def _init_population(n):
    
    x = [0] * bitstring_len
    pars = [x for i in range(n)]
    
    return pars


def _two_point_crossover(parents):
    
    childs = []
    
    do = True
    
    while(do):
        
        rand_1 = np.random.randint(low=0, high=cnt_parents)
        rand_2 = np.random.randint(low=0, high=cnt_parents)
                
        if rand_1 != rand_2:
            
#            print(rand_1, rand_2)
            
            parent_1 = parents[rand_1]
            parent_2 = parents[rand_2]
            
#            print(parent_1, parent_2)
            
            cpoint_1 = np.random.randint(low=0, high=bitstring_len-1)
            cpoint_2 = np.random.randint(low=cpoint_1+1, high=bitstring_len)
            
            assert cpoint_1 < cpoint_2 and cpoint_1 != cpoint_2
            
            child_1 = [parent_1[i] if i <= cpoint_1 or i > cpoint_2 else parent_2[i] for i in range(bitstring_len)]
            child_2 = [parent_1[i] if i > cpoint_1 and i <= cpoint_2 else parent_2[i] for i in range(bitstring_len)]
            
            childs.append(child_1)
            childs.append(child_2)
        
        if len(childs) >= 100:
            do = False
    
    return childs


def _mutation(childs):
    
    childs_mutated = [[(bit+1)%2 if np.random.random() < mut_sigma else bit for bit in child] for child in childs]
    
    return childs_mutated


def _selection(childs, fitness):
    
    assert len(childs) == len(fitness)
    
    mapped_childs = list(zip(fitness, childs))
    
    sorted_mapped_childs = sorted(mapped_childs, reverse=True)
    top_childs = sorted_mapped_childs[:cnt_parents]
    
    return_childs = [child[1] for child in top_childs]
    
    return return_childs


for k in range(stat_iters):

    # Create new initial population
    parents = _init_population(cnt_parents)
    
    for i in range(cnt_generations):
                    
        # crossover with parents
        childs = _two_point_crossover(parents)
        
        # mutate new childrens
        childs_mutated = _mutation(childs)
        
        # include parent generation
        childs_mutated.extend(parents)
        
        # calculate fitness
        fitness = _fitness(childs_mutated, False)
                
        # select new parents for next generation
        parents = _selection(childs_mutated, fitness)
        
        # termination condition
        if np.max(fitness) == bitstring_len:
            break
    
    print('Fitness of run {}:'.format(k))
    _fitness(parents, True)
    print('-'*20)








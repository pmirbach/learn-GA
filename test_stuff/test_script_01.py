# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:07:59 2017

@author: Philip
"""

import numpy as np

class chromosome():
    
    
    def __init__(self, x):
        
        self.x = x
        self.fitness = self.__fitness()
    
    
    def __fitness(self):
        
        return np.dot(self.x, self.x)


a = chromosome([1,2,3])
print(a.x)
print(a.fitness)
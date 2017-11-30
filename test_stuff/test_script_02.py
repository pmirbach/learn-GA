
import matplotlib.pyplot as plt
import numpy as np



class A():
    
    def __init__(self,x):
        self.x = x


class B(A):
    
    def __init__(self,x,y):
        A.__init__(self,x)
        self.y = y


obj1 = A(2)
obj2 = B(1,2)

print(obj1.x)
print(obj2.x, obj2.y)













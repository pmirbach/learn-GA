
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




#a = [1.11,2.22,3.33]
#b = [4.44,5.55,6.66,7.77]
#
#format_str = 'x: {3.2f}, sum: {}'
#
#print(format_str.format(b, sum(b)))




g = 1
G = 5

print(g/G)

print([g]*3/[G]*3)



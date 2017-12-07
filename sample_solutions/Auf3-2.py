import numpy as np
import matplotlib.pyplot as plt
import math

N = 2
counter = 1000
sigma = 1.
tau = 2
mean = 3
sD = 1

Solution = np.random.uniform(-1,1,N)

def Sphere(Solution):
    return np.dot(Solution,Solution)

s = 0
R = 10
t = 0
fittness_Evolution = []
sigma_Evolution = []
for i in range(0,counter):
    
    t+=1
    print "Generation :",i,"fitness :",Sphere(Solution)
    
    #sigma_mutated = np.random.lognormal(mean,sigma,1)
    #sigma_mutated = sigma*np.random.lognormal(mean,sD,1
    #sigma_mutated = sigma*math.exp(tau*np.random.lognormal(mean,sD,1))
    sigma_mutated = sigma*math.exp(tau*np.random.standard_normal(1))
    Solution_mutated = Solution + sigma_mutated * np.random.standard_normal(N)
    sigma = sigma_mutated

    if Sphere(Solution_mutated) < Sphere(Solution):
                Solution = Solution_mutated
                s+=1
    fittness_Evolution.append(Sphere(Solution))

    if t%R==0:
            if float(s)/float(R)<1/5. :
                sigma /= tau
            else:
                sigma *= tau
            
            s=0
            
    sigma_Evolution.append(sigma)
    
pp=np.arange(0,counter,1)
plt.figure(1)
plt.subplot(221)
plt.plot(pp,fittness_Evolution)
plt.subplot(222)
plt.plot(pp,sigma_Evolution)
plt.grid(True)
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:52:58 2017

@author: Philip
"""

import numpy as np
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator


def sphere(x):
    """
    Computes the value of the Sphere benchmark function.
    """
    return np.dot(x,x)


def rosenbrock(x, a=1, b=100):
    """
    Computes the value of the Rosenbrock benchmark function.
    """
    res = 0
    for i in range(len(x)-1):
        res += b * ( x[i+1] - x[i]**2 )**2 + ( a - x[i] )**2
    return res


def rastrigin(x, A=10):
    """
    Computes the value of the Rastrigin benchmark function.
    x[i] in [-5.12, 5.12]
    """
    res = A * len(x)
    for i in range(len(x)):
        res += x[i]**2 - A * np.cos( 2 * np.pi * x[i] )
    return res


def schwefel(x):
    """
    Computes the value of the Schwefel benchmark function.
    x[i] in [-500, 500]
    """
    res = 418.9829 * len(x)
    for i in range(len(x)):
        res += x[i] * np.sin( np.sqrt( np.abs( x[i] ) ) )
    return res

def get_parameter_range(func_name):
    if func_name == "sphere":
        xi_low, xi_high = -10, 10
    elif func_name == "rosenbrock":
        xi_low, xi_high = -3, 3
    elif func_name == "rastrigin":
        xi_low, xi_high = -5.12, 5.12
    elif func_name == "schwefel":
        xi_low, xi_high = -500, 500
    else:
        assert "Not a valid function name"
    return (xi_low, xi_high)



if __name__ == "__main__":
    x = y = np.linspace(start=-500, stop=500, num=200)
    X, Y = np.meshgrid(x, y)
    
    #Zs = np.array( [sphere(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    Zs = np.array( [schwefel(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
    
    Z = np.reshape(Zs, X.shape)
    
    
    
    levels = MaxNLocator(nbins=30).tick_values(Z.min(), Z.max())
    #levels = MaxNLocator(nbins=30).tick_values(Z.min(), Z.min() + 10)
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    fig, ax0 = plt.subplots(nrows=1)
    
    im = ax0.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    
    ax0.plot()
    
    plt.show()





















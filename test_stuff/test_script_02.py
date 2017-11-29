"""
Layer images above one another using alpha blending
"""
#from __future__ import division

from benchmark_functions import rastrigin, get_parameter_range


import matplotlib.pyplot as plt
import numpy as np


fitness_fun = rastrigin

def penalty_fun( x ):
    g = np.sum(x) - 2
    return - np.min( [g, 0] )





def func3(x, y):
    return (1 - x/2 + x**5 + y**3)*np.exp(-(x**2 + y**2))

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects


xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
extent = xmin, xmax, ymin, ymax



fig, ax = plt.subplots()



Zs = np.array( [ fitness_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
Z = np.reshape(Zs, X.shape)

Penaltys = np.array( [ penalty_fun(x) for x in zip( np.ravel(X), np.ravel(Y) ) ] )
Penalty = np.reshape(Penaltys, X.shape)


im1 = ax.imshow(Z, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent,origin="lower")
im2 = ax.imshow(Penalty, cmap=plt.cm.Reds, alpha=.5, interpolation='bilinear',
                 extent=extent,origin="lower")

#im1 = ax.pcolormesh(X,Y,Z, cmap=plt.cm.gray)
#im2 = ax.pcolormesh(X,Y,Penalty, cmap=plt.cm.Reds, alpha=.2)
fig.colorbar(im1, ax=ax)

ax.plot([0,1,2],[0,1,2])

plt.show()



















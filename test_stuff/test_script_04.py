#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:30:10 2017

@author: pmirbach
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


N_images = 10
im_counter = 1

images = []
for _ in range(N_images):
    C=np.random.rand(500).reshape((20,25))
    images.append(C)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)
#t = np.arange(0.0, 1.0, 0.001)
#a0 = 5
#f0 = 3
#s = a0*np.sin(2*np.pi*f0*t)
#l, = plt.plot(t, s, lw=2, color='red')
#plt.axis([0, 1, -10, 10])
ax.imshow(images[im_counter-1])

axcolor = 'white'
axfreq = plt.axes([0.25, 0.05, 0.65, 0.03])

sfreq = Slider(axfreq, 'Generation', 1, N_images, valfmt='%0.0f', valinit=im_counter)


def update(val):
    global im_counter
    new_counter = int(round(sfreq.val))
    
    if new_counter != im_counter:
        ax.imshow(images[new_counter-1])
        im_counter = new_counter
        fig.canvas.draw_idle()
sfreq.on_changed(update)



#plt.imshow(images[-1])
plt.show()
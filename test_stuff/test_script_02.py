# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:32:08 2017

@author: Philip
"""

import matplotlib.pyplot as plt
import numpy as np

## Plot circle of radius 3.
#
#an = np.linspace(0, 2 * np.pi, 100)
#fig, axs = plt.subplots(2, 2)
#
#axs[0, 0].plot(3 * np.cos(an), 3 * np.sin(an))
#axs[0, 0].set_title('not equal, looks like ellipse', fontsize=10)
#
#axs[0, 1].plot(3 * np.cos(an), 3 * np.sin(an))
#axs[0, 1].axis('equal')
#axs[0, 1].set_title('equal, looks like circle', fontsize=10)
#
#axs[1, 0].plot(3 * np.cos(an), 3 * np.sin(an))
#axs[1, 0].axis('equal')
#axs[1, 0].axis([-3, 3, -3, 3])
#axs[1, 0].set_title('still a circle, even after changing limits', fontsize=10)
#
#axs[1, 1].plot(3 * np.cos(an), 3 * np.sin(an))
#axs[1, 1].set_aspect('equal', 'box')
#axs[1, 1].set_title('still a circle, auto-adjusted data limits', fontsize=10)
#
#fig.tight_layout()
#
#plt.show()




#fig, ax = plt.subplots()
#axes = [ax, ax.twinx(), ax.twinx()]
#
#fig.subplots_adjust(right=0.75)
#axes[-1].spines['right'].set_position(('axes',1.2))
#
#axes[-1].set_frame_on(True)
#axes[-1].patch.set_visible(False)
#
#colors = ('Green','Red','Blue')
#for ax, color in zip(axes, colors):
#    data = np.random.random(1)* np.random.random(10)
#    ax.plot(data, marker='o',linestyle='none', color=color)
#    ax.set_ylabel('{} Thing'.format(color),color=color)
#    ax.tick_params(axis='y', colors=color)
#axes[0].set_xlabel('X-axis')
#
#plt.show()





a = np.array([1,2,3])
print(a)
a = np.where( a <= 2.0, a, 2 )
print(a)




























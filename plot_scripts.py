# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:14:55 2017

@author: Philip
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap



def get_restrain_cmap():
    colors = [(1, 1, 1),(1, 0, 0),(1, 0, 0),(1, 0, 0)] 
    nbins = 100
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm


def plot_f_logf_sigma(ax, f, sigma):
    axes = [ax, ax.twinx(), ax.twinx()]
    axes[-1].spines['right'].set_position(('axes',1.2))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    
    plot_data = [f, np.log10(f), sigma]
    ylabels = ['Fitness', '$\log_{10}$ Fitness', 'Muationrate $\sigma$']
    linestyles = [':','-','--']
    colors = ['Blue', 'Red', 'Green']
    
    for ax, data, ylabel, linestyle, color in zip(axes, plot_data, ylabels, linestyles, colors):
        ax.plot(data, linestyle=linestyle, color=color)
        ax.set_ylabel(ylabel, color=color,fontsize=14)
        ax.tick_params(axis='y', colors=color)
        
    return axes


#def multi_scales_plot(ax, x_data, data_list):
#    """
#    Uses multiple instances of twinx to create multi scale axes.
#    Input: Axes the plot will be drawn in. x_data. List of tuples, each contains: (y_data, ylabel, linestyle, color)
#    x_data for must be equal!
#    Returns axes.
#    """
#    N = len(data_list)
#    axes = [ax]
#    for i in range(N-1):
#        axes.append(ax.twinx())
#    for ax, (data, ylabel) in zip(axes, data_list):
#        
#    
#    return axes





if __name__ == "__main__":
    fig, axes = plt.subplots()
    
    x_test = np.linspace(0,10,100)
    y1_test = np.exp(-x_test)
    y2_test = np.sin(x_test)
    test_data_list = [(y1_test, 'exp(-x)'), (y2_test,'sin(x)')]
    
#    print(zip([1,2],test_data_list))
    
    for num, (data, lab) in zip([1,2],test_data_list):
        print(num,data,lab)
#    ax = multi_scales_plot(axes, test_data_list)
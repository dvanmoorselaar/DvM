  
"""
Support functions for plotting

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def shifted_color_map(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. 
    Useful for data with a negative min and positive max and you want the middle of 
    the colormap's dynamic range to be at zero. Function sets the new start 
    (defaults to no lower offset; should be between 0 and 'midpoint'), 
    and the new stop (defaults to no upper ofset; should be between
    'midpoint` and 1.0)
    
    Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Arguments
    - - - - -
    cmap: The matplotlib colormap to be altered.
    min_val (float): new min value of the shifted color map 
    max_val (float): new max value of the shifted color map
    name (str): name of the shifted color map

    Returns
    - - - -

    newcmap : shifted colormap
    '''

    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    return newcmap
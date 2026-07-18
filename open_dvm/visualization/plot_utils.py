  
"""
Support functions for plotting

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

from typing import List, Optional, Tuple, Union

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

def resolve_cnd_diff_list(
        cnd_diff: Union[Tuple[str, str], List[Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
    """Normalize `cnd_diff` to a list of (cnd_a, cnd_b) contrast tuples.

    Accepts either a single pair (e.g. ('easy', 'hard')) or a list of
    pairs (e.g. [('easy', 'hard'), ('medium', 'hard')]) so callers can
    request one or several condition-difference tests in a single
    plotting call.
    """
    if cnd_diff is None:
        return []
    if isinstance(cnd_diff[0], str):
        return [cnd_diff]
    return list(cnd_diff)

def resolve_cnd_diff_colors(
        cnd_diff_color: Optional[Union[str, List[str]]],
        n_contrasts: int, default: str = 'grey'
    ) -> List[str]:
    """Resolve `cnd_diff_color` to one flat color per contrast.

    - None: a single contrast falls back to `default` ('grey', matching
      the toolbox's established manual convention); multiple contrasts
      auto-cycle through a default palette so they stay distinguishable.
    - str: applied to every contrast (the user's explicit choice, even
      if that makes multiple contrasts indistinguishable).
    - list of str: one color per contrast, must match `n_contrasts`.
    """
    if cnd_diff_color is None:
        if n_contrasts == 1:
            return [default]
        palette = list(mcolors.TABLEAU_COLORS.values())
        return [palette[i % len(palette)] for i in range(n_contrasts)]
    if isinstance(cnd_diff_color, str):
        return [cnd_diff_color] * n_contrasts
    if len(cnd_diff_color) != n_contrasts:
        raise ValueError(
            f"cnd_diff_color list length ({len(cnd_diff_color)}) must "
            f"match the number of cnd_diff contrasts ({n_contrasts})."
        )
    return list(cnd_diff_color)

def cnd_diff_point_colors(
        cnd_a: str, cnd_b: str, y1: np.ndarray, y2: np.ndarray,
        cnds: Optional[list], colors: Optional[list]
    ) -> Optional[np.ndarray]:
    """Auto-derive per-timepoint marker colors for a condition-difference
    contrast, using each condition's own assigned plot color.

    Returns an array (one color per timepoint, `cnd_a`'s color where its
    mean is higher, else `cnd_b`'s) if both conditions were assigned a
    color in `cnds`/`colors`, else None (caller falls back to a flat
    color -- no colors are available to alternate between).
    """
    if cnds is None or colors is None:
        return None
    cnds = list(cnds)
    if cnd_a not in cnds or cnd_b not in cnds:
        return None
    idx_a, idx_b = cnds.index(cnd_a), cnds.index(cnd_b)
    if idx_a >= len(colors) or idx_b >= len(colors):
        return None
    return np.where(y1.mean(axis=0) >= y2.mean(axis=0),
                     colors[idx_a], colors[idx_b])
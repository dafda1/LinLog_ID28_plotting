# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:20:40 2025

@author: dafda1
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

#%%

def ID28_style_cmap_values (intensity, cutoff = None, logpar = None,
                            reverse = False):
    if reverse:
        if cutoff is not None and logpar is not None:
            intensity_real = np.where(intensity <= 0,
                                      cutoff*(intensity + 1),
                                      cutoff*(logpar**intensity))
            
            return intensity_real
            
        else:
            raise ValueError("If 'reverse', must provide both 'cutoff' and 'logpar'.")
    
    else:
        if cutoff is None and logpar is None:
            raise ValueError("Must provide either 'cutoff' or 'logpar'.")
        elif cutoff is not None and logpar is not None:
            raise ValueError("Must provide EITHER 'cutoff' or 'logpar', not both.")
        
        Imax = np.max(intensity)
        
        if cutoff is None:
            print("Calculating cutoff from logpar and maximum intensity.")
            cutoff = Imax*1.0/logpar
            print(f"cutoff = {cutoff}")
        
        elif logpar is None:
            print("Calculating logpar from logpar and maximum intensity.")
            logpar = Imax*1.0/cutoff
            print(f"logpar = {logpar}")
        
        relative_intensity = intensity*1.0/cutoff
        
        Iplot = np.where(intensity <= cutoff,
                         relative_intensity - 1,
                         np.log(relative_intensity)/np.log(logpar))
        
        return Iplot, cutoff, logpar
    
def get_ID28_style_cmap (linemap = plt.cm.binary,
                         heatmap = plt.cm.hot,
                         resolution = 128):
    
    colors1 = linemap(np.linspace(0, 1, resolution))
    colors2 = heatmap(np.linspace(0, 1, resolution))
    
    colors = np.vstack((colors1, colors2))
    
    return LinearSegmentedColormap.from_list("ID28", colors)

def generate_cmap_tick_marks (cutoff, logpar):
    "assumes cutoff is roughly 1000 and last order is roughly 1e6 (for now, wip)"
    
    pre_cutoff_intensities = np.linspace(0, cutoff, 5)
    
    minlog = int(np.log10(cutoff)) + 1
    maxlog = int(np.log10(logpar*cutoff))
    
    post_cutoff_logs = np.arange(minlog, maxlog + 1)
    
    post_cutoff_intensities = np.append(10**post_cutoff_logs, logpar*cutoff)
    
    all_intensities = np.concatenate((pre_cutoff_intensities,
                                      post_cutoff_intensities))
    
    tick_labels = ["%.0f" % val for val in pre_cutoff_intensities] +\
                  [f"$10^{val}$" % val for val in post_cutoff_logs]
    
    tick_positions, _, _ = ID28_style_cmap_values(all_intensities, cutoff = cutoff)
    
    return tick_positions[:-1], tick_labels

def generate_colorbar (mappable, cutoff, logpar):
    ticks, labels = generate_cmap_tick_marks(cutoff, logpar)
    
    cbar = plt.colorbar(mappable)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    
    cbar.set_label("Intensity (arb. units)")
    
    return cbar